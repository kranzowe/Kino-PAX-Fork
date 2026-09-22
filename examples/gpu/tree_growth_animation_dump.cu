// ========================================================================
// TREE GROWTH ANIMATION DUMP
//
// Same tool as tree_checkpoint_dump.cu (same three planners, same fixed
// tunables, same tree-CSV sparsification, same stopping conditions, same
// wall-clock-checkpoint mechanism) -- the ONLY difference is the checkpoint
// schedule: instead of 6 hand-picked points (50/100/150/1000/2000/4000ms),
// this dumps a checkpoint every CHECKPOINT_STEP_MS (30ms) out to
// CHECKPOINT_MAX_MS (3000ms), i.e. 100 checkpoints, so plots/tree_growth_animation.py
// can render an actual animation of the tree filling in over time instead of
// a handful of static frames.
//
//   KPAX             pure explorer reference
//   KinoPaxPlus       pure optimizer reference
//   CountingStars     bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15,
//                     cost_frac 0.75, hopelessGuard ON (v3.5) -- the exact
//                     canonical point paper_benchmark_v2.cu uses for this planner
//
// Per-iteration timing uses the same cudaEvent_t/plannerMs pattern as
// paper_benchmark_v2.cu's benchmark*() functions (propagate+update only inside
// the timed window). "Solution found?" differs by planner family exactly as
// in tree_checkpoint_dump.cu -- see that file's header comment for the full
// explanation of KPAX's persistent bestGoalIdx/bestCost tracking vs.
// KinoPaxPlus/CountingStars's h_minCost_ + findGoalIdxByScan.
//
// Tree CSVs are SPARSIFIED exactly as in tree_checkpoint_dump.cu: only
// TREE_LEAF_SAMPLE_FRAC (1%, see below) of the tree's LEAF nodes are kept, each with its
// FULL ancestor chain back to root. NOT changed here -- same fraction, same
// algorithm.
//
// A (planner, checkpoint) pair can be MISSING from the output entirely -- no tree/traj CSV, no
// row in the stats file below -- if that planner's own loop exits before reaching that
// checkpoint's wall-clock threshold. There is no meaningful iteration cap here
// (MAX_LOOP_ITERATIONS is a generous safety net, not a real stopping criterion); the two actual
// causes are printed to the console when they happen: "stopped early: tree full" (h_treeSize_
// hit MAX_TREE_SIZE) or "stopped early: h_propIterations_ == 0" (the planner itself decided it
// has no room left to propagate). Either way the tree filled its capacity before the wall-clock
// checkpoint was reached -- not a data/plotting bug.
//
// {env}_checkpoint_stats.csv (one row per (planner, checkpoint) ACTUALLY reached) records the
// REAL, non-sparsified tree_size and its percentage of MAX_TREE_SIZE at that checkpoint -- see
// writeCheckpointStats(). This is the only place that percentage is available: the tree CSVs
// themselves are sparsified (see above), so their row count is not the true tree size.
//
// Render the output with plots/tree_growth_animation.py.
// ========================================================================
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "planners/KPAX.cuh"
#include "planners/KinoPaxPlus.cuh"
#include "planners/CountingStars.cuh"

// CountingStars' canonical operating point -- exact same values as
// paper_benchmark_v2.cu's CS_BUFFER_SLOPE/CS_BUFFER_FLOOR/CS_EXPLORE_FRAC/
// CS_COST_FRAC/CS_HOPELESS_GUARD/CS_RAMP_FILL_ITERS. Unchanged from
// tree_checkpoint_dump.cu.
static const float CS_BUFFER_SLOPE   = 1.2f;
static const float CS_BUFFER_FLOOR   = 0.4f;
static const float CS_EXPLORE_FRAC   = 0.15f;
static const float CS_COST_FRAC      = 0.75f;
static const bool  CS_HOPELESS_GUARD = true;
static const int   CS_RAMP_FILL_ITERS = 700;

// Fraction of each tree's LEAF nodes (see writeTreeCSV) kept in the tree CSVs. Lower than
// tree_checkpoint_dump.cu's 4% (this tool's own, separate constant -- the static checkpoint dump
// is untouched) because 100 checkpoints' worth of tree CSVs, each re-rendered as a full
// scatter+edge plot, adds up fast; 1% keeps the animation responsive to render while still
// preserving complete, connected leaf-to-root paths (see writeTreeCSV's ancestor-chain logic).
static const float TREE_LEAF_SAMPLE_FRAC = 0.01f;

// ========================================================================
// One host-side snapshot of a planner's tree, copied once per checkpoint and
// reused for both the tree CSV and (if a solution exists) the trajectory CSV --
// avoids a second device->host round trip for the same data.
// ========================================================================
struct TreeSnapshot
{
    std::vector<float> samples;   // [treeSize * SAMPLE_DIM]
    std::vector<int>   parents;   // [treeSize]
    std::vector<float> costs;     // [treeSize] -- cumulative cost-from-root per node
    int treeSize;
};

TreeSnapshot copyTreeToHost(float* d_treeSamples_ptr, int* d_parents_ptr,
                            float* d_costs_ptr, int treeSize)
{
    TreeSnapshot snap;
    snap.treeSize = treeSize;
    snap.samples.resize((size_t)treeSize * SAMPLE_DIM);
    snap.parents.resize(treeSize);
    snap.costs.resize(treeSize);
    cudaMemcpy(snap.samples.data(), d_treeSamples_ptr,
               (size_t)treeSize * SAMPLE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(snap.parents.data(), d_parents_ptr, treeSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(snap.costs.data(), d_costs_ptr, treeSize * sizeof(float), cudaMemcpyDeviceToHost);
    return snap;
}

// One row per KEPT node: idx,x,y,z,vx,vy,vz,parent,cost (Model-1-shaped state
// columns). Only leafSampleFrac of the tree's LEAF nodes (nodes no other node
// names as parent) are kept, each with its FULL ancestor chain back to root --
// this keeps every retained path complete/connected rather than a random
// scatter of disconnected dots. Kept nodes are renumbered densely (0..K-1),
// preserving relative insertion order, with parent references remapped to
// match. Unchanged from tree_checkpoint_dump.cu.
void writeTreeCSV(const TreeSnapshot& snap, const std::string& path, float leafSampleFrac)
{
    int n = snap.treeSize;
    std::vector<uint8_t> keep(n, 1);   // default: keep everything

    if(leafSampleFrac < 1.0f && n > 0)
    {
        std::vector<uint8_t> hasChild(n, 0);
        for(int i = 0; i < n; i++)
        {
            int p = snap.parents[i];
            if(p >= 0) hasChild[p] = 1;
        }
        std::vector<int> leaves;
        for(int i = 0; i < n; i++)
            if(!hasChild[i]) leaves.push_back(i);

        int nSample = std::max(1, (int)std::round(leafSampleFrac * (double)leaves.size()));
        std::fill(keep.begin(), keep.end(), 0);

        if(nSample >= (int)leaves.size())
        {
            std::fill(keep.begin(), keep.end(), 1);   // sampling everything anyway
        }
        else
        {
            // Evenly spaced over the leaf list (leaves is built by a forward scan, so it is
            // already in ascending idx/insertion order) -- matches this repo's established
            // "evenly spaced, not random" subsampling convention. Early-exit once a chain hits an
            // already-kept node: everything above it was already marked by a previous leaf's
            // walk, so this both saves work and deduplicates automatically.
            for(int k = 0; k < nSample; k++)
            {
                size_t li = (nSample == 1) ? 0
                          : (size_t)std::llround((double)k * (double)(leaves.size() - 1) / (double)(nSample - 1));
                int cur = leaves[li];
                while(cur >= 0 && !keep[cur])
                {
                    keep[cur] = 1;
                    cur = snap.parents[cur];
                }
            }
        }
    }

    std::vector<int> remap(n, -1);
    int newIdx = 0;
    for(int i = 0; i < n; i++)
        if(keep[i]) remap[i] = newIdx++;

    std::ofstream file(path);
    file << "idx,x,y,z,vx,vy,vz,parent,cost\n";
    file << std::fixed << std::setprecision(6);
    for(int i = 0; i < n; i++)
    {
        if(!keep[i]) continue;
        const float* s = &snap.samples[(size_t)i * SAMPLE_DIM];
        int newParent = (snap.parents[i] >= 0) ? remap[snap.parents[i]] : -1;
        file << remap[i];
        for(int d = 0; d < 6; d++) file << "," << s[d];
        file << "," << newParent << "," << snap.costs[i] << "\n";
    }
    file.close();
}

// Walks parents[] from goalIdx back to root, collecting each node's state + its
// already-cumulative-from-root cost, then reverses so the CSV reads root-first/goal-last. Writes
// a header-only (0 row) CSV when goalIdx < 0. Unchanged from tree_checkpoint_dump.cu.
void writeTrajectoryCSV(const TreeSnapshot& snap, int goalIdx, const std::string& path)
{
    std::ofstream file(path);
    file << "step,x,y,z,vx,vy,vz,cost\n";
    file << std::fixed << std::setprecision(6);

    if(goalIdx < 0 || goalIdx >= snap.treeSize)
    {
        file.close();
        return;
    }

    std::vector<int> chain;   // goal -> root order
    int cur = goalIdx;
    while(cur >= 0)
    {
        chain.push_back(cur);
        cur = snap.parents[cur];
    }

    int step = 0;
    for(auto it = chain.rbegin(); it != chain.rend(); ++it)
    {
        const float* s = &snap.samples[(size_t)(*it) * SAMPLE_DIM];
        file << step;
        for(int d = 0; d < 6; d++) file << "," << s[d];
        file << "," << snap.costs[*it] << "\n";
        step++;
    }
    file.close();
}

// Host-side rescan for KinoPaxPlus/CountingStars: neither exposes a goal tree
// index directly, only a best-cost scalar (h_minCost_). Unchanged from
// tree_checkpoint_dump.cu.
int findGoalIdxByScan(bool* d_goalSet_ptr, const TreeSnapshot& snap, float minCost)
{
    if(!(minCost < MAX_FLOAT)) return -1;

    std::vector<uint8_t> h_goalSet(snap.treeSize);
    cudaMemcpy(h_goalSet.data(), d_goalSet_ptr, snap.treeSize * sizeof(bool), cudaMemcpyDeviceToHost);

    for(int i = 0; i < snap.treeSize; i++)
        if(h_goalSet[i] != 0 && snap.costs[i] == minCost)
            return i;
    return -1;   // shouldn't happen once minCost < MAX_FLOAT, but stay defensive
}

std::string checkpointTreePath(const std::string& vizDir, const std::string& env,
                               const std::string& token, int ms)
{
    std::ostringstream ss;
    ss << vizDir << "/" << env << "_" << token << "_t" << ms << "ms_tree.csv";
    return ss.str();
}

std::string checkpointTrajPath(const std::string& vizDir, const std::string& env,
                               const std::string& token, int ms)
{
    std::ostringstream ss;
    ss << vizDir << "/" << env << "_" << token << "_t" << ms << "ms_traj.csv";
    return ss.str();
}

// Same shape as tree_checkpoint_dump.cu's writeVizMeta.
void writeVizMeta(const std::string& path, const float* h_initial, const float* h_goal)
{
    std::ofstream file(path);
    file << "W_DIM,W_R1_LENGTH,V_R1_LENGTH,W_MIN,W_MAX,V_MIN,V_MAX,STATE_DIM,SAMPLE_DIM,"
         << "start_x,start_y,start_z,goal_x,goal_y,goal_z\n";
    file << std::fixed << std::setprecision(6)
         << W_DIM << "," << W_R1_LENGTH << "," << V_R1_LENGTH << ","
         << W_MIN << "," << W_MAX << "," << V_MIN << "," << V_MAX << ","
         << STATE_DIM << "," << SAMPLE_DIM << ","
         << h_initial[0] << "," << h_initial[1] << "," << h_initial[2] << ","
         << h_goal[0] << "," << h_goal[1] << "," << h_goal[2] << "\n";
    file.close();
    printf("  [viz] meta written: %s\n", path.c_str());
}

// Checkpoint schedule: every CHECKPOINT_STEP_MS out to CHECKPOINT_MAX_MS (inclusive) -- this is
// the one deliberate difference from tree_checkpoint_dump.cu's 6 hand-picked points. Built once
// in main() into a std::vector (100 entries at the default 30/3000 split) rather than a giant
// literal array.
static const float CHECKPOINT_STEP_MS = 30.0f;
static const float CHECKPOINT_MAX_MS  = 3000.0f;
static const int   MAX_LOOP_ITERATIONS = 200000;   // generous safety cap, not the stopping criterion --
                                                    // bumped vs. tree_checkpoint_dump.cu's 100000 only
                                                    // because 100 fine-grained checkpoints need more
                                                    // total iterations to all be reached; the two real
                                                    // stopping conditions (tree-full / propIterations==0)
                                                    // are unchanged.

std::vector<float> buildCheckpointSchedule(float stepMs, float maxMs)
{
    std::vector<float> ms;
    for(float t = stepMs; t <= maxMs + 1e-3f; t += stepMs)
        ms.push_back(t);
    return ms;
}

// Appended to once per (planner, checkpoint) actually reached -- see writeCheckpointStats().
// Unchanged from tree_checkpoint_dump.cu.
void writeCheckpointStats(std::ofstream& statsFile, const std::string& token, int checkpointMs,
                          int iteration, unsigned int treeSize, bool solved)
{
    float pctFilled = 100.0f * (float)treeSize / (float)MAX_TREE_SIZE;
    statsFile << token << "," << checkpointMs << "," << iteration << "," << treeSize << ","
              << MAX_TREE_SIZE << "," << std::fixed << std::setprecision(4) << pctFilled << ","
              << (solved ? 1 : 0) << "\n";
    statsFile.flush();   // each planner's block can exit early/abnormally; don't lose prior rows
}

// ========================================================================
// KPAX: has graph_, goal signal is h_pathToGoal_ (direct tree index), tracked
// into a persistent best-known (bestGoalIdx/bestCost) across iterations.
// Unchanged from tree_checkpoint_dump.cu except taking checkpointsMs by
// reference instead of the fixed-size CHECKPOINTS_MS array.
// ========================================================================
void runKPAXCheckpoints(KPAX& planner, const std::string& token, const std::string& env,
                        const std::string& vizDir, float* h_initial, float* h_goal,
                        float* d_obstacles, uint numObstacles, std::ofstream& statsFile,
                        const std::vector<float>& checkpointsMs)
{
    printf("\n--- %s ---\n", token.c_str());
    planner.resetPlanner(h_initial, h_goal);

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);
    float plannerMs = 0.0f;
    float iterMs    = 0.0f;
    int zero = 0;
    size_t nextCheckpoint = 0;
    int itr = 0;

    int   bestGoalIdx = -1;
    float bestCost    = MAX_FLOAT;

    while(itr < MAX_LOOP_ITERATIONS && nextCheckpoint < checkpointsMs.size())
    {
        itr++;
        planner.h_itr_++;

        // Reset pathToGoal before each iteration so a later, different solution
        // isn't missed (same pattern as paper_benchmark_v2.cu's benchmarkKPAX).
        cudaMemcpy(planner.d_pathToGoal_ptr_, &zero, sizeof(int), cudaMemcpyHostToDevice);
        planner.h_pathToGoal_ = 0;

        cudaEventRecord(iterStart);
        planner.propagateFrontier(d_obstacles, numObstacles);
        planner.graph_.updateVertices();
        planner.updateFrontier();
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&iterMs, iterStart, iterStop);
        plannerMs += iterMs;

        if(planner.h_pathToGoal_ != 0)
        {
            float pathCost;
            cudaMemcpy(&pathCost, &planner.d_treeSampleCosts_ptr_[planner.h_pathToGoal_],
                       sizeof(float), cudaMemcpyDeviceToHost);
            if(pathCost < bestCost)
            {
                bestCost    = pathCost;
                bestGoalIdx = planner.h_pathToGoal_;
            }
        }

        while(nextCheckpoint < checkpointsMs.size() && plannerMs >= checkpointsMs[nextCheckpoint])
        {
            int ms = (int)checkpointsMs[nextCheckpoint];
            TreeSnapshot snap = copyTreeToHost(planner.d_treeSamples_ptr_,
                                               planner.d_treeSamplesParentIdxs_ptr_,
                                               planner.d_treeSampleCosts_ptr_, planner.h_treeSize_);
            writeTreeCSV(snap, checkpointTreePath(vizDir, env, token, ms), TREE_LEAF_SAMPLE_FRAC);
            writeTrajectoryCSV(snap, bestGoalIdx, checkpointTrajPath(vizDir, env, token, ms));
            writeCheckpointStats(statsFile, token, ms, itr, planner.h_treeSize_, bestGoalIdx >= 0);
            if(ms % 500 == 0)
                printf("  t=%dms: itr=%d tree=%u %s\n", ms, itr, planner.h_treeSize_,
                       bestGoalIdx >= 0 ? "SOLVED" : "no solution yet");
            nextCheckpoint++;
        }

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) { printf("  stopped early: tree full\n"); break; }
        // NOTE: no h_propIterations_==0 break here, unlike the other two loops below -- see
        // tree_checkpoint_dump.cu's header comment for why (stale/0 for KPAX during normal
        // operation; checking it here would exit the loop on essentially iteration 1).
    }

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
}

// ========================================================================
// KinoPaxPlus: no graph_. Goal signal is h_minCost_; goalIdx recovered via
// findGoalIdxByScan. Unchanged from tree_checkpoint_dump.cu except taking
// checkpointsMs by reference.
// ========================================================================
void runKinoPaxPlusCheckpoints(KinoPaxPlus& planner, const std::string& token, const std::string& env,
                               const std::string& vizDir, float* h_initial, float* h_goal,
                               float* d_obstacles, uint numObstacles, std::ofstream& statsFile,
                               const std::vector<float>& checkpointsMs)
{
    printf("\n--- %s ---\n", token.c_str());
    planner.resetPlanner(h_initial, h_goal);

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);
    float plannerMs = 0.0f;
    float iterMs    = 0.0f;
    size_t nextCheckpoint = 0;
    int itr = 0;

    while(itr < MAX_LOOP_ITERATIONS && nextCheckpoint < checkpointsMs.size())
    {
        itr++;
        planner.h_itr_++;

        cudaEventRecord(iterStart);
        planner.propagateFrontier(d_obstacles, numObstacles);
        if(planner.h_propIterations_ == 0) { printf("  stopped early: h_propIterations_ == 0\n"); break; }
        planner.updateFrontier();
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&iterMs, iterStart, iterStop);
        plannerMs += iterMs;

        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);

        while(nextCheckpoint < checkpointsMs.size() && plannerMs >= checkpointsMs[nextCheckpoint])
        {
            int ms = (int)checkpointsMs[nextCheckpoint];
            TreeSnapshot snap = copyTreeToHost(planner.d_treeSamples_ptr_,
                                               planner.d_treeSamplesParentIdxs_ptr_,
                                               planner.d_treeSampleCosts_ptr_, planner.h_treeSize_);
            writeTreeCSV(snap, checkpointTreePath(vizDir, env, token, ms), TREE_LEAF_SAMPLE_FRAC);
            int goalIdx = findGoalIdxByScan(planner.d_goalSet_ptr_, snap, planner.h_minCost_);
            writeTrajectoryCSV(snap, goalIdx, checkpointTrajPath(vizDir, env, token, ms));
            writeCheckpointStats(statsFile, token, ms, itr, planner.h_treeSize_, goalIdx >= 0);
            if(ms % 500 == 0)
                printf("  t=%dms: itr=%d tree=%u %s\n", ms, itr, planner.h_treeSize_,
                       goalIdx >= 0 ? "SOLVED" : "no solution yet");
            nextCheckpoint++;
        }

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) { printf("  stopped early: tree full\n"); break; }
    }

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
}

// ========================================================================
// CountingStars: has graph_ but NEVER calls graph_.updateVertices(). Same
// h_minCost_/findGoalIdxByScan mechanism as KinoPaxPlus. Fixed canonical
// tunables set before resetPlanner() (which does not touch them). Unchanged
// from tree_checkpoint_dump.cu except taking checkpointsMs by reference.
// ========================================================================
void runCountingStarsCheckpoints(CountingStars& planner, const std::string& token, const std::string& env,
                                 const std::string& vizDir, float* h_initial, float* h_goal,
                                 float* d_obstacles, uint numObstacles, std::ofstream& statsFile,
                                 const std::vector<float>& checkpointsMs)
{
    planner.h_fillIters_     = CS_RAMP_FILL_ITERS;
    planner.h_bufferSlope_   = CS_BUFFER_SLOPE;
    planner.h_bufferFloor_   = CS_BUFFER_FLOOR;
    planner.h_exploreFrac_   = CS_EXPLORE_FRAC;
    planner.h_costFrac_      = CS_COST_FRAC;
    planner.h_hopelessGuard_ = CS_HOPELESS_GUARD;

    printf("\n--- %s ---\n", token.c_str());
    planner.resetPlanner(h_initial, h_goal);

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);
    float plannerMs = 0.0f;
    float iterMs    = 0.0f;
    size_t nextCheckpoint = 0;
    int itr = 0;

    while(itr < MAX_LOOP_ITERATIONS && nextCheckpoint < checkpointsMs.size())
    {
        itr++;
        planner.h_itr_++;

        cudaEventRecord(iterStart);
        planner.propagateFrontier(d_obstacles, numObstacles);
        // NO graph_.updateVertices() -- see header comment above.
        planner.updateFrontier();
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&iterMs, iterStart, iterStop);
        plannerMs += iterMs;

        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);

        while(nextCheckpoint < checkpointsMs.size() && plannerMs >= checkpointsMs[nextCheckpoint])
        {
            int ms = (int)checkpointsMs[nextCheckpoint];
            TreeSnapshot snap = copyTreeToHost(planner.d_treeSamples_ptr_,
                                               planner.d_treeSamplesParentIdxs_ptr_,
                                               planner.d_treeSampleCosts_ptr_, planner.h_treeSize_);
            writeTreeCSV(snap, checkpointTreePath(vizDir, env, token, ms), TREE_LEAF_SAMPLE_FRAC);
            int goalIdx = findGoalIdxByScan(planner.d_goalSet_ptr_, snap, planner.h_minCost_);
            writeTrajectoryCSV(snap, goalIdx, checkpointTrajPath(vizDir, env, token, ms));
            writeCheckpointStats(statsFile, token, ms, itr, planner.h_treeSize_, goalIdx >= 0);
            if(ms % 500 == 0)
                printf("  t=%dms: itr=%d tree=%u %s\n", ms, itr, planner.h_treeSize_,
                       goalIdx >= 0 ? "SOLVED" : "no solution yet");
            nextCheckpoint++;
        }

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) { printf("  stopped early: tree full\n"); break; }
        if(planner.h_propIterations_ == 0) { printf("  stopped early: h_propIterations_ == 0\n"); break; }
    }

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
}

int main(int argc, char* argv[])
{
    std::string obstaclePath = (argc > 1) ? argv[1] : "../include/config/obstacles/zigzag/obstacles.csv";
    std::string envName      = (argc > 2) ? argv[2] : "zigzag";

    std::string vizDir = "Data/Viz/TreeGrowthAnimation";
    std::filesystem::create_directories(vizDir);

    std::vector<float> checkpointsMs = buildCheckpointSchedule(CHECKPOINT_STEP_MS, CHECKPOINT_MAX_MS);

    // Start/goal in workspace coordinates, matching the benchmarks' main() and tree_checkpoint_dump.cu.
    float h_initial[SAMPLE_DIM] = {0};
    float h_goal[SAMPLE_DIM]    = {0};
    h_initial[0] = W_MIN + 0.1f * W_SIZE;
    h_initial[1] = W_MIN + 0.08f * W_SIZE;
    h_initial[2] = W_MIN + 0.05f * W_SIZE;
    h_goal[0]    = W_MIN + 0.8f * W_SIZE;
    h_goal[1]    = W_MIN + 0.95f * W_SIZE;
    h_goal[2]    = W_MIN + 0.9f * W_SIZE;

    printf("=======================================================\n");
    printf("    TREE GROWTH ANIMATION DUMP\n");
    printf("=======================================================\n");
    printf("NUM_R1_REGIONS: %d\n", NUM_R1_REGIONS);
    printf("MAX_TREE_SIZE:  %d\n", MAX_TREE_SIZE);
    printf("Cost metric:    %s (COST_MODE=%d)\n",
           (COST_MODE == 2) ? "path time" : (COST_MODE == 1) ? "control effort" : "workspace path length",
           COST_MODE);
    printf("Obstacle file:  %s\n", obstaclePath.c_str());
    printf("Environment:    %s\n", envName.c_str());
    printf("Checkpoints:    every %.0fms out to %.0fms (%zu checkpoints)\n",
           CHECKPOINT_STEP_MS, CHECKPOINT_MAX_MS, checkpointsMs.size());
    printf("Tree sampling:  %.0f%% of leaf trajectories per checkpoint\n", 100.0f * TREE_LEAF_SAMPLE_FRAC);
    printf("Output:         %s\n", vizDir.c_str());
    printf("=======================================================\n");

    int numObstacles;
    float* d_obstacles;
    std::vector<float> obstacles = readObstaclesFromCSV(obstaclePath, numObstacles, W_DIM);
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(), numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);
    printf("Loaded %d obstacles from %s\n", numObstacles, obstaclePath.c_str());

    writeVizMeta(vizDir + "/meta.csv", h_initial, h_goal);

    std::ofstream statsFile(vizDir + "/" + envName + "_checkpoint_stats.csv");
    statsFile << "token,checkpoint_ms,iteration,tree_size,max_tree_size,pct_filled,solved\n";

    {
        KPAX planner;
        runKPAXCheckpoints(planner, "KPAX", envName, vizDir, h_initial, h_goal, d_obstacles, numObstacles, statsFile, checkpointsMs);
    }
    {
        KinoPaxPlus planner;
        runKinoPaxPlusCheckpoints(planner, "KinoPaxPlus", envName, vizDir, h_initial, h_goal, d_obstacles, numObstacles, statsFile, checkpointsMs);
    }
    {
        CountingStars planner;
        runCountingStarsCheckpoints(planner, "CountingStars", envName, vizDir, h_initial, h_goal, d_obstacles, numObstacles, statsFile, checkpointsMs);
    }
    statsFile.close();
    printf("Checkpoint stats written: %s\n", (vizDir + "/" + envName + "_checkpoint_stats.csv").c_str());

    cudaFree(d_obstacles);

    printf("\n=======================================================\n");
    printf("    DUMP COMPLETE\n");
    printf("=======================================================\n");
    printf("Trees/trajectories in: %s\n", vizDir.c_str());
    printf("Plot with: plots/tree_growth_animation.py\n");
    printf("=======================================================\n");

    return 0;
}
