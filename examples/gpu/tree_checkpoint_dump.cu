// ========================================================================
// TREE CHECKPOINT DUMP
//
// Dumps the planner's full tree, AND its solution trajectory (with cost) if one
// exists yet, at fixed WALL-CLOCK-TIME checkpoints -- 50ms, 100ms, 150ms, 1000ms,
// 4000ms of cumulative planner-only time -- for three planners on one environment:
//
//   KPAX             pure explorer reference
//   KinoPaxPlus       pure optimizer reference
//   CountingStars     bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15,
//                     cost_frac 0.75, hopelessGuard ON (v3.5) -- the exact
//                     canonical point paper_benchmark_v2.cu uses for this planner
//
// This is the time-checkpoint sibling of tree_growth_dump.cu (which dumps after
// each of the first 8 ITERATIONS, with no timing and no solution capture). That
// file is untouched; this is a separate, additive tool.
//
// Per-iteration timing uses the same cudaEvent_t/plannerMs pattern as
// paper_benchmark_v2.cu's benchmark*() functions (propagate+update only inside
// the timed window). "Solution found?" differs by planner family:
//   KPAX: h_pathToGoal_ is the goal tree index directly once non-zero, but it is
//     RESET TO 0 EVERY ITERATION (so a later, different solution isn't missed) --
//     since checkpoints are only inspected every so many iterations, the raw flag
//     alone would silently lose a solution found between checkpoints the moment
//     the next iteration's reset zeroes it again. A persistent best-known
//     (bestGoalIdx/bestCost, ratcheted every iteration h_pathToGoal_ is non-zero)
//     is tracked across the whole run instead, and THAT is what gets checked at
//     each checkpoint. Also note: KPAX's loop deliberately does NOT break on
//     h_propIterations_==0 (unlike the other two below) -- for KPAX that field is
//     only ever assigned inside propagateFrontier()'s near-full-tree branch
//     (KPAX.cu:254-271) and stays stale/0 for the whole normal, plenty-of-room
//     phase of a run, so checking it here made KPAX's loop exit on essentially
//     iteration 1 every time. paper_benchmark_v2.cu's own benchmarkKPAX never
//     checks it either -- only tree-full and the timeout.
//   KinoPaxPlus / CountingStars: h_pathToGoal_ is not wired up for these; the
//     real signal is h_minCost_ (< MAX_FLOAT), which the planner itself
//     maintains as a genuine running minimum across the whole run (monotonic,
//     never reset by this tool) -- so, unlike KPAX, no separate persistent
//     tracking is needed here; h_minCost_ at checkpoint time already reflects
//     the best-ever-found cost. Neither planner exposes a host-side goal index
//     directly though, so one is recovered by scanning d_goalSet_/
//     d_treeSampleCosts_ on the host for the node where goalSet[i] &&
//     costs[i] == h_minCost_ -- the same cost-equality semantics their own
//     internal getControlPathToGoal() kernels use, just done host-side instead
//     of relying on that kernel's output buffer (which never reports how many
//     entries it validly wrote). d_goalSet_ is only ever reset once, at
//     resetPlanner(), and minCost only ever decreases (atomicMinFloat), so a
//     stale `true` flag can never false-match a later, lower h_minCost_.
//
// Tree CSVs are SPARSIFIED: a full tree can reach millions of nodes (COST_MODE 2
// / TIME's weak per-region cost signal lets KinoPaxPlus/CountingStars admit far
// more nodes per R1 region than effort/distance cost did), so only
// TREE_LEAF_SAMPLE_FRAC (10%) of the tree's LEAF nodes are kept, each together
// with its FULL ancestor chain back to root -- every retained path is complete
// and connected (a meaningful "10% of the possible trajectories through the
// tree"), not a random scatter of disconnected dots. See writeTreeCSV().
//
// Render the output with scripts/plot_tree_checkpoint.m.
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
// CS_COST_FRAC/CS_HOPELESS_GUARD/CS_RAMP_FILL_ITERS.
static const float CS_BUFFER_SLOPE   = 1.2f;
static const float CS_BUFFER_FLOOR   = 0.4f;
static const float CS_EXPLORE_FRAC   = 0.15f;
static const float CS_COST_FRAC      = 0.75f;
static const bool  CS_HOPELESS_GUARD = true;
static const int   CS_RAMP_FILL_ITERS = 700;

// Fraction of each tree's LEAF nodes (see writeTreeCSV) kept in the tree CSVs.
static const float TREE_LEAF_SAMPLE_FRAC = 0.02f;

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
// match -- the CSV's idx/parent contract is therefore unchanged either way, so
// plot_tree_checkpoint.m needs no changes. leafSampleFrac >= 1.0 keeps
// everything (no sparsification).
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
            // "evenly spaced, not random" subsampling convention (plot_tree_growth_iters.m's
            // drawTree, etc.). Early-exit once a chain hits an already-kept node: everything
            // above it was already marked by a previous leaf's walk, so this both saves work and
            // deduplicates automatically -- no separate "skip duplicates" pass needed.
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
// already-cumulative-from-root cost (snap.costs -- no need to re-sum edgeCost()),
// then reverses so the CSV reads root-first/goal-last. Writes a header-only (0
// row) CSV when goalIdx < 0 (no solution yet at this checkpoint), so every
// (planner, checkpoint) pair always produces exactly one traj file. NOT
// sparsified -- a single solution path is already small.
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
// index directly (h_pathToGoal_ is dead for both), only a best-cost scalar
// (h_minCost_). Mirrors their own internal getControlPathToGoal() kernels' own
// cost-equality filter, done host-side so we get a clean int back instead of an
// on-device buffer of unknown valid length.
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

// Same shape as tree_growth_dump.cu's writeVizMeta.
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

static const int   NUM_CHECKPOINTS         = 5;
static const float CHECKPOINTS_MS[NUM_CHECKPOINTS] = {50.0f, 100.0f, 150.0f, 1000.0f, 4000.0f};
static const int   MAX_LOOP_ITERATIONS     = 100000;   // generous safety cap, not the stopping criterion

// ========================================================================
// KPAX: has graph_, goal signal is h_pathToGoal_ (direct tree index), tracked
// into a persistent best-known (bestGoalIdx/bestCost) across iterations -- see
// the file header comment for why the raw per-iteration flag alone isn't enough.
// ========================================================================
void runKPAXCheckpoints(KPAX& planner, const std::string& token, const std::string& env,
                        const std::string& vizDir, float* h_initial, float* h_goal,
                        float* d_obstacles, uint numObstacles)
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

    while(itr < MAX_LOOP_ITERATIONS && nextCheckpoint < NUM_CHECKPOINTS)
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

        while(nextCheckpoint < NUM_CHECKPOINTS && plannerMs >= CHECKPOINTS_MS[nextCheckpoint])
        {
            int ms = (int)CHECKPOINTS_MS[nextCheckpoint];
            TreeSnapshot snap = copyTreeToHost(planner.d_treeSamples_ptr_,
                                               planner.d_treeSamplesParentIdxs_ptr_,
                                               planner.d_treeSampleCosts_ptr_, planner.h_treeSize_);
            writeTreeCSV(snap, checkpointTreePath(vizDir, env, token, ms), TREE_LEAF_SAMPLE_FRAC);
            writeTrajectoryCSV(snap, bestGoalIdx, checkpointTrajPath(vizDir, env, token, ms));
            printf("  t=%dms: itr=%d tree=%u %s\n", ms, itr, planner.h_treeSize_,
                   bestGoalIdx >= 0 ? "SOLVED" : "no solution yet");
            nextCheckpoint++;
        }

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) { printf("  stopped early: tree full\n"); break; }
        // NOTE: no h_propIterations_==0 break here, unlike the other two loops below. For KPAX
        // this field is only ever assigned inside propagateFrontier()'s near-full-tree branch
        // (KPAX.cu:254-271, "if(h_frontierRepeatSize_*h_activeBlockSize_ > MAX_TREE_SIZE-treeSize)")
        // -- during the normal, plenty-of-room phase of a run it is never touched and stays at
        // whatever stale/initial value it had (0 by default), which is NOT a "propagation
        // stalled" signal for this planner. Checking it here previously made KPAX's loop exit on
        // essentially iteration 1 every time. paper_benchmark_v2.cu's own benchmarkKPAX correctly
        // never checks this field either -- only tree-full and the timeout.
    }

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
}

// ========================================================================
// KinoPaxPlus: no graph_. Goal signal is h_minCost_ (< MAX_FLOAT, planner-
// maintained running minimum -- no persistent tracking needed here); goalIdx
// recovered via findGoalIdxByScan.
// ========================================================================
void runKinoPaxPlusCheckpoints(KinoPaxPlus& planner, const std::string& token, const std::string& env,
                               const std::string& vizDir, float* h_initial, float* h_goal,
                               float* d_obstacles, uint numObstacles)
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

    while(itr < MAX_LOOP_ITERATIONS && nextCheckpoint < NUM_CHECKPOINTS)
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

        while(nextCheckpoint < NUM_CHECKPOINTS && plannerMs >= CHECKPOINTS_MS[nextCheckpoint])
        {
            int ms = (int)CHECKPOINTS_MS[nextCheckpoint];
            TreeSnapshot snap = copyTreeToHost(planner.d_treeSamples_ptr_,
                                               planner.d_treeSamplesParentIdxs_ptr_,
                                               planner.d_treeSampleCosts_ptr_, planner.h_treeSize_);
            writeTreeCSV(snap, checkpointTreePath(vizDir, env, token, ms), TREE_LEAF_SAMPLE_FRAC);
            int goalIdx = findGoalIdxByScan(planner.d_goalSet_ptr_, snap, planner.h_minCost_);
            writeTrajectoryCSV(snap, goalIdx, checkpointTrajPath(vizDir, env, token, ms));
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
// CountingStars: has graph_ but NEVER calls graph_.updateVertices() -- it
// consumes nothing Graph produces, and that kernel is not cheap (matches
// paper_benchmark_v2.cu's benchmarkCountingStars exactly). Same h_minCost_/
// findGoalIdxByScan mechanism as KinoPaxPlus. Fixed canonical tunables set
// before resetPlanner() (which does not touch them): CS_BUFFER_SLOPE/
// CS_BUFFER_FLOOR/CS_EXPLORE_FRAC/CS_COST_FRAC/CS_HOPELESS_GUARD/
// CS_RAMP_FILL_ITERS, all matching paper_benchmark_v2.cu's own constants.
// ========================================================================
void runCountingStarsCheckpoints(CountingStars& planner, const std::string& token, const std::string& env,
                                 const std::string& vizDir, float* h_initial, float* h_goal,
                                 float* d_obstacles, uint numObstacles)
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

    while(itr < MAX_LOOP_ITERATIONS && nextCheckpoint < NUM_CHECKPOINTS)
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

        while(nextCheckpoint < NUM_CHECKPOINTS && plannerMs >= CHECKPOINTS_MS[nextCheckpoint])
        {
            int ms = (int)CHECKPOINTS_MS[nextCheckpoint];
            TreeSnapshot snap = copyTreeToHost(planner.d_treeSamples_ptr_,
                                               planner.d_treeSamplesParentIdxs_ptr_,
                                               planner.d_treeSampleCosts_ptr_, planner.h_treeSize_);
            writeTreeCSV(snap, checkpointTreePath(vizDir, env, token, ms), TREE_LEAF_SAMPLE_FRAC);
            int goalIdx = findGoalIdxByScan(planner.d_goalSet_ptr_, snap, planner.h_minCost_);
            writeTrajectoryCSV(snap, goalIdx, checkpointTrajPath(vizDir, env, token, ms));
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

    std::string vizDir = "Data/Viz/TreeCheckpoints";
    std::filesystem::create_directories(vizDir);

    // Start/goal in workspace coordinates, matching the benchmarks' main().
    float h_initial[SAMPLE_DIM] = {0};
    float h_goal[SAMPLE_DIM]    = {0};
    h_initial[0] = W_MIN + 0.1f * W_SIZE;
    h_initial[1] = W_MIN + 0.08f * W_SIZE;
    h_initial[2] = W_MIN + 0.05f * W_SIZE;
    h_goal[0]    = W_MIN + 0.8f * W_SIZE;
    h_goal[1]    = W_MIN + 0.95f * W_SIZE;
    h_goal[2]    = W_MIN + 0.9f * W_SIZE;

    printf("=======================================================\n");
    printf("    TREE CHECKPOINT DUMP\n");
    printf("=======================================================\n");
    printf("NUM_R1_REGIONS: %d\n", NUM_R1_REGIONS);
    printf("MAX_TREE_SIZE:  %d\n", MAX_TREE_SIZE);
    printf("Cost metric:    %s (COST_MODE=%d)\n",
           (COST_MODE == 2) ? "path time" : (COST_MODE == 1) ? "control effort" : "workspace path length",
           COST_MODE);
    printf("Obstacle file:  %s\n", obstaclePath.c_str());
    printf("Environment:    %s\n", envName.c_str());
    printf("Checkpoints:    50ms, 100ms, 150ms, 1000ms, 4000ms\n");
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

    {
        KPAX planner;
        runKPAXCheckpoints(planner, "KPAX", envName, vizDir, h_initial, h_goal, d_obstacles, numObstacles);
    }
    {
        KinoPaxPlus planner;
        runKinoPaxPlusCheckpoints(planner, "KinoPaxPlus", envName, vizDir, h_initial, h_goal, d_obstacles, numObstacles);
    }
    {
        CountingStars planner;
        runCountingStarsCheckpoints(planner, "CountingStars", envName, vizDir, h_initial, h_goal, d_obstacles, numObstacles);
    }

    cudaFree(d_obstacles);

    printf("\n=======================================================\n");
    printf("    DUMP COMPLETE\n");
    printf("=======================================================\n");
    printf("Trees/trajectories in: %s\n", vizDir.c_str());
    printf("Plot with: scripts/plot_tree_checkpoint.m (run it from that directory)\n");
    printf("=======================================================\n");

    return 0;
}
