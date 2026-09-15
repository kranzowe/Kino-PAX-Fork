// ========================================================================
// TREE CHECKPOINT DUMP
//
// Dumps the planner's full tree, AND its solution trajectory (with cost) if one
// exists yet, at fixed WALL-CLOCK-TIME checkpoints -- 50ms, 100ms, 150ms, 1000ms,
// 4000ms of cumulative planner-only time -- for three planners on one environment:
//
//   KPAX             pure explorer reference
//   KinoPaxPlus       pure optimizer reference
//   KinoPaxSTARTrue   syclopCap 1.0 (no cap), ancestorPrune = 1 -- same canonical
//                     point paper_benchmark_v2.cu uses for this planner
//
// This is the time-checkpoint sibling of tree_growth_dump.cu (which dumps after
// each of the first 8 ITERATIONS, with no timing and no solution capture). That
// file is untouched; this is a separate, additive tool.
//
// Per-iteration timing uses the same cudaEvent_t/plannerMs pattern as
// paper_benchmark_v2.cu's benchmark*() functions (propagate+update only inside
// the timed window). "Solution found?" differs by planner family:
//   KPAX: h_pathToGoal_ IS the goal tree index directly once non-zero (reset to
//     0 every iteration so a later, different solution isn't missed).
//   KinoPaxPlus / KinoPaxSTARTrue: h_pathToGoal_ is not wired up for these; the
//     real signal is h_minCost_ (< MAX_FLOAT). Neither exposes a host-side goal
//     index directly, so one is recovered by scanning d_goalSet_/
//     d_treeSampleCosts_ on the host for the node where goalSet[i] &&
//     costs[i] == h_minCost_ -- the same cost-equality semantics their own
//     internal getControlPathToGoal() kernels use, just done host-side instead
//     of relying on that kernel's output buffer (which never reports how many
//     entries it validly wrote). d_goalSet_ is only ever reset once, at
//     resetPlanner(), and minCost only ever decreases (atomicMinFloat), so a
//     stale `true` flag can never false-match a later, lower h_minCost_.
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
#include "planners/KPAX.cuh"
#include "planners/KinoPaxPlus.cuh"
#include "planners/KinoPaxSTARTrue.cuh"

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

// One row per node: idx,x,y,z,vx,vy,vz,parent,cost (Model-1-shaped state columns,
// same format as tree_growth_dump.cu's dumpTreeCSV). Node idx == insertion order.
void writeTreeCSV(const TreeSnapshot& snap, const std::string& path)
{
    std::ofstream file(path);
    file << "idx,x,y,z,vx,vy,vz,parent,cost\n";
    file << std::fixed << std::setprecision(6);
    for(int i = 0; i < snap.treeSize; i++)
    {
        const float* s = &snap.samples[(size_t)i * SAMPLE_DIM];
        file << i;
        for(int d = 0; d < 6; d++) file << "," << s[d];   // x,y,z,vx,vy,vz
        file << "," << snap.parents[i] << "," << snap.costs[i] << "\n";
    }
    file.close();
}

// Walks parents[] from goalIdx back to root, collecting each node's state + its
// already-cumulative-from-root cost (snap.costs -- no need to re-sum edgeCost()),
// then reverses so the CSV reads root-first/goal-last. Writes a header-only (0
// row) CSV when goalIdx < 0 (no solution yet at this checkpoint), so every
// (planner, checkpoint) pair always produces exactly one traj file.
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

// Host-side rescan for KinoPaxPlus/KinoPaxSTARTrue: neither exposes a goal tree
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
// KPAX: has graph_, goal signal is h_pathToGoal_ (direct tree index).
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

        while(nextCheckpoint < NUM_CHECKPOINTS && plannerMs >= CHECKPOINTS_MS[nextCheckpoint])
        {
            int ms = (int)CHECKPOINTS_MS[nextCheckpoint];
            TreeSnapshot snap = copyTreeToHost(planner.d_treeSamples_ptr_,
                                               planner.d_treeSamplesParentIdxs_ptr_,
                                               planner.d_treeSampleCosts_ptr_, planner.h_treeSize_);
            writeTreeCSV(snap, checkpointTreePath(vizDir, env, token, ms));
            int goalIdx = (planner.h_pathToGoal_ != 0) ? planner.h_pathToGoal_ : -1;
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

// ========================================================================
// KinoPaxPlus: no graph_. Goal signal is h_minCost_ (< MAX_FLOAT); goalIdx
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
            writeTreeCSV(snap, checkpointTreePath(vizDir, env, token, ms));
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
// KinoPaxSTARTrue: has graph_. Same h_minCost_/findGoalIdxByScan mechanism as
// KinoPaxPlus. Fixed canonical config: syclopCap 1.0 (no cap), ancestorPrune 1
// -- matching paper_benchmark_v2.cu's own ANCESTOR_PRUNE/SYCLOP_CAP constants.
// ========================================================================
void runKinoPaxSTARTrueCheckpoints(KinoPaxSTARTrue& planner, const std::string& token, const std::string& env,
                                   const std::string& vizDir, float* h_initial, float* h_goal,
                                   float* d_obstacles, uint numObstacles)
{
    // resetPlanner does not touch h_syclopCap_/h_ancestorPrune_, so setting them
    // at entry holds for the whole run (same note as paper_benchmark_v2.cu).
    planner.h_syclopCap_     = 1.0f;
    planner.h_ancestorPrune_ = 1;

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
        planner.graph_.updateVertices();
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
            writeTreeCSV(snap, checkpointTreePath(vizDir, env, token, ms));
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
        KinoPaxSTARTrue planner;
        runKinoPaxSTARTrueCheckpoints(planner, "KinoPaxSTARTrue", envName, vizDir, h_initial, h_goal, d_obstacles, numObstacles);
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
