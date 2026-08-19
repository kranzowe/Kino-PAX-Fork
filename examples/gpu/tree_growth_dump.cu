// ========================================================================
// TREE GROWTH DUMP
//
// Dumps the planner's full tree after each of the first N iterations (default 8)
// for four configurations, so the early growth can be compared side by side:
//
//   ancestor_off  KinoPaxSTARancestor, h_ancestorPrune_ = 0  (== stock KinoPaxSTAR)
//   ancestor_on   KinoPaxSTARancestor, h_ancestorPrune_ = 2  (memoized ancestor chain)
//   KPAX          pure explorer, reference for coverage
//   KinoPaxPlus   pure optimizer, reference for what ancestor pruning is meant to look like
//
// This is a standalone runner, not a benchmark: it drives the raw iteration loop
// (propagateFrontier -> graph_.updateVertices -> updateFrontier) the same way
// jetson_smoke_test.cu does, with no timing, no CSV of per-iteration metrics, and
// no sweeps. Render the output with scripts/plot_tree_growth_iters.m.
//
// Per-iteration dumping is exact here: parents and costs are written once at
// insertion and never rewired, so the tree after iteration k is precisely the
// prefix [0, tree_size(k)) of any later dump.
// ========================================================================
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include "planners/KPAX.cuh"
#include "planners/KinoPaxPlus.cuh"
#include "planners/KinoPaxSTARancestor.cuh"

// ========================================================================
// Dump one tree snapshot to CSV.
// One row per node: idx,x,y,z,vx,vy,vz,parent,cost  (state columns only).
// Node idx == insertion order (the tree only appends; pruning tombstones nodes
// in place, no compaction). Model 1 state layout: [x,y,z,vx,vy,vz,...].
// ========================================================================
void dumpTreeCSV(float* d_treeSamples_ptr, int* d_parents_ptr, float* d_costs_ptr,
                 int treeSize, const std::string& path)
{
    std::vector<float> h_treeSamples((size_t)treeSize * SAMPLE_DIM);
    std::vector<int>   h_parents(treeSize);
    std::vector<float> h_costs(treeSize);
    cudaMemcpy(h_treeSamples.data(), d_treeSamples_ptr,
               (size_t)treeSize * SAMPLE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_parents.data(), d_parents_ptr, treeSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_costs.data(), d_costs_ptr, treeSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream file(path);
    file << "idx,x,y,z,vx,vy,vz,parent,cost\n";
    file << std::fixed << std::setprecision(6);
    for(int i = 0; i < treeSize; i++)
    {
        const float* s = &h_treeSamples[(size_t)i * SAMPLE_DIM];
        file << i;
        for(int d = 0; d < 6; d++) file << "," << s[d];   // x,y,z,vx,vy,vz
        file << "," << h_parents[i] << "," << h_costs[i] << "\n";
    }
    file.close();
}

// Build the per-iteration dump path: {vizDir}/{env}_{token}_iter{k}_tree.csv
std::string iterTreePath(const std::string& vizDir, const std::string& env,
                         const std::string& token, int iter)
{
    std::ostringstream ss;
    ss << vizDir << "/" << env << "_" << token << "_iter" << iter << "_tree.csv";
    return ss.str();
}

// Write a small numeric metadata row so the MATLAB script knows the R1 grid,
// workspace/velocity bounds, and start/goal without re-parsing config.h.
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

// ========================================================================
// Run one planner for numIters iterations, dumping the tree after each.
// Templated on planner type; the KinoPaxPlus specialization below drops the
// graph_.updateVertices() call because KinoPaxPlus has no Syclop Graph.
// ========================================================================
template<typename PlannerType>
void runAndDump(PlannerType& planner, const std::string& token, const std::string& env,
                const std::string& vizDir, float* h_initial, float* h_goal,
                float* d_obstacles, uint numObstacles, int numIters)
{
    printf("\n--- %s ---\n", token.c_str());
    planner.resetPlanner(h_initial, h_goal);

    for(int itr = 1; itr <= numIters; itr++)
    {
        planner.h_itr_++;
        planner.propagateFrontier(d_obstacles, numObstacles);
        planner.graph_.updateVertices();
        planner.updateFrontier();
        cudaDeviceSynchronize();

        dumpTreeCSV(planner.d_treeSamples_ptr_, planner.d_treeSamplesParentIdxs_ptr_,
                    planner.d_treeSampleCosts_ptr_, planner.h_treeSize_,
                    iterTreePath(vizDir, env, token, itr));
        printf("  iter %d: tree=%u  frontier=%u\n", itr, planner.h_treeSize_, planner.h_frontierSize_);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1)
        {
            printf("  stopped early: tree full\n");
            break;
        }
        if(planner.h_propIterations_ == 0)
        {
            printf("  stopped early: h_propIterations_ == 0\n");
            break;
        }
    }
}

// KinoPaxPlus has no graph_ member, so its loop omits updateVertices().
void runAndDumpKinoPaxPlus(KinoPaxPlus& planner, const std::string& token, const std::string& env,
                           const std::string& vizDir, float* h_initial, float* h_goal,
                           float* d_obstacles, uint numObstacles, int numIters)
{
    printf("\n--- %s ---\n", token.c_str());
    planner.resetPlanner(h_initial, h_goal);

    for(int itr = 1; itr <= numIters; itr++)
    {
        planner.h_itr_++;
        planner.propagateFrontier(d_obstacles, numObstacles);
        if(planner.h_propIterations_ == 0)
        {
            printf("  stopped early: h_propIterations_ == 0\n");
            break;
        }
        planner.updateFrontier();
        cudaDeviceSynchronize();

        dumpTreeCSV(planner.d_treeSamples_ptr_, planner.d_treeSamplesParentIdxs_ptr_,
                    planner.d_treeSampleCosts_ptr_, planner.h_treeSize_,
                    iterTreePath(vizDir, env, token, itr));
        printf("  iter %d: tree=%u\n", itr, planner.h_treeSize_);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1)
        {
            printf("  stopped early: tree full\n");
            break;
        }
    }
}

int main(int argc, char* argv[])
{
    std::string obstaclePath = (argc > 1) ? argv[1] : "../include/config/obstacles/zigzag/obstacles.csv";
    std::string envName      = (argc > 2) ? argv[2] : "zigzag";

    const int NUM_ITERS = 8;

    std::string vizDir = "Data/Viz/TreeGrowth";
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
    printf("    TREE GROWTH DUMP (first %d iterations)\n", NUM_ITERS);
    printf("=======================================================\n");
    printf("NUM_R1_REGIONS: %d\n", NUM_R1_REGIONS);
    printf("MAX_TREE_SIZE:  %d\n", MAX_TREE_SIZE);
    printf("Cost metric:    %s (COST_MODE=%d)\n",
           (COST_MODE == 1) ? "control effort" : "workspace path length", COST_MODE);
    printf("Obstacle file:  %s\n", obstaclePath.c_str());
    printf("Environment:    %s\n", envName.c_str());
    printf("Output:         %s\n", vizDir.c_str());
    printf("=======================================================\n");

    int numObstacles;
    float* d_obstacles;
    std::vector<float> obstacles = readObstaclesFromCSV(obstaclePath, numObstacles, W_DIM);
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(), numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);
    printf("Loaded %d obstacles from %s\n", numObstacles, obstaclePath.c_str());

    writeVizMeta(vizDir + "/meta.csv", h_initial, h_goal);

    // --- ancestor pruning OFF (identical to stock KinoPaxSTAR) ---
    {
        KinoPaxSTARancestor planner;
        planner.h_ancestorPrune_ = 0;
        runAndDump(planner, "ancestor_off", envName, vizDir, h_initial, h_goal,
                   d_obstacles, numObstacles, NUM_ITERS);
    }

    // --- ancestor pruning ON (memoized chain) ---
    {
        KinoPaxSTARancestor planner;
        planner.h_ancestorPrune_ = 2;
        runAndDump(planner, "ancestor_on", envName, vizDir, h_initial, h_goal,
                   d_obstacles, numObstacles, NUM_ITERS);
    }

    // --- KPAX reference (pure exploration) ---
    {
        KPAX planner;
        runAndDump(planner, "KPAX", envName, vizDir, h_initial, h_goal,
                   d_obstacles, numObstacles, NUM_ITERS);
    }

    // --- KinoPaxPlus reference (pure optimization, has the original ancestor pruning) ---
    {
        KinoPaxPlus planner;
        runAndDumpKinoPaxPlus(planner, "KinoPaxPlus", envName, vizDir, h_initial, h_goal,
                              d_obstacles, numObstacles, NUM_ITERS);
    }

    cudaFree(d_obstacles);

    printf("\n=======================================================\n");
    printf("    DUMP COMPLETE\n");
    printf("=======================================================\n");
    printf("Trees in: %s\n", vizDir.c_str());
    printf("Plot with: scripts/plot_tree_growth_iters.m (run it from that directory)\n");
    printf("=======================================================\n");

    return 0;
}
