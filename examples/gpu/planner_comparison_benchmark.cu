#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include "planners/KPAX.cuh"
#include "planners/PruneKPAX.cuh"
#include "planners/KinoPaxPlus.cuh"

struct IterationData
{
    int iteration;
    int frontier_size;
    int tree_size;
    float elapsed_time_ms;
    float best_cost;   // cumulative path cost from root to best goal found so far
};

struct RunResult
{
    std::string planner_name;
    std::string environment;
    int run_number;
    double total_time_seconds;
    int first_solution_iteration;
    float first_solution_cost;   // cumulative path cost at first solution
    float final_best_cost;       // cumulative path cost of best solution at end
    int final_tree_size;
    int total_iterations;
    std::vector<IterationData> per_iteration;
};

// =========================================================================
// Compute cumulative root-to-goal path cost on the CPU by walking the tree.
// This produces the same metric that KinoPaxPlus tracks via h_minCost_:
//   cost = sum of distance(node, parent) along the path from root to goalIdx
// Only the first W_DIM components of each SAMPLE_DIM sample are workspace
// positions; distance() uses exactly those.
// =========================================================================
float computePathCost(
    const std::vector<float>& h_treeSamples,   // flat [treeSize * SAMPLE_DIM]
    const std::vector<int>&   h_parents,        // [treeSize]
    int                       goalIdx)
{
    float totalCost = 0.0f;
    int cur = goalIdx;
    while(true)
    {
        int par = h_parents[cur];
        if(par < 0) break;  // reached root (parent of root is -1)

        float dist = 0.0f;
        for(int d = 0; d < W_DIM; d++)
        {
            float diff = h_treeSamples[cur * SAMPLE_DIM + d]
                       - h_treeSamples[par * SAMPLE_DIM + d];
            dist += diff * diff;
        }
        totalCost += std::sqrt(dist);
        cur = par;
    }
    return totalCost;
}

// Copies the tree to host and computes path cost to goalIdx.
// Call only when a new solution is found to avoid redundant copies.
float devicePathCost(float* d_treeSamples_ptr, int* d_treeSamplesParentIdxs_ptr, int treeSize, int goalIdx)
{
    std::vector<float> h_treeSamples(treeSize * SAMPLE_DIM);
    std::vector<int>   h_parents(treeSize);
    cudaMemcpy(h_treeSamples.data(), d_treeSamples_ptr,
               treeSize * SAMPLE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_parents.data(), d_treeSamplesParentIdxs_ptr,
               treeSize * sizeof(int), cudaMemcpyDeviceToHost);
    return computePathCost(h_treeSamples, h_parents, goalIdx);
}

void writePerIterationCSV(const RunResult& result, const std::string& outputDir)
{
    std::ostringstream filename;
    filename << outputDir << "/" << result.environment << "_" << result.planner_name
             << "_run" << result.run_number << ".csv";

    std::ofstream file(filename.str());
    file << "iteration,frontier_size,tree_size,elapsed_time_ms,best_cost\n";

    for(const auto& d : result.per_iteration)
    {
        file << d.iteration << ","
             << d.frontier_size << ","
             << d.tree_size << ","
             << std::fixed << std::setprecision(3) << d.elapsed_time_ms << ","
             << std::fixed << std::setprecision(6) << d.best_cost << "\n";
    }
    file.close();
}

void writeSummaryCSV(const std::vector<RunResult>& results, const std::string& outputDir)
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");

    std::ostringstream filename;
    filename << outputDir << "/planner_comparison_" << timestamp.str() << "_summary.csv";

    std::ofstream file(filename.str());
    // Note: best_cost is the cumulative workspace path length (same metric for all planners)
    file << "environment,planner,run,total_time_s,first_sol_iteration,first_sol_cost,"
         << "final_best_cost,final_tree_size,total_iterations\n";

    for(const auto& r : results)
    {
        file << r.environment << ","
             << r.planner_name << ","
             << r.run_number << ","
             << std::fixed << std::setprecision(4) << r.total_time_seconds << ","
             << r.first_solution_iteration << ","
             << std::fixed << std::setprecision(6) << r.first_solution_cost << ","
             << std::fixed << std::setprecision(6) << r.final_best_cost << ","
             << r.final_tree_size << ","
             << r.total_iterations << "\n";
    }
    file.close();
    printf("Summary written to %s\n", filename.str().c_str());
}

// ========================================================================
// KPAX Benchmark
//
// Cost metric: cumulative path length from root to goal, computed by walking
// d_treeSamplesParentIdxs_ on the CPU the first time a solution is found.
// (d_treeSampleCosts_ in KPAX stores distance-to-goal, NOT path length.)
// ========================================================================
RunResult benchmarkKPAX(
    KPAX& planner,
    const std::string& environment,
    int runNumber,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations)
{
    RunResult result;
    result.planner_name = "KPAX";
    result.environment = environment;
    result.run_number = runNumber;
    result.first_solution_iteration = -1;
    result.first_solution_cost = INFINITY;
    result.final_best_cost = INFINITY;

    cudaEvent_t start, iterStop;
    float elapsedMs = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&iterStop);

    planner.resetPlanner(h_initial, h_goal);
    cudaEventRecord(start);

    int itr = 0;
    while(itr < maxIterations)
    {
        itr++;
        planner.h_itr_++;
        planner.propagateFrontier(d_obstacles, numObstacles);
        planner.graph_.updateVertices();
        planner.updateFrontier();

        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&elapsedMs, start, iterStop);

        if(planner.h_pathToGoal_ != 0 && result.first_solution_iteration == -1)
        {
            // Compute the true cumulative path cost by walking the tree
            float pathCost = devicePathCost(
                planner.d_treeSamples_ptr_,
                planner.d_treeSamplesParentIdxs_ptr_,
                planner.h_treeSize_,
                planner.h_pathToGoal_);

            result.first_solution_iteration = itr;
            result.first_solution_cost      = pathCost;
            result.final_best_cost          = pathCost;
            // KPAX finds one solution and the tree stops improving; carry it forward
        }

        IterationData d;
        d.iteration    = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size    = planner.h_treeSize_;
        d.elapsed_time_ms = elapsedMs;
        d.best_cost    = result.final_best_cost;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
    }

    result.total_time_seconds = elapsedMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(start);
    cudaEventDestroy(iterStop);
    return result;
}

// ========================================================================
// PruneKPAX Benchmark
//
// Same cost fix as KPAX: walk the parent chain to get the true path length.
// (d_treeSampleCosts_ in PruneKPAX stores distance-to-goal, NOT path length.)
// ========================================================================
RunResult benchmarkPruneKPAX(
    PruneKPAX& planner,
    const std::string& environment,
    int runNumber,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations)
{
    RunResult result;
    result.planner_name = "PruneKPAX";
    result.environment = environment;
    result.run_number = runNumber;
    result.first_solution_iteration = -1;
    result.first_solution_cost = INFINITY;
    result.final_best_cost = INFINITY;

    cudaEvent_t start, iterStop;
    float elapsedMs = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&iterStop);

    planner.resetPlanner(h_initial, h_goal);
    cudaEventRecord(start);

    int itr = 0;
    while(itr < maxIterations)
    {
        itr++;
        planner.h_itr_++;
        planner.propagateFrontier(d_obstacles, numObstacles);
        planner.graph_.updateVertices();
        planner.updateFrontier();

        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&elapsedMs, start, iterStop);

        if(planner.h_pathToGoal_ != 0 && result.first_solution_iteration == -1)
        {
            float pathCost = devicePathCost(
                planner.d_treeSamples_ptr_,
                planner.d_treeSamplesParentIdxs_ptr_,
                planner.h_treeSize_,
                planner.h_pathToGoal_);

            result.first_solution_iteration = itr;
            result.first_solution_cost      = pathCost;
            result.final_best_cost          = pathCost;
        }

        IterationData d;
        d.iteration    = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size    = planner.h_treeSize_;
        d.elapsed_time_ms = elapsedMs;
        d.best_cost    = result.final_best_cost;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
    }

    result.total_time_seconds = elapsedMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(start);
    cudaEventDestroy(iterStop);
    return result;
}

// ========================================================================
// KinoPaxPlus Benchmark
//
// h_minCost_ already IS the cumulative path length from root to the best
// goal node found so far — no fix needed here. It is updated via
// atomicMinFloat(d_minCost_ptr_, cost) in the updateFrontier kernel, where
// cost = parentCost + distance(parent, node).
//
// First-solution detection: h_minCost_ drops below MAX_FLOAT when any goal
// node is first accepted. h_pathToGoal_ is NOT used by KinoPaxPlus.
// ========================================================================
RunResult benchmarkKinoPaxPlus(
    KinoPaxPlus& planner,
    const std::string& environment,
    int runNumber,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations)
{
    RunResult result;
    result.planner_name = "KinoPaxPlus";
    result.environment = environment;
    result.run_number = runNumber;
    result.first_solution_iteration = -1;
    result.first_solution_cost = INFINITY;
    result.final_best_cost = INFINITY;

    cudaEvent_t start, iterStop;
    float elapsedMs = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&iterStop);

    planner.resetPlanner(h_initial, h_goal);
    cudaEventRecord(start);

    int itr = 0;
    while(itr < maxIterations)
    {
        itr++;
        planner.h_itr_++;
        planner.propagateFrontier(d_obstacles, numObstacles);
        if(planner.h_propIterations_ == 0) break;
        planner.updateFrontier();

        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&elapsedMs, start, iterStop);

        // h_minCost_ is the cumulative root-to-goal path length (set in GPU kernel)
        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);

        if(planner.h_minCost_ < MAX_FLOAT && result.first_solution_iteration == -1)
        {
            result.first_solution_iteration = itr;
            result.first_solution_cost      = planner.h_minCost_;
        }
        if(planner.h_minCost_ < result.final_best_cost)
            result.final_best_cost = planner.h_minCost_;

        IterationData d;
        d.iteration    = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size    = planner.h_treeSize_;
        d.elapsed_time_ms = elapsedMs;
        d.best_cost    = result.final_best_cost;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
    }

    result.total_time_seconds = elapsedMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(start);
    cudaEventDestroy(iterStop);
    return result;
}

// ========================================================================
// Run all planners on one environment
// ========================================================================
void runEnvironmentBenchmark(
    const std::string& environment_name,
    const std::string& obstacle_path,
    float* h_initial,
    float* h_goal,
    std::vector<RunResult>& all_results,
    const std::string& outputDir,
    int numRuns,
    int maxIterations)
{
    printf("\n========================================\n");
    printf("ENVIRONMENT: %s\n", environment_name.c_str());
    printf("========================================\n");

    int numObstacles;
    float* d_obstacles;
    std::vector<float> obstacles = readObstaclesFromCSV(obstacle_path, numObstacles, W_DIM);
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(), numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);
    printf("Loaded %d obstacles\n", numObstacles);

    // --- KPAX ---
    printf("\n--- KPAX ---\n");
    {
        KPAX planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkKPAX(planner, environment_name, run, h_initial, h_goal,
                                              d_obstacles, numObstacles, maxIterations);
            printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, path_cost=%.3f\n",
                   run + 1, numRuns, result.total_time_seconds, result.total_iterations,
                   result.final_tree_size, result.first_solution_iteration, result.final_best_cost);
            writePerIterationCSV(result, outputDir);
            all_results.push_back(result);

            if(run < numRuns - 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    // --- PruneKPAX ---
    printf("\n--- PruneKPAX ---\n");
    {
        PruneKPAX planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkPruneKPAX(planner, environment_name, run, h_initial, h_goal,
                                                   d_obstacles, numObstacles, maxIterations);
            printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, path_cost=%.3f\n",
                   run + 1, numRuns, result.total_time_seconds, result.total_iterations,
                   result.final_tree_size, result.first_solution_iteration, result.final_best_cost);
            writePerIterationCSV(result, outputDir);
            all_results.push_back(result);

            if(run < numRuns - 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    // --- KinoPaxPlus ---
    printf("\n--- KinoPaxPlus ---\n");
    {
        KinoPaxPlus planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkKinoPaxPlus(planner, environment_name, run, h_initial, h_goal,
                                                     d_obstacles, numObstacles, maxIterations);
            printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, path_cost=%.3f\n",
                   run + 1, numRuns, result.total_time_seconds, result.total_iterations,
                   result.final_tree_size, result.first_solution_iteration, result.final_best_cost);
            writePerIterationCSV(result, outputDir);
            all_results.push_back(result);

            if(run < numRuns - 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    cudaFree(d_obstacles);
}

int main(void)
{
    system("rm -rf Data/Benchmarks/PlannerComparison/*");

    const int NUM_RUNS      = 10;
    const int MAX_ITERATIONS = 300;

    std::string outputDir = "Data/Benchmarks/PlannerComparison";
    std::filesystem::create_directories(outputDir);

    printf("=======================================================\n");
    printf("    PLANNER COMPARISON BENCHMARK\n");
    printf("=======================================================\n");
    printf("Planners: KPAX, PruneKPAX, KinoPaxPlus\n");
    printf("Environments: Empty, House, NarrowPassage, Trees\n");
    printf("Runs per configuration: %d\n", NUM_RUNS);
    printf("Max iterations: %d\n", MAX_ITERATIONS);
    printf("Cost metric: cumulative workspace path length (root to goal)\n");
    printf("=======================================================\n");

    float h_initial[SAMPLE_DIM] = {10.0, 8, 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float h_goal[SAMPLE_DIM]    = {80, 95.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    std::vector<RunResult> all_results;

    runEnvironmentBenchmark("Empty",
        "../include/config/obstacles/empty/obstacles.csv",
        h_initial, h_goal, all_results, outputDir, NUM_RUNS, MAX_ITERATIONS);

    runEnvironmentBenchmark("House",
        "../include/config/obstacles/house/obstacles.csv",
        h_initial, h_goal, all_results, outputDir, NUM_RUNS, MAX_ITERATIONS);

    runEnvironmentBenchmark("NarrowPassage",
        "../include/config/obstacles/narrowPassage/obstacles.csv",
        h_initial, h_goal, all_results, outputDir, NUM_RUNS, MAX_ITERATIONS);

    runEnvironmentBenchmark("Trees",
        "../include/config/obstacles/quadTrees/obstacles.csv",
        h_initial, h_goal, all_results, outputDir, NUM_RUNS, MAX_ITERATIONS);

    writeSummaryCSV(all_results, outputDir);

    printf("\n=======================================================\n");
    printf("    BENCHMARK COMPLETE\n");
    printf("=======================================================\n");
    printf("Total configurations: %d (4 environments x 3 planners)\n", 4 * 3);
    printf("Total runs: %zu\n", all_results.size());
    printf("Results saved to: %s\n", outputDir.c_str());
    printf("=======================================================\n");

    return 0;
}
