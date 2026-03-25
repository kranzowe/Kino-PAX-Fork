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
#include "planners/KinoPaxPlus.cuh"
#include "planners/KPAX.cuh"

struct IterationData
{
    int iteration;
    int frontier_size;
    int tree_size;
    float elapsed_time_ms;
    float best_cost;
};

struct RunResult
{
    std::string delta_label;
    std::string environment;
    int run_number;
    double total_time_seconds;
    int first_solution_iteration;
    float first_solution_cost;
    float final_best_cost;
    int final_tree_size;
    int total_iterations;
    std::vector<IterationData> per_iteration;
};

// ========================================================================
// Compute cumulative root-to-goal path cost by walking the parent chain.
// Same metric KinoPaxPlus tracks via h_minCost_: sum of workspace distance
// along path from root to goalIdx.
// ========================================================================
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

// Copies tree to host and computes path cost to goalIdx.
float devicePathCost(float* d_treeSamples_ptr, int* d_treeSamplesParentIdxs_ptr,
                     int treeSize, int goalIdx)
{
    std::vector<float> h_treeSamples(treeSize * SAMPLE_DIM);
    std::vector<int>   h_parents(treeSize);
    cudaMemcpy(h_treeSamples.data(), d_treeSamples_ptr,
               treeSize * SAMPLE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_parents.data(), d_treeSamplesParentIdxs_ptr,
               treeSize * sizeof(int), cudaMemcpyDeviceToHost);
    return computePathCost(h_treeSamples, h_parents, goalIdx);
}

// ========================================================================
// Write per-iteration CSV for a single run
// ========================================================================
void writePerIterationCSV(const RunResult& result, const std::string& outputDir)
{
    std::ostringstream filename;
    // KPAX baseline: {env}_KPAX_run{n}.csv
    // KinoPaxPlus:   {env}_delta{label}_run{n}.csv
    if(result.delta_label == "KPAX")
        filename << outputDir << "/" << result.environment << "_KPAX_run"
                 << result.run_number << ".csv";
    else
        filename << outputDir << "/" << result.environment << "_delta" << result.delta_label
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

// ========================================================================
// Write summary CSV aggregating all runs
// ========================================================================
void writeSummaryCSV(const std::vector<RunResult>& results, const std::string& outputDir,
                     const std::string& deltaLabel)
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");

    std::ostringstream filename;
    filename << outputDir << "/delta_benchmark_" << timestamp.str() << "_summary.csv";

    std::ofstream file(filename.str());
    file << "environment,delta_label,num_regions,run,total_time_s,first_sol_iteration,"
         << "first_sol_cost,final_best_cost,final_tree_size,total_iterations\n";

    for(const auto& r : results)
    {
        int regions = (r.delta_label == "KPAX") ? 0 : NUM_R1_REGIONS;
        file << r.environment << ","
             << r.delta_label << ","
             << regions << ","
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
// KinoPaxPlus Benchmark
//
// h_minCost_ is the cumulative path length from root to the best goal node
// found so far — updated via atomicMinFloat in the updateFrontier kernel.
// ========================================================================
RunResult benchmarkKinoPaxPlus(
    KinoPaxPlus& planner,
    const std::string& deltaLabel,
    const std::string& environment,
    int runNumber,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations,
    float maxTimeMs)
{
    RunResult result;
    result.delta_label = deltaLabel;
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

        // h_minCost_ is the cumulative root-to-goal path length
        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);

        if(planner.h_minCost_ < MAX_FLOAT && result.first_solution_iteration == -1)
        {
            result.first_solution_iteration = itr;
            result.first_solution_cost      = planner.h_minCost_;
        }
        if(planner.h_minCost_ < result.final_best_cost)
            result.final_best_cost = planner.h_minCost_;

        IterationData d;
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = elapsedMs;
        d.best_cost     = result.final_best_cost;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        
        // Timeout check
        if(elapsedMs >= maxTimeMs) break;
    }

    result.total_time_seconds = elapsedMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(start);
    cudaEventDestroy(iterStop);
    return result;
}

// ========================================================================
// KPAX Benchmark (naive optimality)
//
// Cost metric: cumulative path length from root to goal, computed by walking
// d_treeSamplesParentIdxs_ on the CPU whenever a new solution is found.
// d_pathToGoal_ is reset to 0 before every iteration so new goal nodes
// discovered in later iterations are detected.
// ========================================================================
RunResult benchmarkKPAX(
    KPAX& planner,
    const std::string& environment,
    int runNumber,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations,
    float maxTimeMs)
{
    RunResult result;
    result.delta_label = "KPAX";
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

    int zero = 0;
    int itr = 0;
    while(itr < maxIterations)
    {
        itr++;
        planner.h_itr_++;

        // Reset pathToGoal before each iteration so we can detect new goals
        cudaMemcpy(planner.d_pathToGoal_ptr_, &zero, sizeof(int), cudaMemcpyHostToDevice);
        planner.h_pathToGoal_ = 0;

        planner.propagateFrontier(d_obstacles, numObstacles);
        planner.graph_.updateVertices();
        planner.updateFrontier();

        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&elapsedMs, start, iterStop);

        // Check if a new path to goal was found THIS iteration
        if(planner.h_pathToGoal_ != 0)
        {
            float pathCost = devicePathCost(
                planner.d_treeSamples_ptr_,
                planner.d_treeSamplesParentIdxs_ptr_,
                planner.h_treeSize_,
                planner.h_pathToGoal_);

            if(result.first_solution_iteration == -1)
            {
                result.first_solution_iteration = itr;
                result.first_solution_cost      = pathCost;
            }
            if(pathCost < result.final_best_cost)
                result.final_best_cost = pathCost;
        }

        IterationData d;
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = elapsedMs;
        d.best_cost     = result.final_best_cost;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        
        // Timeout check
        if(elapsedMs >= maxTimeMs) break;
    }

    result.total_time_seconds = elapsedMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(start);
    cudaEventDestroy(iterStop);
    return result;
}

// ========================================================================
// Run KPAX baseline on one environment for 20 runs
// ========================================================================
void runKPAXBaseline(
    const std::string& environment_name,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    std::vector<RunResult>& all_results,
    const std::string& outputDir,
    int numRuns,
    int maxIterations,
    float maxTimeMs)
{
    printf("\n========================================\n");
    printf("KPAX BASELINE: %s | %d runs\n", environment_name.c_str(), numRuns);
    printf("========================================\n");

    {
        KPAX planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkKPAX(planner, environment_name, run,
                                             h_initial, h_goal, d_obstacles,
                                             numObstacles, maxIterations, maxTimeMs);
            printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
                   run + 1, numRuns, result.total_time_seconds, result.total_iterations,
                   result.final_tree_size, result.first_solution_iteration,
                   result.first_solution_cost, result.final_best_cost);
            writePerIterationCSV(result, outputDir);
            all_results.push_back(result);

            if(run < numRuns - 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

// ========================================================================
// Run KinoPaxPlus on one environment for multiple runs
// ========================================================================
void runKinoPaxPlusBenchmark(
    const std::string& environment_name,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    std::vector<RunResult>& all_results,
    const std::string& outputDir,
    const std::string& deltaLabel,
    int numRuns,
    int maxIterations,
    float maxTimeMs)
{
    printf("\n========================================\n");
    printf("KINOPAXPLUS: %s | Delta: %s | Regions: %d\n",
           environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    {
        KinoPaxPlus planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkKinoPaxPlus(planner, deltaLabel, environment_name, run,
                                                     h_initial, h_goal, d_obstacles,
                                                     numObstacles, maxIterations, maxTimeMs);
            printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f\n",
                   run + 1, numRuns, result.total_time_seconds, result.total_iterations,
                   result.final_tree_size, result.first_solution_iteration, result.final_best_cost);
            writePerIterationCSV(result, outputDir);
            all_results.push_back(result);

            if(run < numRuns - 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

int main(int argc, char* argv[])
{
    std::string deltaLabel    = (argc > 1) ? argv[1] : "unknown";
    std::string obstaclePath  = (argc > 2) ? argv[2] : "../include/config/obstacles/trees/obstacles.csv";
    std::string envName       = (argc > 3) ? argv[3] : "trees";

    // Check for --run-kpax flag in remaining args
    bool runKPAX = false;
    for(int i = 4; i < argc; i++)
    {
        if(std::string(argv[i]) == "--run-kpax")
            runKPAX = true;
    }

    const int NUM_KPAX_RUNS      = 20;
    const int NUM_KPAXPLUS_RUNS  = 10;
    const int MAX_ITERATIONS     = 300;
    const float MAX_TIME_MS      = 10000.0f;  // 10 second timeout

    std::string outputDir = "Data/Benchmarks/KinoPaxPlusDelta";
    std::filesystem::create_directories(outputDir);

    printf("=======================================================\n");
    printf("    KINOPAXPLUS DELTA BENCHMARK\n");
    printf("=======================================================\n");
    printf("Delta label:    %s\n", deltaLabel.c_str());
    printf("NUM_R1_REGIONS: %d\n", NUM_R1_REGIONS);
    printf("MAX_TREE_SIZE:  %d\n", MAX_TREE_SIZE);
    printf("W_R1_LENGTH=%d  C_R1_LENGTH=%d  V_R1_LENGTH=%d\n", W_R1_LENGTH, C_R1_LENGTH, V_R1_LENGTH);
    printf("Obstacle file:  %s\n", obstaclePath.c_str());
    printf("Environment:    %s\n", envName.c_str());
    printf("KPAX baseline:  %s (%d runs)\n", runKPAX ? "YES" : "NO", NUM_KPAX_RUNS);
    printf("KinoPaxPlus:    %d runs\n", NUM_KPAXPLUS_RUNS);
    printf("Max iterations: %d\n", MAX_ITERATIONS);
    printf("=======================================================\n");

    // Start/goal states — workspace coordinates via W_MIN/W_SIZE from config.h
    // Model 1 [0,1]^3: (0.1,0.08,0.05) -> (0.8,0.95,0.9)
    float h_initial[SAMPLE_DIM] = {0};
    float h_goal[SAMPLE_DIM]    = {0};
    h_initial[0] = W_MIN + 0.1f * W_SIZE;
    h_initial[1] = W_MIN + 0.08f * W_SIZE;
    h_initial[2] = W_MIN + 0.05f * W_SIZE;
    h_goal[0]    = W_MIN + 0.8f * W_SIZE;
    h_goal[1]    = W_MIN + 0.95f * W_SIZE;
    h_goal[2]    = W_MIN + 0.9f * W_SIZE;

    // Load obstacles
    int numObstacles;
    float* d_obstacles;
    std::vector<float> obstacles = readObstaclesFromCSV(obstaclePath, numObstacles, W_DIM);
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(), numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);
    printf("Loaded %d obstacles from %s\n", numObstacles, obstaclePath.c_str());

    std::vector<RunResult> all_results;

    // --- KPAX baseline (only when flagged) ---
    if(runKPAX)
    {
        runKPAXBaseline(envName, h_initial, h_goal, d_obstacles, numObstacles,
                        all_results, outputDir, NUM_KPAX_RUNS, MAX_ITERATIONS, MAX_TIME_MS);
    }

    // --- KinoPaxPlus delta benchmark ---
    runKinoPaxPlusBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                            all_results, outputDir, deltaLabel, NUM_KPAXPLUS_RUNS, MAX_ITERATIONS, MAX_TIME_MS);

    cudaFree(d_obstacles);

    writeSummaryCSV(all_results, outputDir, deltaLabel);

    printf("\n=======================================================\n");
    printf("    BENCHMARK COMPLETE (delta=%s, env=%s)\n", deltaLabel.c_str(), envName.c_str());
    printf("=======================================================\n");
    printf("Total runs: %zu\n", all_results.size());
    printf("Results saved to: %s\n", outputDir.c_str());
    printf("=======================================================\n");

    return 0;
}
