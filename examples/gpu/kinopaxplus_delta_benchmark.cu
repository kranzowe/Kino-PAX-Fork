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
// Write per-iteration CSV for a single run
// ========================================================================
void writePerIterationCSV(const RunResult& result, const std::string& outputDir)
{
    std::ostringstream filename;
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
        file << r.environment << ","
             << r.delta_label << ","
             << NUM_R1_REGIONS << ","
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
    int maxIterations)
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
    }

    result.total_time_seconds = elapsedMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(start);
    cudaEventDestroy(iterStop);
    return result;
}

// ========================================================================
// Run KinoPaxPlus on one environment for multiple runs
// ========================================================================
void runEnvironmentBenchmark(
    const std::string& environment_name,
    const std::string& obstacle_path,
    float* h_initial,
    float* h_goal,
    std::vector<RunResult>& all_results,
    const std::string& outputDir,
    const std::string& deltaLabel,
    int numRuns,
    int maxIterations)
{
    printf("\n========================================\n");
    printf("ENVIRONMENT: %s | Delta: %s | Regions: %d\n",
           environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    int numObstacles;
    float* d_obstacles;
    std::vector<float> obstacles = readObstaclesFromCSV(obstacle_path, numObstacles, W_DIM);
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(), numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);
    printf("Loaded %d obstacles\n", numObstacles);

    printf("\n--- KinoPaxPlus (delta=%s) ---\n", deltaLabel.c_str());
    {
        KinoPaxPlus planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkKinoPaxPlus(planner, deltaLabel, environment_name, run,
                                                     h_initial, h_goal, d_obstacles, numObstacles, maxIterations);
            printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f\n",
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

int main(int argc, char* argv[])
{
    std::string deltaLabel   = (argc > 1) ? argv[1] : "unknown";
    std::string obstaclePath = (argc > 2) ? argv[2] : "../include/config/obstacles/quadTrees/obstacles.csv";

    const int NUM_RUNS       = 10;
    const int MAX_ITERATIONS = 300;

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
    printf("Environment:    Trees\n");
    printf("Runs:           %d\n", NUM_RUNS);
    printf("Max iterations: %d\n", MAX_ITERATIONS);
    printf("=======================================================\n");

    // Start/goal states — workspace coordinates match W_MIN/W_MAX from config.h
    // Model 3 [0,100]^3: (10,8,5) -> (80,95,90)
    float h_initial[SAMPLE_DIM] = {0};
    float h_goal[SAMPLE_DIM]    = {0};
    h_initial[0] = W_MIN + 0.1f * W_SIZE;
    h_initial[1] = W_MIN + 0.08f * W_SIZE;
    h_initial[2] = W_MIN + 0.05f * W_SIZE;
    h_goal[0]    = W_MIN + 0.8f * W_SIZE;
    h_goal[1]    = W_MIN + 0.95f * W_SIZE;
    h_goal[2]    = W_MIN + 0.9f * W_SIZE;

    std::vector<RunResult> all_results;

    runEnvironmentBenchmark("Trees",
        obstaclePath,
        h_initial, h_goal, all_results, outputDir, deltaLabel, NUM_RUNS, MAX_ITERATIONS);

    writeSummaryCSV(all_results, outputDir, deltaLabel);

    printf("\n=======================================================\n");
    printf("    BENCHMARK COMPLETE (delta=%s)\n", deltaLabel.c_str());
    printf("=======================================================\n");
    printf("Total runs: %zu\n", all_results.size());
    printf("Results saved to: %s\n", outputDir.c_str());
    printf("=======================================================\n");

    return 0;
}
