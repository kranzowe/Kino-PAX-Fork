#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>
#include <chrono>
#include "planners/KinoPaxPlus.cuh"
#include "planners/PruneKPAX.cuh"

struct IterationData
{
    int iteration;
    float min_cost;
    float execution_time_ms;
    int tree_size;
    int goal_set_size;
};

struct BenchmarkResult
{
    std::string environment;
    std::string planner_type;
    int run_number;
    double total_time_seconds;
    float final_cost;
    int final_iterations;
    int final_tree_size;
    std::vector<IterationData> iteration_history;
};

void writeResultsToCSV(const std::vector<BenchmarkResult>& results, const std::string& filename)
{
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/Benchmarks");
    std::filesystem::create_directories("Data/Benchmarks/KinoPaxPlus");

    // Write summary file
    std::ofstream summary("Data/Benchmarks/KinoPaxPlus/" + filename + "_summary.csv");
    summary << "environment,planner_type,run_number,total_time_seconds,final_cost,final_iterations,final_tree_size\n";

    for(const auto& result : results)
    {
        summary << result.environment << ","
                << result.planner_type << ","
                << result.run_number << ","
                << result.total_time_seconds << ","
                << result.final_cost << ","
                << result.final_iterations << ","
                << result.final_tree_size << "\n";
    }
    summary.close();

    // Write detailed iteration files
    for(const auto& result : results)
    {
        std::stringstream ss;
        ss << "Data/Benchmarks/KinoPaxPlus/" << filename << "_"
           << result.environment << "_"
           << result.planner_type << "_run"
           << result.run_number << ".csv";

        std::ofstream detail(ss.str());
        detail << "iteration,min_cost,execution_time_ms,tree_size,goal_set_size\n";

        for(const auto& iter : result.iteration_history)
        {
            detail << iter.iteration << ","
                   << iter.min_cost << ","
                   << iter.execution_time_ms << ","
                   << iter.tree_size << ","
                   << iter.goal_set_size << "\n";
        }
        detail.close();
    }

    printf("\n✓ Results written to Data/Benchmarks/KinoPaxPlus/%s_*.csv\n", filename.c_str());
}

// Modified version of KinoPaxPlus planOptimize that tracks iteration data
BenchmarkResult runKinoPaxPlusBenchmark(
    KinoPaxPlus& planner,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    int numObstacles,
    const std::string& environment_name,
    const std::string& planner_type,
    int run_number,
    int max_iterations = 300)
{
    BenchmarkResult result;
    result.environment = environment_name;
    result.planner_type = planner_type;
    result.run_number = run_number;

    cudaEvent_t start, stop, iter_start, iter_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&iter_start);
    cudaEventCreate(&iter_stop);

    cudaEventRecord(start);

    // Initialize planner
    planner.resetPlanner(h_initial, h_goal);

    // Run planning with iteration tracking
    while(planner.h_itr_ < max_iterations)
    {
        cudaEventRecord(iter_start);

        planner.h_itr_++;
        planner.propagateFrontier(d_obstacles, numObstacles);

        if(planner.h_propIterations_ == 0) break;

        planner.updateFrontier();

        cudaEventRecord(iter_stop);
        cudaEventSynchronize(iter_stop);

        float iter_ms = 0;
        cudaEventElapsedTime(&iter_ms, iter_start, iter_stop);

        // Get goal set size
        int goal_set_size = thrust::count(planner.d_goalSet_.begin(),
                                          planner.d_goalSet_.begin() + planner.h_treeSize_,
                                          true);

        // Copy current min cost from device
        float current_min_cost;
        cudaMemcpy(&current_min_cost, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);

        // Record iteration data
        IterationData iter_data;
        iter_data.iteration = planner.h_itr_;
        iter_data.min_cost = current_min_cost;
        iter_data.execution_time_ms = iter_ms;
        iter_data.tree_size = planner.h_treeSize_;
        iter_data.goal_set_size = goal_set_size;

        result.iteration_history.push_back(iter_data);

        // Print progress every 50 iterations
        if(planner.h_itr_ % 50 == 0)
        {
            printf("  Iter %d: cost=%.3f, tree_size=%d, goal_set=%d\n",
                   planner.h_itr_, current_min_cost, planner.h_treeSize_, goal_set_size);
        }

        // Check if we have a path to goal
        if(planner.h_pathToGoal_ != 0)
        {
            cudaMemcpy(planner.h_controlPathsToGoal_, planner.d_controlPathsToGoal_ptr_,
                      planner.h_itr_ * SAMPLE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    // Get final control path to goal
    planner.getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);

    result.total_time_seconds = total_ms / 1000.0;
    result.final_cost = planner.h_minCost_;
    result.final_iterations = planner.h_itr_;
    result.final_tree_size = planner.h_treeSize_;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(iter_start);
    cudaEventDestroy(iter_stop);

    return result;
}

void runEnvironmentBenchmark(
    const std::string& env_name,
    const std::string& obstacle_path,
    float* h_initial,
    float* h_goal,
    std::vector<BenchmarkResult>& results,
    int num_runs = 10,
    int max_iterations = 300)
{
    printf("\n========================================\n");
    printf("ENVIRONMENT: %s\n", env_name.c_str());
    printf("========================================\n");

    // Load obstacles
    int numObstacles;
    std::vector<float> obstacles = readObstaclesFromCSV(obstacle_path, numObstacles, W_DIM);

    float* d_obstacles;
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(), numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);

    printf("Loaded %d obstacles\n", numObstacles);
    printf("Running %d iterations per run, %d runs total\n\n", max_iterations, num_runs);

    // Test KinoPaxPlus (original)
    printf("--- Testing KinoPaxPlus (Original) ---\n");
    {
        KinoPaxPlus planner;

        for(int run = 0; run < num_runs; run++)
        {
            printf("Run %d/%d:\n", run + 1, num_runs);

            BenchmarkResult result = runKinoPaxPlusBenchmark(
                planner, h_initial, h_goal, d_obstacles, numObstacles,
                env_name, "KinoPaxPlus_Original", run, max_iterations);

            printf("  Completed: %.3fs, final_cost=%.3f, iterations=%d, tree_size=%d\n",
                   result.total_time_seconds, result.final_cost,
                   result.final_iterations, result.final_tree_size);

            results.push_back(result);
        }
    }

    // Test KinoPaxPlus with spatial hashing
    printf("\n--- Testing KinoPaxPlus with Spatial Hashing (PruneKPAX) ---\n");
    // Note: PruneKPAX uses spatial hashing internally
    // We would need to adapt it to work like KinoPaxPlus with iteration tracking
    // For now, commenting this out as it would require significant modifications
    /*
    {
        PruneKPAX planner;

        for(int run = 0; run < num_runs; run++)
        {
            printf("Run %d/%d:\n", run + 1, num_runs);

            // PruneKPAX would need a similar benchmark function
            // This is left as a TODO since it has a different interface

            results.push_back(result);
        }
    }
    */

    cudaFree(d_obstacles);
}

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║   KinoPaxPlus Comprehensive Benchmark - Iteration Tracking    ║\n");
    printf("║   300 iterations × 10 runs × 3 environments             ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    std::vector<BenchmarkResult> all_results;

    // Configuration
    const int NUM_RUNS = 10;
    const int MAX_ITERATIONS = 300;

    // Environment 1: Empty
    {
        float h_initial[SAMPLE_DIM] = {10.0, 8.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        float h_goal[SAMPLE_DIM]    = {80.0, 95.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        runEnvironmentBenchmark(
            "Empty",
            "../include/config/obstacles/empty/obstacles.csv",
            h_initial, h_goal, all_results, NUM_RUNS, MAX_ITERATIONS);
    }

    // Environment 2: Pillars
    {
        float h_initial[SAMPLE_DIM] = {10.0, 8.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        float h_goal[SAMPLE_DIM]    = {80.0, 95.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        runEnvironmentBenchmark(
            "Pillars",
            "../include/config/obstacles/pillars/obstacles.csv",
            h_initial, h_goal, all_results, NUM_RUNS, MAX_ITERATIONS);
    }

    // Environment 3: Narrow Passage
    {
        float h_initial[SAMPLE_DIM] = {10.0, 8.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        float h_goal[SAMPLE_DIM]    = {80.0, 95.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        runEnvironmentBenchmark(
            "NarrowPassage",
            "../include/config/obstacles/narrowPassage/obstacles.csv",
            h_initial, h_goal, all_results, NUM_RUNS, MAX_ITERATIONS);
    }

    // Write all results
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");

    std::string filename = "kinopaxplus_benchmark_" + timestamp.str();
    writeResultsToCSV(all_results, filename);

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║                  Benchmark Complete!                     ║\n");
    printf("║  Total runs: %-3zu                                       ║\n", all_results.size());
    printf("╚══════════════════════════════════════════════════════════╝\n");

    return 0;
}
