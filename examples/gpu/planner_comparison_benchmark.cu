#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include "planners/KPAX.cuh"
#include "planners/PruneKPAX.cuh"
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
    std::string planner_name;
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

        // Record timing
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&elapsedMs, start, iterStop);

        // Check if goal was found and read cost
        float bestCost = INFINITY;
        if(planner.h_pathToGoal_ != 0)
        {
            // Read cost of the goal node from device
            float goalCost;
            cudaMemcpy(&goalCost, planner.d_treeSampleCosts_ptr_ + planner.h_pathToGoal_, sizeof(float), cudaMemcpyDeviceToHost);
            bestCost = goalCost;

            if(result.first_solution_iteration == -1)
            {
                result.first_solution_iteration = itr;
                result.first_solution_cost = bestCost;
            }
            if(bestCost < result.final_best_cost)
                result.final_best_cost = bestCost;
        }

        IterationData d;
        d.iteration = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size = planner.h_treeSize_;
        d.elapsed_time_ms = elapsedMs;
        d.best_cost = result.final_best_cost;
        result.per_iteration.push_back(d);

        // KPAX breaks on first solution in normal mode, but for benchmark we continue
        // to track if tree is full
        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
    }

    result.total_time_seconds = elapsedMs / 1000.0;
    result.final_tree_size = planner.h_treeSize_;
    result.total_iterations = itr;

    cudaEventDestroy(start);
    cudaEventDestroy(iterStop);

    return result;
}

// ========================================================================
// PruneKPAX Benchmark
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

        // Record timing
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&elapsedMs, start, iterStop);

        // Check if goal was found and read cost
        float bestCost = INFINITY;
        if(planner.h_pathToGoal_ != 0)
        {
            float goalCost;
            cudaMemcpy(&goalCost, planner.d_treeSampleCosts_ptr_ + planner.h_pathToGoal_, sizeof(float), cudaMemcpyDeviceToHost);
            bestCost = goalCost;

            if(result.first_solution_iteration == -1)
            {
                result.first_solution_iteration = itr;
                result.first_solution_cost = bestCost;
            }
            if(bestCost < result.final_best_cost)
                result.final_best_cost = bestCost;
        }

        IterationData d;
        d.iteration = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size = planner.h_treeSize_;
        d.elapsed_time_ms = elapsedMs;
        d.best_cost = result.final_best_cost;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
    }

    result.total_time_seconds = elapsedMs / 1000.0;
    result.final_tree_size = planner.h_treeSize_;
    result.total_iterations = itr;

    cudaEventDestroy(start);
    cudaEventDestroy(iterStop);

    return result;
}

// ========================================================================
// KinoPaxPlus Benchmark
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

        // Record timing
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&elapsedMs, start, iterStop);

        // Read min cost from device
        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
        float bestCost = planner.h_minCost_;

        if(planner.h_pathToGoal_ != 0 && result.first_solution_iteration == -1)
        {
            result.first_solution_iteration = itr;
            result.first_solution_cost = bestCost;
        }
        if(bestCost < result.final_best_cost)
            result.final_best_cost = bestCost;

        IterationData d;
        d.iteration = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size = planner.h_treeSize_;
        d.elapsed_time_ms = elapsedMs;
        d.best_cost = result.final_best_cost;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
    }

    result.total_time_seconds = elapsedMs / 1000.0;
    result.final_tree_size = planner.h_treeSize_;
    result.total_iterations = itr;

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

    // Load obstacles
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
            printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, best_cost=%.3f\n",
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
            printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, best_cost=%.3f\n",
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
            printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, best_cost=%.3f\n",
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

    const int NUM_RUNS = 10;
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
    printf("=======================================================\n");

    float h_initial[SAMPLE_DIM] = {10.0, 8, 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float h_goal[SAMPLE_DIM]    = {80, 95.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    std::vector<RunResult> all_results;

    // --- Empty ---
    runEnvironmentBenchmark(
        "Empty",
        "../include/config/obstacles/empty/obstacles.csv",
        h_initial, h_goal, all_results, outputDir, NUM_RUNS, MAX_ITERATIONS);

    // --- House ---
    runEnvironmentBenchmark(
        "House",
        "../include/config/obstacles/house/obstacles.csv",
        h_initial, h_goal, all_results, outputDir, NUM_RUNS, MAX_ITERATIONS);

    // --- Narrow Passage ---
    runEnvironmentBenchmark(
        "NarrowPassage",
        "../include/config/obstacles/narrowPassage/obstacles.csv",
        h_initial, h_goal, all_results, outputDir, NUM_RUNS, MAX_ITERATIONS);

    // --- Trees ---
    runEnvironmentBenchmark(
        "Trees",
        "../include/config/obstacles/quadTrees/obstacles.csv",
        h_initial, h_goal, all_results, outputDir, NUM_RUNS, MAX_ITERATIONS);

    // --- Write summary ---
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
