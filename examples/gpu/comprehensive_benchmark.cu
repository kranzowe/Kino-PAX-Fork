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
#include "ReKino/ReKinoLite.cuh"

/*
 * Comprehensive Benchmark Script
 *
 * Tests multiple planners across multiple environments:
 * - Environments: House, Narrow Passage, Trees (quadTrees)
 * - Planners: KPAX, KPAX with Spatial Hashing, ReKinoLite
 * - 50 runs per configuration
 *
 * Output: Single CSV file with all results for box plot analysis
 */

struct BenchmarkResult
{
    std::string environment;
    std::string planner;
    int run_number;
    double execution_time;
    bool success;
};

void writeResultsToCSV(const std::vector<BenchmarkResult>& results)
{
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/Benchmarks");

    std::ofstream file("Data/Benchmarks/comprehensive_benchmark.csv");

    // Header
    file << "environment,planner,run_number,execution_time,success\n";

    // Data
    for(const auto& r : results)
    {
        file << r.environment << ","
             << r.planner << ","
             << r.run_number << ","
             << r.execution_time << ","
             << (r.success ? "1" : "0") << "\n";
    }

    file.close();
    printf("\n=== Results written to Data/Benchmarks/comprehensive_benchmark.csv ===\n");
}

void runBenchmark(
    const std::string& environment_name,
    const std::string& obstacle_path,
    float* h_initial,
    float* h_goal,
    std::vector<BenchmarkResult>& results)
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

    const int NUM_RUNS = 50;
    const double MAX_TIME_SECONDS = 6.0;  // 6 second timeout per run

    // ========================================================================
    // PLANNER 1: KPAX (without spatial hashing)
    // ========================================================================
    printf("\n--- Testing KPAX (No Spatial Hash) ---\n");
    {
        KPAX planner;

        for(int run = 0; run < NUM_RUNS; run++)
        {
            cudaEvent_t start, stop;
            float milliseconds = 0;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            planner.plan(h_initial, h_goal, d_obstacles, numObstacles, false);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            double seconds = milliseconds / 1000.0;
            bool success = (seconds <= MAX_TIME_SECONDS);

            if(seconds > MAX_TIME_SECONDS)
            {
                seconds = MAX_TIME_SECONDS;
                printf("  Run %d/%d: TIMEOUT (%.3fs)\n", run + 1, NUM_RUNS, seconds);
            }
            else
            {
                printf("  Run %d/%d: %.3fs\n", run + 1, NUM_RUNS, seconds);
            }

            BenchmarkResult result;
            result.environment = environment_name;
            result.planner = "KPAX";
            result.run_number = run + 1;
            result.execution_time = seconds;
            result.success = success;
            results.push_back(result);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            // Add delay between runs to prevent GPU thermal throttling
            if(run < NUM_RUNS - 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
    }

    // ========================================================================
    // PLANNER 2: KPAX with Spatial Hashing
    // ========================================================================
    printf("\n--- Testing KPAX (With Spatial Hash) ---\n");
    {
        KPAX planner;  // Assuming KPAX uses spatial hashing by default

        for(int run = 0; run < NUM_RUNS; run++)
        {
            cudaEvent_t start, stop;
            float milliseconds = 0;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            planner.plan(h_initial, h_goal, d_obstacles, numObstacles, false);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            double seconds = milliseconds / 1000.0;
            bool success = (seconds <= MAX_TIME_SECONDS);

            if(seconds > MAX_TIME_SECONDS)
            {
                seconds = MAX_TIME_SECONDS;
                printf("  Run %d/%d: TIMEOUT (%.3fs)\n", run + 1, NUM_RUNS, seconds);
            }
            else
            {
                printf("  Run %d/%d: %.3fs\n", run + 1, NUM_RUNS, seconds);
            }

            BenchmarkResult result;
            result.environment = environment_name;
            result.planner = "KPAX_SpatialHash";
            result.run_number = run + 1;
            result.execution_time = seconds;
            result.success = success;
            results.push_back(result);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            // Add delay between runs to prevent GPU thermal throttling
            if(run < NUM_RUNS - 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
    }

    // ========================================================================
    // PLANNER 3: ReKinoLite (samplesPerThread=1, epsilon=0.2)
    // ========================================================================
    printf("\n--- Testing ReKinoLite (samples=1, epsilon=0.2) ---\n");
    {
        ReKinoLite planner;
        planner.h_samplesPerThread_ = 1;
        planner.h_epsilonGreedy_ = 0.2f;

        for(int run = 0; run < NUM_RUNS; run++)
        {
            cudaEvent_t start, stop;
            float milliseconds = 0;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            planner.plan(h_initial, h_goal, d_obstacles, numObstacles, false);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            double seconds = milliseconds / 1000.0;
            bool success = (seconds <= MAX_TIME_SECONDS);

            if(seconds > MAX_TIME_SECONDS)
            {
                seconds = MAX_TIME_SECONDS;
                printf("  Run %d/%d: TIMEOUT (%.3fs)\n", run + 1, NUM_RUNS, seconds);
            }
            else
            {
                printf("  Run %d/%d: %.3fs\n", run + 1, NUM_RUNS, seconds);
            }

            BenchmarkResult result;
            result.environment = environment_name;
            result.planner = "ReKinoLite_s1_e0.2";
            result.run_number = run + 1;
            result.execution_time = seconds;
            result.success = success;
            results.push_back(result);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            // Add delay between runs to prevent GPU thermal throttling
            if(run < NUM_RUNS - 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
    }

    // Cleanup
    cudaFree(d_obstacles);
}

int main(void)
{
    // Remove previous data
    system("rm -rf Data/Benchmarks/*");

    printf("=======================================================\n");
    printf("    COMPREHENSIVE PLANNER BENCHMARK\n");
    printf("=======================================================\n");
    printf("Environments: House, Narrow Passage, Trees (quadTrees)\n");
    printf("Planners: KPAX, KPAX+SpatialHash, ReKinoLite\n");
    printf("Runs per configuration: 50\n");
    printf("Timeout: 6 seconds\n");
    printf("=======================================================\n");

    // Problem setup - same initial/goal for all environments
    float h_initial[SAMPLE_DIM] = {10.0, 8, 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float h_goal[SAMPLE_DIM]    = {80, 95.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    std::vector<BenchmarkResult> all_results;

    // ========================================================================
    // ENVIRONMENT 1: House
    // ========================================================================
    runBenchmark(
        "House",
        "../include/config/obstacles/house/obstacles.csv",
        h_initial,
        h_goal,
        all_results
    );

    // ========================================================================
    // ENVIRONMENT 2: Narrow Passage
    // ========================================================================
    runBenchmark(
        "NarrowPassage",
        "../include/config/obstacles/narrowPassage/obstacles.csv",
        h_initial,
        h_goal,
        all_results
    );

    // ========================================================================
    // ENVIRONMENT 3: Trees (quadTrees)
    // ========================================================================
    runBenchmark(
        "Trees",
        "../include/config/obstacles/quadTrees/obstacles.csv",
        h_initial,
        h_goal,
        all_results
    );

    // ========================================================================
    // Write all results to CSV
    // ========================================================================
    writeResultsToCSV(all_results);

    printf("\n=======================================================\n");
    printf("    BENCHMARK COMPLETE\n");
    printf("=======================================================\n");
    printf("Total configurations: %d\n", 3 * 3);  // 3 environments × 3 planners
    printf("Total runs: %zu\n", all_results.size());
    printf("Results saved to: Data/Benchmarks/comprehensive_benchmark.csv\n");
    printf("=======================================================\n");

    return 0;
}
