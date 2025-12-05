#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <thread>
#include <chrono>
#include "planners/PruneKPAX.cuh"

/*
 * PruneKPAX Parameter Tuning Script
 *
 * This script explores different parameter combinations for goal-biased pruning:
 * - maxRegression: Maximum allowed regression in distance to goal (5.0, 10.0, 20.0, 50.0)
 * - explorationBias: Base exploration probability (0.1, 0.3, 0.5, 0.7)
 * - goalBias: Goal-directed bias multiplier (0.3, 0.5, 0.7, 0.9)
 *
 * For each configuration, it runs 50 trials and:
 * 1. Measures execution time
 * 2. Tracks success rate (reaching goal within 6 seconds)
 * 3. Saves all individual runtimes to CSV for statistical analysis
 */

struct TuningResult
{
    float maxRegression;
    float explorationBias;
    float goalBias;
    int trials;
    int successes;
    double avgTime;
    double minTime;
    double maxTime;
    float successRate;
    std::vector<double> allTimes;  // Store all individual runtimes
};

void writeResultsToCSV(const std::vector<TuningResult>& results)
{
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/Tuning");

    std::ofstream file("Data/Tuning/prune_kpax_tuning_results.csv");

    // Write header: parameters + run_1, run_2, ..., run_50
    file << "maxRegression,explorationBias,goalBias";
    for(int i = 1; i <= 50; i++)
    {
        file << ",run_" << i;
    }
    file << "\n";

    // Write each configuration's data
    for(const auto& r : results)
    {
        // Write parameters
        file << r.maxRegression << "," << r.explorationBias << "," << r.goalBias;

        // Write all runtimes
        for(size_t i = 0; i < 50; i++)
        {
            file << ",";
            if(i < r.allTimes.size())
            {
                file << r.allTimes[i];
            }
        }
        file << "\n";
    }

    file.close();
    printf("Results written to Data/Tuning/prune_kpax_tuning_results.csv\n");
}

void runTuningExperiment(
    float maxRegression,
    float explorationBias,
    float goalBias,
    int numTrials,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    int numObstacles,
    TuningResult& result)
{
    result.maxRegression = maxRegression;
    result.explorationBias = explorationBias;
    result.goalBias = goalBias;
    result.trials = numTrials;
    result.successes = 0;
    result.avgTime = 0.0;
    result.minTime = 1e9;
    result.maxTime = 0.0;

    printf("\n=== Testing: maxReg=%.1f, explBias=%.2f, goalBias=%.2f ===\n",
           maxRegression, explorationBias, goalBias);

    // Create planner and set parameters
    PruneKPAX planner;
    planner.h_maxRegression_ = maxRegression;
    planner.h_explorationBias_ = explorationBias;
    planner.h_goalBias_ = goalBias;

    const double MAX_TIME_SECONDS = 6.0;

    for(int trial = 0; trial < numTrials; trial++)
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

        // Check if execution exceeded timeout
        if(seconds > MAX_TIME_SECONDS)
        {
            seconds = MAX_TIME_SECONDS;  // Cap at max time
            printf("  Trial %d/%d TIMEOUT (%.3fs > %.1fs) - recording as %.1fs and continuing\n",
                   trial + 1, numTrials, milliseconds / 1000.0, MAX_TIME_SECONDS, MAX_TIME_SECONDS);
        }
        else
        {
            result.successes++;
            if(seconds < result.minTime) result.minTime = seconds;
            if(seconds > result.maxTime) result.maxTime = seconds;
        }

        result.avgTime += seconds;
        result.allTimes.push_back(seconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Add delay between trials to prevent GPU thermal throttling
        if(trial < numTrials - 1)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        // Progress update every 10 trials
        if((trial + 1) % 10 == 0)
        {
            printf("  Trial %d/%d complete (%.3fs)\n", trial + 1, numTrials, seconds);
        }
    }

    result.avgTime /= numTrials;
    result.successRate = (float)result.successes / numTrials;

    printf("Results: Avg=%.3fs, Min=%.3fs, Max=%.3fs, Success=%.1f%%\n",
           result.avgTime, result.minTime, result.maxTime, result.successRate * 100.0f);
}

int main(void)
{
    // Remove previous tuning data
    system("rm -rf Data/Tuning/prune_kpax_tuning_results.csv");

    printf("=======================================================\n");
    printf("    PRUNE KPAX PARAMETER TUNING\n");
    printf("=======================================================\n");
    printf("Environment: Narrow Passage\n");
    printf("Trials per configuration: 50\n");
    printf("Timeout: 6.0 seconds\n");
    printf("=======================================================\n");

    // Problem setup
    float h_initial[SAMPLE_DIM] = {10.0, 8, 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float h_goal[SAMPLE_DIM]    = {80, 95.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Load obstacles
    int numObstacles;
    std::vector<float> obstacles = readObstaclesFromCSV("../include/config/obstacles/narrowPassage/obstacles.csv", numObstacles, W_DIM);
    float* d_obstacles;
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(), numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);

    printf("Loaded %d obstacles\n", numObstacles);

    // Parameter ranges to test
    std::vector<float> maxRegressionValues = {5.0f, 10.0f, 20.0f, 50.0f};
    std::vector<float> explorationBiasValues = {0.1f, 0.3f, 0.5f, 0.7f};
    std::vector<float> goalBiasValues = {0.3f, 0.5f, 0.7f, 0.9f};

    std::vector<TuningResult> results;

    int totalConfigs = maxRegressionValues.size() * explorationBiasValues.size() * goalBiasValues.size();
    int currentConfig = 0;

    // Test all combinations
    for(float maxReg : maxRegressionValues)
    {
        for(float explBias : explorationBiasValues)
        {
            for(float goalBias : goalBiasValues)
            {
                currentConfig++;
                printf("\n[Configuration %d/%d]\n", currentConfig, totalConfigs);

                TuningResult result;
                runTuningExperiment(
                    maxReg, explBias, goalBias,
                    50,  // numTrials
                    h_initial, h_goal,
                    d_obstacles, numObstacles,
                    result
                );

                results.push_back(result);

                // Save intermediate results
                writeResultsToCSV(results);
            }
        }
    }

    // Cleanup
    cudaFree(d_obstacles);

    printf("\n=======================================================\n");
    printf("    TUNING COMPLETE\n");
    printf("=======================================================\n");
    printf("Total configurations tested: %zu\n", results.size());
    printf("Results saved to: Data/Tuning/prune_kpax_tuning_results.csv\n");
    printf("=======================================================\n");

    return 0;
}
