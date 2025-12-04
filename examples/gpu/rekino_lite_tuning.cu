#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include "ReKino/ReKinoLite.cuh"

/*
 * ReKinoLite Parameter Tuning Script
 *
 * This script explores different parameter combinations:
 * - samplesPerThread: How many controls each thread samples (1, 2, 4, 8, 16)
 * - epsilonGreedy: Probability of random vs greedy selection (0.0, 0.1, 0.2, 0.5)
 *
 * For each configuration, it runs N trials and:
 * 1. Measures execution time
 * 2. Tracks success rate
 * 3. Generates a visualization of one representative tree
 */

struct TuningResult
{
    int samplesPerThread;
    float epsilonGreedy;
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

    std::ofstream file("Data/Tuning/tuning_results.csv");

    // Write header: parameters + run_1, run_2, ..., run_50
    file << "samplesPerThread,epsilonGreedy";
    for(int i = 1; i <= 50; i++)
    {
        file << ",run_" << i;
    }
    file << "\n";

    // Write each configuration's data
    for(const auto& r : results)
    {
        // Write parameters
        file << r.samplesPerThread << "," << r.epsilonGreedy;

        // Write all runtimes (or empty if trial timed out/wasn't run)
        for(size_t i = 0; i < 50; i++)
        {
            file << ",";
            if(i < r.allTimes.size())
            {
                file << r.allTimes[i];
            }
            // Empty cell if this trial wasn't completed
        }
        file << "\n";
    }

    file.close();
    printf("Results written to Data/Tuning/tuning_results.csv\n");
}

void runTuningExperiment(
    int samplesPerThread,
    float epsilonGreedy,
    int numTrials,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    int numObstacles,
    TuningResult& result)
{
    result.samplesPerThread = samplesPerThread;
    result.epsilonGreedy = epsilonGreedy;
    result.trials = numTrials;
    result.successes = 0;
    result.avgTime = 0.0;
    result.minTime = 1e9;
    result.maxTime = 0.0;

    printf("\n=== Testing: samplesPerThread=%d, epsilon=%.2f ===\n",
           samplesPerThread, epsilonGreedy);

    // Create planner and set parameters
    ReKinoLite planner;
    planner.h_samplesPerThread_ = samplesPerThread;
    planner.h_epsilonGreedy_ = epsilonGreedy;

    // Set unique output filename for tree
    std::ostringstream prefix;
    prefix << "s" << samplesPerThread << "_e" << std::fixed << std::setprecision(2) << epsilonGreedy;
    planner.setTreeOutputPrefix(prefix.str());

    std::vector<double> times;
    const double MAX_TIME_SECONDS = 6.0;  // 6 second timeout per trial

    for(int trial = 0; trial < numTrials; trial++)
    {
        cudaEvent_t start, stop;
        float milliseconds = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Run planner - only save tree for first trial with unique filename
        bool saveTree = (trial == 0);
        planner.plan(h_initial, h_goal, d_obstacles, numObstacles, saveTree);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        double seconds = milliseconds / 1000.0;

        // Check for timeout - cap at MAX_TIME_SECONDS but continue
        if(seconds > MAX_TIME_SECONDS)
        {
            printf("  Trial %d TIMEOUT (%.1fs > %.1fs) - recording as %.1fs and continuing\n",
                   trial + 1, seconds, MAX_TIME_SECONDS, MAX_TIME_SECONDS);
            seconds = MAX_TIME_SECONDS;  // Cap at max time
        }

        times.push_back(seconds);
        result.successes++;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Add delay between runs to prevent GPU thermal throttling
        if(trial < numTrials - 1)  // Don't sleep after last trial
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));  // 500ms cooldown
        }

        if((trial + 1) % 10 == 0)
        {
            printf("  Trial %d/%d complete (%.3fs)\n", trial + 1, numTrials, seconds);
        }
    }

    // Store all times
    result.allTimes = times;

    // Compute statistics
    for(double t : times)
    {
        result.avgTime += t;
        if(t < result.minTime) result.minTime = t;
        if(t > result.maxTime) result.maxTime = t;
    }
    result.avgTime /= numTrials;
    result.successRate = (float)result.successes / numTrials;

    printf("  Results: %.2f%% success, avg time: %.3fs (min: %.3fs, max: %.3fs)\n",
           result.successRate * 100.0f, result.avgTime, result.minTime, result.maxTime);
}

int main(void)
{
    // Remove previous data
    system("rm -rf Data/Tuning/*");
    system("rm -rf Data/ReKinoLiteTree/*");

    // Problem setup - same for all trials
    float h_initial[SAMPLE_DIM] = {10.0, 8, 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          h_goal[SAMPLE_DIM]    = {80, 95.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    int numObstacles;
    float* d_obstacles;

    // Load obstacles
    std::vector<float> obstacles = readObstaclesFromCSV(
        "../include/config/obstacles/quadTrees/obstacles.csv",
        numObstacles, W_DIM);

    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(),
               numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);

    printf("=== ReKinoLite Parameter Tuning ===\n");
    printf("Initial: (%.1f, %.1f, %.1f)\n", h_initial[0], h_initial[1], h_initial[2]);
    printf("Goal: (%.1f, %.1f, %.1f)\n", h_goal[0], h_goal[1], h_goal[2]);
    printf("Obstacles: %d\n", numObstacles);
    printf("Warps: 512 (16,384 threads)\n");

    // Parameter ranges to test
    std::vector<int> samplesPerThreadValues = {1, 2, 4};  // Reduced from {1, 2, 4, 8, 16}
    std::vector<float> epsilonValues = {0.0f, 0.1f, 0.2f, 0.5f};
    int trialsPerConfig = 50;

    std::vector<TuningResult> results;

    // Test all combinations
    for(int samples : samplesPerThreadValues)
    {
        for(float epsilon : epsilonValues)
        {
            TuningResult result;
            runTuningExperiment(samples, epsilon, trialsPerConfig,
                              h_initial, h_goal, d_obstacles, numObstacles,
                              result);
            results.push_back(result);
        }
    }

    // Write all results
    writeResultsToCSV(results);

    printf("\n=== Tuning Complete ===\n");
    printf("Total configurations tested: %zu\n", results.size());
    printf("Total trials run: %zu\n", results.size() * trialsPerConfig);
    printf("\nBest configurations:\n");

    // Sort by success rate, then by time
    std::sort(results.begin(), results.end(),
              [](const TuningResult& a, const TuningResult& b) {
                  if(abs(a.successRate - b.successRate) < 0.01)
                      return a.avgTime < b.avgTime;
                  return a.successRate > b.successRate;
              });

    // Print top 5
    for(int i = 0; i < min(5, (int)results.size()); i++)
    {
        const auto& r = results[i];
        printf("  %d. samples=%d, epsilon=%.2f: %.1f%% success, %.3fs avg\n",
               i+1, r.samplesPerThread, r.epsilonGreedy,
               r.successRate * 100.0f, r.avgTime);
    }

    // Cleanup
    cudaFree(d_obstacles);

    return 0;
}
