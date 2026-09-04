// PAPER BENCHMARK -- a fixed, five-way comparison, not a sweep.
//
// countingstars_sweep.cu exists to TUNE CountingStars: it sweeps a grid of its internal knobs and
// logs two dozen door/cutoff/budget diagnostic columns to explain WHY a point on that grid behaves
// the way it does. This tool has a different job: five ALREADY-CHOSEN operating points, run
// identically across every environment and every discretization, to produce the headline figures
// and results table for the paper. There is nothing to tune here, so there is nothing to diagnose --
// the per-iteration CSV carries only what the outcome-comparison plots actually read.
//
// THE FIVE SERIES, fixed for the whole file:
//   KPAX                   defaults
//   KinoPaxPlus             defaults
//   KinoPaxSTARCleanCost    r2 off, w 0.9, k 1.0, cap 0.03 (countingstars_sweep.cu's CLEAN_BASE_*)
//   CountingStars (bs 1.0)  explore_frac 0.3, cost_frac 0.3, bufferFloor 0.1, bufferSlope 1.0
//   CountingStars (bs 0.5)  same, bufferSlope 0.5
//
// EVERY SERIES RUNS AT EVERY DELTA -- unlike countingstars_sweep.cu, where only KinoPaxPlus runs
// past the coarse discretization. There is no --only-kinopaxplus concept here because there is no
// reason to partition: nothing is being swept, so nothing is expensive enough to restrict.
//
// LIMITS ARE THE POINT. MAX_TREE_SIZE and MAX_TIME_MS are meant to be what actually stops a run;
// MAX_ITERATIONS below is set high enough that it should not be. config.h's MAX_ITER macro is
// DELIBERATELY NOT touched by this file. CountingStars' buffer ramp reads h_fillIters_ to compute
// x = itr/h_fillIters_, clamped to [0,1] -- rescaling MAX_ITER would silently rescale B at every
// point on the ramp (it is in the denominator), so that stays put. But h_fillIters_ is NOT left at
// its class default (MAX_ITER) either: at MAX_TREE_SIZE=3,000,000 and a 10s timeout, a real run only
// completes ~700 iterations, well short of MAX_ITER (1000) -- x would never reach 1 and B would
// never reach its ramp maximum for a run's entire duration, not just plateau there. See
// CS_RAMP_FILL_ITERS below, set explicitly in benchmarkCountingStars() to the real run length.
//
// Usage: paper_benchmark <deltaLabel> <obstaclePath> <envName>
// No flags. There is nothing left to opt in or out of.
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
#include <cstdio>
#include "planners/KinoPaxPlus.cuh"
#include "planners/KPAX.cuh"
#include "planners/KinoPaxSTARCleanCost.cuh"
#include "planners/CountingStars.cuh"

// ========================================================================
// Per-iteration record. Four columns -- everything the three plots (Cost vs Time, Tree Growth vs
// Iteration, Tradeoff Scatter) and the summary table read, traced against
// process_countingstars_summary_plots.m, and nothing else. No planner-specific diagnostics: every
// row from every planner has the same four columns, so there is no clearCountingStarsCols()
// equivalent needed -- there is nothing to blank.
// ========================================================================
struct IterationData
{
    int   iteration;
    int   tree_size;
    float elapsed_time_ms;
    float best_cost;
};

struct RunResult
{
    std::string delta_label;   // series identity for the filename: "KPAX", a planner label, or the plain delta
    std::string build_delta;   // this binary's discretization label (large_length, fine_effort, ...)
    std::string environment;
    int run_number;
    double total_time_seconds;
    int first_solution_iteration;
    float first_solution_cost;
    int first_solution_tree_size;
    float final_best_cost;
    int final_tree_size;
    int total_iterations;
    std::vector<IterationData> per_iteration;
};

// ========================================================================
// Cumulative root-to-goal path cost by walking the parent chain -- KPAX carries no running
// h_minCost_, so a goal is detected by zeroing d_pathToGoal_ each iteration and, when a new one
// appears, reconstructing its cost on the host. Identical to countingstars_sweep.cu's version.
// ========================================================================
float computePathCost(const std::vector<float>& h_treeSamples, const std::vector<int>& h_parents, int goalIdx)
{
    float totalCost = 0.0f;
    int cur = goalIdx;
    while(true)
    {
        int par = h_parents[cur];
        if(par < 0) break;
        totalCost += edgeCost(&h_treeSamples[par * SAMPLE_DIM], &h_treeSamples[cur * SAMPLE_DIM]);
        cur = par;
    }
    return totalCost;
}

float devicePathCost(float* d_treeSamples_ptr, int* d_treeSamplesParentIdxs_ptr, int treeSize, int goalIdx)
{
    std::vector<float> h_treeSamples(treeSize * SAMPLE_DIM);
    std::vector<int>   h_parents(treeSize);
    cudaMemcpy(h_treeSamples.data(), d_treeSamples_ptr, treeSize * SAMPLE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_parents.data(), d_treeSamplesParentIdxs_ptr, treeSize * sizeof(int), cudaMemcpyDeviceToHost);
    return computePathCost(h_treeSamples, h_parents, goalIdx);
}

// ========================================================================
// Series-identity labels for the two tuned planners. Same token conventions as
// countingstars_sweep.cu (round(100 x float) for bs/bf/w/k/cap, round(1000 x float) for ef/cf) so
// the plot script's startsWith('CountingStars')/startsWith('KinoPaxSTAR') dispatch needs no changes.
// ========================================================================
std::string countingStarsLabel(float bufferSlope, float bufferFloor, float exploreFrac, float costFrac)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "CountingStars_bs%d_bf%d_ef%d_cf%d",
             (int)lroundf(100.0f * bufferSlope), (int)lroundf(100.0f * bufferFloor),
             (int)lroundf(1000.0f * exploreFrac), (int)lroundf(1000.0f * costFrac));
    return std::string(buf);
}

std::string cleanLabel(bool r2Accept, float w, float k, float cap)
{
    char buf[96];
    snprintf(buf, sizeof(buf), "KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d",
             r2Accept ? "on" : "off", (int)lroundf(100.0f * w), (int)lroundf(100.0f * k), (int)lroundf(100.0f * cap));
    return std::string(buf);
}

// ========================================================================
// Write per-iteration CSV for a single run. Same filename dispatch as countingstars_sweep.cu, minus
// the KPAXCap arm (not in this comparison): KPAX keys on an exact "KPAX" match, CountingStars/
// KinoPaxSTAR variants carry their own label (must, so length/effort builds of the same tuned point
// don't collide), everything else (KinoPaxPlus) keys on the plain delta.
// ========================================================================
void writePerIterationCSV(const RunResult& result, const std::string& outputDir)
{
    std::ostringstream filename;
    if(result.delta_label == "KPAX")
        filename << outputDir << "/" << result.environment << "_KPAX_delta" << result.build_delta
                 << "_run" << result.run_number << ".csv";
    else if(result.delta_label.rfind("CountingStars", 0) == 0 || result.delta_label.rfind("KinoPaxSTAR", 0) == 0)
        filename << outputDir << "/" << result.environment << "_" << result.delta_label << "_delta" << result.build_delta
                 << "_run" << result.run_number << ".csv";
    else
        filename << outputDir << "/" << result.environment << "_delta" << result.delta_label
                 << "_run" << result.run_number << ".csv";

    std::ofstream file(filename.str());
    file << "iteration,tree_size,elapsed_time_ms,best_cost\n";
    for(const auto& d : result.per_iteration)
    {
        file << d.iteration << ","
             << d.tree_size << ","
             << std::fixed << std::setprecision(3) << d.elapsed_time_ms << ","
             << std::fixed << std::setprecision(6) << d.best_cost << "\n";
    }
    file.close();
}

// ========================================================================
// Per-run summary CSV, aggregated across all series/runs for this (delta, environment, metric)
// invocation -- a quick sanity check without needing MATLAB. Same schema as countingstars_sweep.cu's.
// ========================================================================
void writeSummaryCSV(const std::vector<RunResult>& results, const std::string& outputDir, const std::string& deltaLabel)
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");

    std::ostringstream filename;
    filename << outputDir << "/paper_benchmark_" << timestamp.str() << "_summary.csv";

    std::ofstream file(filename.str());
    file << "environment,delta_label,num_regions,run,total_time_s,first_sol_iteration,"
         << "first_sol_cost,first_sol_tree_size,final_best_cost,final_tree_size,total_iterations\n";

    for(const auto& r : results)
    {
        int regions = NUM_R1_REGIONS;
        file << r.environment << ","
             << r.delta_label << ","
             << regions << ","
             << r.run_number << ","
             << std::fixed << std::setprecision(4) << r.total_time_seconds << ","
             << r.first_solution_iteration << ","
             << std::fixed << std::setprecision(6) << r.first_solution_cost << ","
             << r.first_solution_tree_size << ","
             << std::fixed << std::setprecision(6) << r.final_best_cost << ","
             << r.final_tree_size << ","
             << r.total_iterations << "\n";
    }
    file.close();
}

// ========================================================================
// KinoPaxPlus
// ========================================================================
RunResult benchmarkKinoPaxPlus(
    KinoPaxPlus& planner, const std::string& deltaLabel, const std::string& environment, int runNumber,
    float* h_initial, float* h_goal, float* d_obstacles, uint numObstacles, int maxIterations, float maxTimeMs)
{
    RunResult result;
    result.delta_label = deltaLabel;
    result.build_delta = deltaLabel;
    result.environment = environment;
    result.run_number = runNumber;
    result.first_solution_iteration = -1;
    result.first_solution_cost = INFINITY;
    result.first_solution_tree_size = -1;
    result.final_best_cost = INFINITY;

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);
    float plannerMs = 0.0f;
    float iterMs    = 0.0f;

    planner.resetPlanner(h_initial, h_goal);

    int itr = 0;
    while(itr < maxIterations)
    {
        itr++;
        planner.h_itr_++;

        cudaEventRecord(iterStart);
        planner.propagateFrontier(d_obstacles, numObstacles);
        if(planner.h_propIterations_ == 0) break;
        planner.updateFrontier();
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&iterMs, iterStart, iterStop);
        plannerMs += iterMs;

        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
        if(planner.h_minCost_ < MAX_FLOAT && result.first_solution_iteration == -1)
        {
            result.first_solution_iteration = itr;
            result.first_solution_cost      = planner.h_minCost_;
            result.first_solution_tree_size = planner.h_treeSize_;
        }
        if(planner.h_minCost_ < result.final_best_cost)
            result.final_best_cost = planner.h_minCost_;

        IterationData d;
        d.iteration       = itr;
        d.tree_size       = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost       = result.final_best_cost;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        if(plannerMs >= maxTimeMs) break;
    }

    result.total_time_seconds = plannerMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
    return result;
}

void runKinoPaxPlusBenchmark(
    const std::string& environment_name, float* h_initial, float* h_goal, float* d_obstacles, uint numObstacles,
    std::vector<RunResult>& all_results, const std::string& outputDir, const std::string& deltaLabel,
    int numRuns, int maxIterations, float maxTimeMs)
{
    printf("\n========================================\n");
    printf("KINOPAXPLUS: %s | Delta: %s | Regions: %d\n", environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    KinoPaxPlus planner;
    for(int run = 0; run < numRuns; run++)
    {
        RunResult result = benchmarkKinoPaxPlus(planner, deltaLabel, environment_name, run,
                                                 h_initial, h_goal, d_obstacles, numObstacles, maxIterations, maxTimeMs);
        printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f\n",
               run + 1, numRuns, result.total_time_seconds, result.total_iterations,
               result.final_tree_size, result.first_solution_iteration, result.final_best_cost);
        writePerIterationCSV(result, outputDir);
        all_results.push_back(result);

        if(run < numRuns - 1)
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// ========================================================================
// KPAX
// ========================================================================
RunResult benchmarkKPAX(
    KPAX& planner, const std::string& deltaLabel, const std::string& environment, int runNumber,
    float* h_initial, float* h_goal, float* d_obstacles, uint numObstacles, int maxIterations, float maxTimeMs)
{
    RunResult result;
    result.delta_label = "KPAX";
    result.build_delta = deltaLabel;
    result.environment = environment;
    result.run_number = runNumber;
    result.first_solution_iteration = -1;
    result.first_solution_cost = INFINITY;
    result.first_solution_tree_size = -1;
    result.final_best_cost = INFINITY;

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);
    float plannerMs = 0.0f;
    float iterMs    = 0.0f;

    planner.resetPlanner(h_initial, h_goal);

    int zero = 0;
    int itr = 0;
    while(itr < maxIterations)
    {
        itr++;
        planner.h_itr_++;

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

        if(planner.h_pathToGoal_ != 0)
        {
            float pathCost = devicePathCost(planner.d_treeSamples_ptr_, planner.d_treeSamplesParentIdxs_ptr_,
                                            planner.h_treeSize_, planner.h_pathToGoal_);
            if(result.first_solution_iteration == -1)
            {
                result.first_solution_iteration = itr;
                result.first_solution_cost      = pathCost;
                result.first_solution_tree_size = planner.h_treeSize_;
            }
            if(pathCost < result.final_best_cost)
                result.final_best_cost = pathCost;
        }

        IterationData d;
        d.iteration       = itr;
        d.tree_size       = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost       = result.final_best_cost;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        if(plannerMs >= maxTimeMs) break;
    }

    result.total_time_seconds = plannerMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
    return result;
}

void runKPAXBenchmark(
    const std::string& environment_name, float* h_initial, float* h_goal, float* d_obstacles, uint numObstacles,
    std::vector<RunResult>& all_results, const std::string& outputDir, const std::string& deltaLabel,
    int numRuns, int maxIterations, float maxTimeMs)
{
    printf("\n========================================\n");
    printf("KPAX: %s | Delta: %s | Regions: %d\n", environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    KPAX planner;
    for(int run = 0; run < numRuns; run++)
    {
        RunResult result = benchmarkKPAX(planner, deltaLabel, environment_name, run,
                                         h_initial, h_goal, d_obstacles, numObstacles, maxIterations, maxTimeMs);
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

// ========================================================================
// KinoPaxSTARCleanCost -- one fixed, well-tuned point (r2 off, w 0.9, k 1.0, cap 0.03).
// ========================================================================
RunResult benchmarkKinoPaxSTARCleanCost(
    KinoPaxSTARCleanCost& planner, const std::string& deltaLabel, const std::string& environment, int runNumber,
    float* h_initial, float* h_goal, float* d_obstacles, uint numObstacles, int maxIterations, float maxTimeMs,
    float costWeight, float costPruneExp, float acceptCapMul, bool r2SeedAccept, const std::string& label)
{
    planner.h_costWeight_   = costWeight;
    planner.h_costPruneExp_ = costPruneExp;
    planner.h_acceptCapMul_ = acceptCapMul;
    planner.h_r2SeedAccept_ = r2SeedAccept;

    RunResult result;
    result.delta_label = label;
    result.build_delta = deltaLabel;
    result.environment = environment;
    result.run_number = runNumber;
    result.first_solution_iteration = -1;
    result.first_solution_cost = INFINITY;
    result.first_solution_tree_size = -1;
    result.final_best_cost = INFINITY;

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);
    float plannerMs = 0.0f;
    float iterMs    = 0.0f;

    planner.resetPlanner(h_initial, h_goal);

    int itr = 0;
    while(itr < maxIterations)
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
        if(planner.h_minCost_ < MAX_FLOAT && result.first_solution_iteration == -1)
        {
            result.first_solution_iteration = itr;
            result.first_solution_cost      = planner.h_minCost_;
            result.first_solution_tree_size = planner.h_treeSize_;
        }
        if(planner.h_minCost_ < result.final_best_cost)
            result.final_best_cost = planner.h_minCost_;

        IterationData d;
        d.iteration       = itr;
        d.tree_size       = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost       = result.final_best_cost;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        if(planner.h_propIterations_ == 0) break;
        if(plannerMs >= maxTimeMs) break;
    }

    result.total_time_seconds = plannerMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
    return result;
}

void runKinoPaxSTARCleanCostBenchmark(
    const std::string& environment_name, float* h_initial, float* h_goal, float* d_obstacles, uint numObstacles,
    std::vector<RunResult>& all_results, const std::string& outputDir, const std::string& deltaLabel,
    int numRuns, int maxIterations, float maxTimeMs)
{
    printf("\n========================================\n");
    printf("KINOPAXSTARCLEANCOST: %s | Delta: %s | Regions: %d\n", environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    const bool  r2  = false;
    const float w   = 0.9f;
    const float k   = 1.0f;
    const float cap = 0.03f;
    const std::string label = cleanLabel(r2, w, k, cap);

    printf("  --- r2 = %s, w = %.2f, k = %.2f, cap = %.2f (%s) ---\n", r2 ? "on" : "off", w, k, cap, label.c_str());
    KinoPaxSTARCleanCost planner;
    for(int run = 0; run < numRuns; run++)
    {
        RunResult result = benchmarkKinoPaxSTARCleanCost(planner, deltaLabel, environment_name, run,
                                             h_initial, h_goal, d_obstacles, numObstacles, maxIterations, maxTimeMs,
                                             w, k, cap, r2, label);
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

// How many iterations a run actually completes inside the 10s wall-clock cap at
// MAX_TREE_SIZE = 3,000,000 (empirical). MUST MATCH countingstars_sweep.cu's copy of this same
// constant -- that sweep is what finds a good (bufferSlope, bufferFloor) point, and this fixed pair
// is meant to reproduce it; a mismatched fill_iters would make the same (slope, floor) mean a
// different ramp in each binary.
static const int CS_RAMP_FILL_ITERS = 700;

// ========================================================================
// CountingStars -- two fixed points (bufferSlope 1.0 and 0.5), same explore_frac/cost_frac/
// bufferFloor. h_fillIters_ is set explicitly to CS_RAMP_FILL_ITERS, not left at its class default
// (MAX_ITER): the ramp's x = itr/h_fillIters_ must track the REAL run length, and at
// MAX_TREE_SIZE=3,000,000 with a 10s timeout that's ~700 iterations, well short of MAX_ITER (1000)
// -- left at the default, x would never reach 1 and B would never reach its ramp maximum for a
// run's entire duration.
// ========================================================================
RunResult benchmarkCountingStars(
    CountingStars& planner, const std::string& deltaLabel, const std::string& environment, int runNumber,
    float* h_initial, float* h_goal, float* d_obstacles, uint numObstacles, int maxIterations, float maxTimeMs,
    float bufferSlope, float bufferFloor, float exploreFrac, float costFrac, const std::string& label)
{
    planner.h_fillIters_   = CS_RAMP_FILL_ITERS;
    planner.h_bufferSlope_ = bufferSlope;
    planner.h_bufferFloor_ = bufferFloor;
    planner.h_exploreFrac_ = exploreFrac;
    planner.h_costFrac_    = costFrac;

    RunResult result;
    result.delta_label = label;
    result.build_delta = deltaLabel;
    result.environment = environment;
    result.run_number = runNumber;
    result.first_solution_iteration = -1;
    result.first_solution_cost = INFINITY;
    result.first_solution_tree_size = -1;
    result.final_best_cost = INFINITY;

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);
    float plannerMs = 0.0f;
    float iterMs    = 0.0f;

    planner.resetPlanner(h_initial, h_goal);

    int itr = 0;
    while(itr < maxIterations)
    {
        itr++;
        planner.h_itr_++;

        cudaEventRecord(iterStart);
        planner.propagateFrontier(d_obstacles, numObstacles);
        // NO graph_.updateVertices() HERE -- CountingStars consumes nothing it produces. See
        // CountingStars.cu's own comment on this; it is not free, and the other planners genuinely
        // use what it computes.
        planner.updateFrontier();
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&iterMs, iterStart, iterStop);
        plannerMs += iterMs;

        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
        if(planner.h_minCost_ < MAX_FLOAT && result.first_solution_iteration == -1)
        {
            result.first_solution_iteration = itr;
            result.first_solution_cost      = planner.h_minCost_;
            result.first_solution_tree_size = planner.h_treeSize_;
        }
        if(planner.h_minCost_ < result.final_best_cost)
            result.final_best_cost = planner.h_minCost_;

        IterationData d;
        d.iteration       = itr;
        d.tree_size       = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost       = result.final_best_cost;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        if(planner.h_propIterations_ == 0) break;
        if(plannerMs >= maxTimeMs) break;
    }

    result.total_time_seconds = plannerMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
    return result;
}

void runCountingStarsBenchmark(
    const std::string& environment_name, float* h_initial, float* h_goal, float* d_obstacles, uint numObstacles,
    std::vector<RunResult>& all_results, const std::string& outputDir, const std::string& deltaLabel,
    int numRuns, int maxIterations, float maxTimeMs)
{
    static const float BUFFER_SLOPES[] = {1.0f, 0.5f};
    static const float BUFFER_FLOOR    = 0.1f;
    static const float EXPLORE_FRAC    = 0.3f;
    static const float COST_FRAC       = 0.3f;

    printf("\n========================================\n");
    printf("COUNTINGSTARS: %s | Delta: %s | Regions: %d\n", environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    for(float bufferSlope : BUFFER_SLOPES)
    {
        const std::string label = countingStarsLabel(bufferSlope, BUFFER_FLOOR, EXPLORE_FRAC, COST_FRAC);
        printf("  --- bufferSlope = %.2f, bufferFloor = %.2f, explore_frac = %.2f, cost_frac = %.2f (%s) ---\n",
               bufferSlope, BUFFER_FLOOR, EXPLORE_FRAC, COST_FRAC, label.c_str());
        CountingStars planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkCountingStars(planner, deltaLabel, environment_name, run,
                                                 h_initial, h_goal, d_obstacles, numObstacles, maxIterations, maxTimeMs,
                                                 bufferSlope, BUFFER_FLOOR, EXPLORE_FRAC, COST_FRAC, label);
            printf("  bs=%.2f Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
                   bufferSlope, run + 1, numRuns, result.total_time_seconds,
                   result.total_iterations, result.final_tree_size, result.first_solution_iteration,
                   result.first_solution_cost, result.final_best_cost);
            writePerIterationCSV(result, outputDir);
            all_results.push_back(result);

            if(run < numRuns - 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

// ========================================================================
// main -- paper_benchmark <deltaLabel> <obstaclePath> <envName>
// ========================================================================
int main(int argc, char* argv[])
{
    std::string deltaLabel   = (argc > 1) ? argv[1] : "unknown";
    std::string obstaclePath = (argc > 2) ? argv[2] : "../include/config/obstacles/empty/obstacles.csv";
    std::string envName      = (argc > 3) ? argv[3] : "empty";

    const int   NUM_RUNS        = 10;
    const int   MAX_ITERATIONS  = 20000;      // non-binding -- see the file header
    const float MAX_TIME_MS     = 10000.0f;   // 10 second per-run timeout

    std::string outputDir = "Data/Benchmarks/Paper/" + envName;
    std::filesystem::create_directories(outputDir);

    printf("=======================================================\n");
    printf("    PAPER BENCHMARK\n");
    printf("=======================================================\n");
    printf("Delta label:    %s\n", deltaLabel.c_str());
    printf("NUM_R1_REGIONS: %d\n", NUM_R1_REGIONS);
    printf("MAX_TREE_SIZE:  %d\n", MAX_TREE_SIZE);
    printf("W_R1_LENGTH=%d  C_R1_LENGTH=%d  V_R1_LENGTH=%d\n", W_R1_LENGTH, C_R1_LENGTH, V_R1_LENGTH);
    printf("Obstacle file:  %s\n", obstaclePath.c_str());
    printf("Environment:    %s\n", envName.c_str());
    printf("Cost metric:    %s (COST_MODE=%d)\n", (COST_MODE == 1) ? "control effort" : "workspace path length", COST_MODE);
    printf("Runs per series: %d\n", NUM_RUNS);
    printf("Max iterations: %d (non-binding; MAX_TREE_SIZE / MAX_TIME_MS are the real limiters)\n", MAX_ITERATIONS);
    printf("Max time:       %.1f s\n", MAX_TIME_MS / 1000.0f);
    printf("Series:         KPAX, KinoPaxPlus, KinoPaxSTARCleanCost (w0.9 k1.0 cap0.03),\n");
    printf("                CountingStars (bufferSlope 1.0), CountingStars (bufferSlope 0.5)\n");
    printf("                explore_frac 0.3, cost_frac 0.3, bufferFloor 0.1 for both CountingStars points\n");
    printf("=======================================================\n");

    // Start/goal states -- identical to countingstars_sweep.cu's, validated across every
    // environment in this comparison including the newly-tightened zigzag.
    float h_initial[SAMPLE_DIM] = {0};
    float h_goal[SAMPLE_DIM]    = {0};
    h_initial[0] = W_MIN + 0.1f * W_SIZE;
    h_initial[1] = W_MIN + 0.08f * W_SIZE;
    h_initial[2] = W_MIN + 0.05f * W_SIZE;
    h_goal[0]    = W_MIN + 0.8f * W_SIZE;
    h_goal[1]    = W_MIN + 0.95f * W_SIZE;
    h_goal[2]    = W_MIN + 0.9f * W_SIZE;

    int numObstacles;
    float* d_obstacles;
    std::vector<float> obstacles = readObstaclesFromCSV(obstaclePath, numObstacles, W_DIM);
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(), numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);
    printf("Loaded %d obstacles from %s\n", numObstacles, obstaclePath.c_str());

    std::vector<RunResult> all_results;

    runKPAXBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                     all_results, outputDir, deltaLabel, NUM_RUNS, MAX_ITERATIONS, MAX_TIME_MS);
    runKinoPaxPlusBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                            all_results, outputDir, deltaLabel, NUM_RUNS, MAX_ITERATIONS, MAX_TIME_MS);
    runKinoPaxSTARCleanCostBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                            all_results, outputDir, deltaLabel, NUM_RUNS, MAX_ITERATIONS, MAX_TIME_MS);
    runCountingStarsBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                            all_results, outputDir, deltaLabel, NUM_RUNS, MAX_ITERATIONS, MAX_TIME_MS);

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
