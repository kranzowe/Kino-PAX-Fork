// ChainReaction vs CountingStars: a pared-down paper_benchmark_v3.cu with just those two planners.
//
// Same CLI shape and output layout as paper_benchmark_v3.cu, so plots/zephyr_common.load_runs reads it:
//   ./ChainReactionBenchmark <tag> <obstacleCsv> <env> <deltaFolder> [--only-chainreaction|--only-countingstars]
//                            [--runs N] [--budget B]
// e.g. tag = "m2_large_length", deltaFolder = "COARSE". Per-run CSVs land in
//   Data/Benchmarks/ChainReaction/<deltaFolder>/<env>/<env>_<planner label>_delta<tag>_run<N>.csv
//
// CountingStars runs at the paper's fixed point (same constants as paper_benchmark_v3.cu). ChainReaction
// runs with a single constant candidate budget B. The per-iteration CSV is a reduced header (just the
// columns the plots read).
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
#include <cstdlib>
#include <algorithm>
#include "planners/ChainReaction.cuh"
#include "planners/CountingStars.cuh"

// CountingStars at paper_benchmark_v3.cu's fixed point (1M-node tree, 3s cap -> ramp length 210).
static const float CS_BUFFER_SLOPE   = 1.2f;
static const float CS_BUFFER_FLOOR   = 0.4f;
static const float CS_EXPLORE_FRAC   = 0.15f;
static const float CS_COST_FRAC      = 0.75f;
static const int   CS_HOPELESS_GUARD = 1;
static const int   CS_RAMP_FILL_ITERS = 210;

// ChainReaction's constant candidate budget; override with --budget. 5000 is about the mean of
// CountingStars' B ramp at this tree size (1.9k -> 7.6k).
static const int CR_BUDGET_DEFAULT = 5000;

static const int   MAX_ITERATIONS = 1000;
static const float MAX_TIME_MS    = 3000.0f;

static const char* CR_LABEL = "ChainReaction";

static std::string countingStarsLabel()
{
    char buf[176];
    snprintf(buf, sizeof(buf), "CountingStars_bs%d_bf%d_ef%d_cf%d_hg%d",
             (int)lroundf(100.0f * CS_BUFFER_SLOPE),
             (int)lroundf(100.0f * CS_BUFFER_FLOOR),
             (int)lroundf(1000.0f * CS_EXPLORE_FRAC),
             (int)lroundf(1000.0f * CS_COST_FRAC),
             CS_HOPELESS_GUARD);
    return std::string(buf);
}

struct IterRow
{
    int iteration;
    int frontier_size;
    int tree_size;
    float elapsed_time_ms;
    float best_cost;
    int prop_attempted;
    int goal_frontier_size;   // B: CountingStars' ramp value, or ChainReaction's constant budget
};

struct RunResult
{
    std::string label;
    std::string tag;
    std::string environment;
    int run_number;
    std::vector<IterRow> rows;
    int first_solution_iteration;
    float first_solution_cost;
    float final_best_cost;
    int final_tree_size;
    int total_iterations;
    double total_time_seconds;
    std::string stop_reason;
    int invariant_violations = 0;   // iterations where tree growth != accepted candidates
};

static void writePerIterationCSV(const RunResult& r, const std::string& outputDir)
{
    std::ostringstream filename;
    filename << outputDir << "/" << r.environment << "_" << r.label << "_delta" << r.tag << "_run" << r.run_number << ".csv";
    std::ofstream file(filename.str());
    file << "iteration,frontier_size,tree_size,elapsed_time_ms,best_cost,prop_attempted,goal_frontier_size\n";
    for(const auto& d : r.rows)
    {
        file << d.iteration << "," << d.frontier_size << "," << d.tree_size << ","
             << std::fixed << std::setprecision(3) << d.elapsed_time_ms << ","
             << std::fixed << std::setprecision(6) << d.best_cost << ","
             << d.prop_attempted << "," << d.goal_frontier_size << "\n";
    }
}

// Shared loop shape: time propagate+update with cudaEvents (like paper_benchmark_v3.cu's
// benchmarkCountingStars), log one row per iteration, stop on tree-full / planner stop / time cap.
template <typename P, typename BudgetFn>
static RunResult runOne(P& planner, const std::string& label, const std::string& tag, const std::string& env,
                        int runNumber, float* h_initial, float* h_goal, float* d_obstacles, uint numObstacles,
                        BudgetFn budgetOf)
{
    RunResult result;
    result.label                    = label;
    result.tag                      = tag;
    result.environment              = env;
    result.run_number               = runNumber;
    result.first_solution_iteration = -1;
    result.first_solution_cost      = INFINITY;
    result.final_best_cost          = INFINITY;
    result.stop_reason              = "max_iterations";

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);
    float plannerMs = 0.0f;
    float iterMs    = 0.0f;

    planner.resetPlanner(h_initial, h_goal);

    int itr = 0;
    while(itr < MAX_ITERATIONS)
    {
        itr++;
        planner.h_itr_++;

        uint oldTreeSize = planner.h_treeSize_;
        cudaEventRecord(iterStart);
        planner.propagateFrontier(d_obstacles, numObstacles);
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
        }
        if(planner.h_minCost_ < result.final_best_cost)
            result.final_best_cost = planner.h_minCost_;

        // Every accepted candidate becomes exactly one new tree node.
        if(planner.h_propIterations_ != 0 && planner.h_treeSize_ - oldTreeSize != planner.h_frontierNextSize_)
            result.invariant_violations++;

        IterRow d;
        d.iteration          = itr;
        d.frontier_size      = (int)planner.h_frontierSize_;
        d.tree_size          = (int)planner.h_treeSize_;
        d.elapsed_time_ms    = plannerMs;
        d.best_cost          = result.final_best_cost;
        d.prop_attempted     = (int)planner.h_propAttempted_;
        d.goal_frontier_size = budgetOf(planner);
        result.rows.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) { result.stop_reason = "tree_full"; break; }
        if(planner.h_propIterations_ == 0)           { result.stop_reason = "planner_stopped"; break; }
        if(plannerMs >= MAX_TIME_MS)                 { result.stop_reason = "time_cap"; break; }
    }

    result.total_time_seconds = plannerMs / 1000.0;
    result.final_tree_size    = (int)planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
    return result;
}

static void report(const RunResult& r, int run, int numRuns)
{
    printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f, stop=%s%s\n",
           run + 1, numRuns, r.total_time_seconds, r.total_iterations, r.final_tree_size,
           r.first_solution_iteration, r.first_solution_cost, r.final_best_cost, r.stop_reason.c_str(),
           r.invariant_violations ? "  ** tree-growth invariant violated **" : "");
}

int main(int argc, char* argv[])
{
    std::string tag          = (argc > 1) ? argv[1] : "unknown";
    std::string obstaclePath = (argc > 2) ? argv[2] : "../include/config/obstacles/zigzag/obstacles.csv";
    std::string envName      = (argc > 3) ? argv[3] : "zigzag";
    std::string deltaFolder  = (argc > 4) ? argv[4] : "unknown";

    bool onlyCR = false, onlyCS = false;
    int numRuns = 5;
    int budget  = CR_BUDGET_DEFAULT;
    for(int i = 5; i < argc; i++)
    {
        std::string a = argv[i];
        if(a == "--only-chainreaction")      onlyCR = true;
        else if(a == "--only-countingstars") onlyCS = true;
        else if(a == "--runs" && i + 1 < argc)   numRuns = std::max(1, atoi(argv[++i]));
        else if(a == "--budget" && i + 1 < argc) budget  = std::max(1, atoi(argv[++i]));
    }
    if(onlyCR && onlyCS)
    {
        fprintf(stderr, "ERROR: pass at most one --only-* flag.\n");
        return 1;
    }
    bool runCR = !onlyCS;
    bool runCS = !onlyCR;

    std::string outputDir = "Data/Benchmarks/ChainReaction/" + deltaFolder + "/" + envName;
    std::filesystem::create_directories(outputDir);

    printf("=======================================================\n");
    printf("    CHAIN REACTION BENCHMARK\n");
    printf("=======================================================\n");
    printf("Tag: %s | Env: %s | Folder: %s | Runs: %d\n", tag.c_str(), envName.c_str(), deltaFolder.c_str(), numRuns);
    printf("MAX_TREE_SIZE: %d | Max iterations: %d | Time cap: %.0f ms\n", MAX_TREE_SIZE, MAX_ITERATIONS, MAX_TIME_MS);
    if(runCR) printf("ChainReaction: constant candidate budget B = %d\n", budget);
    if(runCS) printf("CountingStars: %s (ramp length %d)\n", countingStarsLabel().c_str(), CS_RAMP_FILL_ITERS);
    printf("=======================================================\n");

    // Start/goal in workspace coordinates, identical to paper_benchmark_v3.cu.
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
    for(float& coord : obstacles)
        coord = W_MIN + coord * W_SIZE;
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(), numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);
    printf("Loaded %d obstacles from %s\n", numObstacles, obstaclePath.c_str());

    if(runCR)
    {
        printf("\nCHAINREACTION: %s | Regions (ignored by this planner): %d\n", envName.c_str(), NUM_R1_REGIONS);
        ChainReaction planner;
        planner.h_budget_ = budget;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult r = runOne(planner, CR_LABEL, tag, envName, run, h_initial, h_goal, d_obstacles, numObstacles,
                                 [](ChainReaction& p) { return p.h_budget_; });
            report(r, run, numRuns);
            writePerIterationCSV(r, outputDir);
            if(run < numRuns - 1) std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    if(runCS)
    {
        const std::string label = countingStarsLabel();
        printf("\nCOUNTINGSTARS: %s | Regions: %d\n", envName.c_str(), NUM_R1_REGIONS);
        CountingStars planner;
        // Tunables must be set before resetPlanner(), which does not touch them.
        planner.h_fillIters_     = CS_RAMP_FILL_ITERS;
        planner.h_bufferSlope_   = CS_BUFFER_SLOPE;
        planner.h_bufferFloor_   = CS_BUFFER_FLOOR;
        planner.h_exploreFrac_   = CS_EXPLORE_FRAC;
        planner.h_costFrac_      = CS_COST_FRAC;
        planner.h_hopelessGuard_ = (CS_HOPELESS_GUARD != 0);
        for(int run = 0; run < numRuns; run++)
        {
            RunResult r = runOne(planner, label, tag, envName, run, h_initial, h_goal, d_obstacles, numObstacles,
                                 [](CountingStars& p) { return p.h_goalFrontierSize_; });
            report(r, run, numRuns);
            writePerIterationCSV(r, outputDir);
            if(run < numRuns - 1) std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    cudaFree(d_obstacles);

    printf("\n=======================================================\n");
    printf("    CHAIN REACTION BENCHMARK COMPLETE\n");
    printf("=======================================================\n");
    printf("Results saved to: %s\n", outputDir.c_str());
    return 0;
}
