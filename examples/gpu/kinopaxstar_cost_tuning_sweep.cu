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
#include "planners/KinoPaxSTARcostprune.cuh"
#include "planners/KinoPaxSTARWeightedCost.cuh"
#include "planners/KinoPaxSTARNoPruneAncestor.cuh"
#include <thrust/count.h>
#include <thrust/reduce.h>

// --- Visualization dump (opt-in via --dump-viz); set in main(), read by the runners. ---
static bool        g_dumpViz = false;
static std::string g_vizDir;

// ---- KinoPaxSTARWeightedCost grid ----
// P_combined = min(1, w*(vertexScore + fAccept) + (1-w)*costProbExp(k) + P_floor).
// w = 1 reproduces KPAX's acceptance, w = 0 is pure cost-greedy -- so the w axis brackets the
// exploration/cost tradeoff with one knob. k is the decay rate of costProbExp.
//
// The ancestor axis is now node-only (1) vs. the memoized chain (2) -- mode 0 (off) was dropped
// once the grid confirmed pruning helps, so this pass is a head-to-head between the two pruning
// rules rather than a prune/no-prune test. Note that leaves the weighted grid without an
// ancestor-off control of its own; KinoPaxSTARNoPruneAncestor_off is the nearest reference, but
// it sits on the NoPrune base, not this one.
static const float WEIGHTS[]       = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
static const float WEIGHTED_EXPS[] = {0.5f, 1.0f, 2.0f};
static const int   WEIGHTED_ANC[]  = {1, 2};
static const int   NUM_WEIGHTS       = sizeof(WEIGHTS) / sizeof(WEIGHTS[0]);
static const int   NUM_WEIGHTED_EXPS = sizeof(WEIGHTED_EXPS) / sizeof(WEIGHTED_EXPS[0]);
static const int   NUM_WEIGHTED_ANC  = sizeof(WEIGHTED_ANC) / sizeof(WEIGHTED_ANC[0]);

// ---- KinoPaxSTARNoPruneAncestor modes (ancestor pruning on the NoPrune base) ----
// Based on KinoPaxSTARNoPrune, not KinoPaxSTAR: NoPrune reactivates with the plain KPAX rule
// `vertexScores + fAccept` (ADDITIVE), so it keeps KPAX-equivalent exploration in cluttered
// maps. That isolates ancestor pruning as the only variable.
// 0 = off, which reproduces stock KinoPaxSTARNoPrune exactly and is the control arm.
// 1 = node-only: prune a node beaten in its own region.
// 2 = memoized ancestor chain: prune if any ancestor is beaten (one parent lookup).
static const int         ANCESTOR_MODES[]  = {0, 1, 2};
static const char* const ANCESTOR_LABELS[] = {"KinoPaxSTARNoPruneAncestor_off",
                                              "KinoPaxSTARNoPruneAncestor_node",
                                              "KinoPaxSTARNoPruneAncestor_chain"};
static const int NUM_ANCESTOR_MODES = sizeof(ANCESTOR_MODES) / sizeof(ANCESTOR_MODES[0]);

// ---- Union-blend cost-prune probes ----
// h_reactivationBlend_ = 1 swaps `costProb * syclop` for `fmaxf(costProb, syclop)`, restoring the
// additive fAccept floor that KPAX has and the product form destroys. It MUST be paired with
// costPruneFloor = 0: costKeepProb ends in fmaxf(p, floor), so under the union a non-zero floor
// becomes a blanket reactivation probability for every dormant node in the tree.
static const float       UNION_CAPS[]   = {0.0f, 1.0f};
static const char* const UNION_LABELS[] = {"KinoPaxSTARcostprune_union_cap0_exp50_floor0",
                                           "KinoPaxSTARcostprune_union_cap100_exp50_floor0"};
static const float UNION_EXP   = 0.5f;
static const float UNION_FLOOR = 0.0f;   // required: see above
static const int   NUM_UNION_POINTS = sizeof(UNION_CAPS) / sizeof(UNION_CAPS[0]);

// "KinoPaxSTARWeightedCost_w50_k100_anc2"
static std::string weightedLabel(float w, float k, int anc)
{
    char buf[96];
    snprintf(buf, sizeof(buf), "KinoPaxSTARWeightedCost_w%d_k%d_anc%d",
             (int)lroundf(100.0f * w), (int)lroundf(100.0f * k), anc);
    return std::string(buf);
}

struct IterationData
{
    int iteration;
    int frontier_size;
    int tree_size;
    float elapsed_time_ms;
    float best_cost;
    int num_regions;          // NUM_R1_REGIONS for this build (all planners share it)
    float r2_coverage_pct;    // % of R2 sub-regions ever activated (KPAX-family planners; NaN otherwise)
    float mean_vertex_score;  // mean Syclop region score (KPAX-family planners; NaN otherwise)
    int reactivated;          // dormant tree nodes re-added to frontier this iter (KPAX-family planners; -1 otherwise)
};

struct RunResult
{
    std::string delta_label;   // planner identity: "KPAX", a tuning-grid label, or the KinoPaxPlus delta label
    std::string build_delta;   // discretization label of this binary (for filename disambiguation)
    std::string environment;
    int run_number;
    double total_time_seconds;
    int first_solution_iteration;
    float first_solution_cost;
    int first_solution_tree_size;   // tree size at the iteration the first solution was found
    float final_best_cost;
    int final_tree_size;
    int total_iterations;
    std::vector<IterationData> per_iteration;
};

// ========================================================================
// Compute cumulative root-to-goal path cost by walking the parent chain, summing edgeCost()
// per edge (COST_MODE-selected; control effort by default). Same metric KinoPaxPlus/STAR
// track via h_minCost_.
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

        // Edge par->cur cost via the shared cost function (control effort under COST_MODE=1;
        // the control that produced 'cur' is stored in cur's sample). Matches the kernels.
        totalCost += edgeCost(&h_treeSamples[par * SAMPLE_DIM],
                              &h_treeSamples[cur * SAMPLE_DIM]);
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
// VISUALIZATION DUMP (opt-in via --dump-viz)
//
// Dump one run's full tree to CSV for the spatial / tree-growth viz.
// One row per node: idx,x,y,z,vx,vy,vz,parent,cost  (state columns only).
// Node idx == insertion order for every variant (the tree only appends;
// pruning tombstones nodes in place, no compaction), so MATLAB reconstructs
// growth from idx + the existing per-iteration tree_size, and density per R1
// region by binning x,y,z. Model 1 state layout: [x,y,z,vx,vy,vz,...].
// ========================================================================
void dumpTreeCSV(float* d_treeSamples_ptr, int* d_parents_ptr, float* d_costs_ptr,
                 int treeSize, const std::string& path)
{
    std::vector<float> h_treeSamples((size_t)treeSize * SAMPLE_DIM);
    std::vector<int>   h_parents(treeSize);
    std::vector<float> h_costs(treeSize);
    cudaMemcpy(h_treeSamples.data(), d_treeSamples_ptr,
               (size_t)treeSize * SAMPLE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_parents.data(), d_parents_ptr, treeSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_costs.data(), d_costs_ptr, treeSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream file(path);
    file << "idx,x,y,z,vx,vy,vz,parent,cost\n";
    file << std::fixed << std::setprecision(6);
    for(int i = 0; i < treeSize; i++)
    {
        const float* s = &h_treeSamples[(size_t)i * SAMPLE_DIM];
        file << i;
        for(int d = 0; d < 6; d++) file << "," << s[d];   // x,y,z,vx,vy,vz
        file << "," << h_parents[i] << "," << h_costs[i] << "\n";
    }
    file.close();
    printf("  [viz] tree dumped: %s (%d nodes)\n", path.c_str(), treeSize);
}

// Build the per-variant tree-dump path: {vizDir}/{env}_{token}_tree.csv
std::string vizTreePath(const std::string& vizDir, const std::string& env, const std::string& token)
{
    return vizDir + "/" + env + "_" + token + "_tree.csv";
}

// Write a small numeric metadata row so the MATLAB script knows the R1 grid,
// workspace/velocity bounds, and start/goal without re-parsing config.h.
void writeVizMeta(const std::string& path, const float* h_initial, const float* h_goal)
{
    std::ofstream file(path);
    file << "W_DIM,W_R1_LENGTH,V_R1_LENGTH,W_MIN,W_MAX,V_MIN,V_MAX,STATE_DIM,SAMPLE_DIM,"
         << "start_x,start_y,start_z,goal_x,goal_y,goal_z\n";
    file << std::fixed << std::setprecision(6)
         << W_DIM << "," << W_R1_LENGTH << "," << V_R1_LENGTH << ","
         << W_MIN << "," << W_MAX << "," << V_MIN << "," << V_MAX << ","
         << STATE_DIM << "," << SAMPLE_DIM << ","
         << h_initial[0] << "," << h_initial[1] << "," << h_initial[2] << ","
         << h_goal[0] << "," << h_goal[1] << "," << h_goal[2] << "\n";
    file.close();
    printf("  [viz] meta written: %s\n", path.c_str());
}

// ========================================================================
// Write per-iteration CSV for a single run
// ========================================================================
void writePerIterationCSV(const RunResult& result, const std::string& outputDir)
{
    std::ostringstream filename;
    // Baselines include the build's delta label so runs at different discretizations
    // don't overwrite each other:
    //   KPAX baseline: {env}_KPAX_delta{build}_run{n}.csv
    //   tuning grid:   {env}_KinoPaxSTARcostprune_cap{N}_exp{N}_floor{N}_delta{build}_run{n}.csv
    //   ancestor:      {env}_KinoPaxSTARNoPruneAncestor_{off,node,chain}_delta{build}_run{n}.csv
    //   union probes:  {env}_KinoPaxSTARcostprune_union_cap{N}_exp50_floor0_delta{build}_run{n}.csv
    //   weighted grid: {env}_KinoPaxSTARWeightedCost_w{N}_k{N}_anc{N}_delta{build}_run{n}.csv
    //   KinoPaxPlus:   {env}_delta{label}_run{n}.csv
    // The build label carries the cost metric (large_effort / large_length), which is a
    // compile-time property of the binary -- see COST_MODE in helper.cuh.
    if(result.delta_label == "KPAX")
        filename << outputDir << "/" << result.environment << "_KPAX_delta" << result.build_delta
                 << "_run" << result.run_number << ".csv";
    else if(result.delta_label.rfind("KinoPaxSTARcostprune", 0) == 0 ||
            result.delta_label.rfind("KinoPaxSTARNoPruneAncestor", 0) == 0 ||
            result.delta_label.rfind("KinoPaxSTARWeightedCost", 0) == 0)
        filename << outputDir << "/" << result.environment << "_" << result.delta_label << "_delta" << result.build_delta
                 << "_run" << result.run_number << ".csv";
    else
        filename << outputDir << "/" << result.environment << "_delta" << result.delta_label
                 << "_run" << result.run_number << ".csv";

    std::ofstream file(filename.str());
    file << "iteration,frontier_size,tree_size,elapsed_time_ms,best_cost,"
         << "num_regions,r2_coverage_pct,mean_vertex_score,reactivated\n";

    for(const auto& d : result.per_iteration)
    {
        file << d.iteration << ","
             << d.frontier_size << ","
             << d.tree_size << ","
             << std::fixed << std::setprecision(3) << d.elapsed_time_ms << ","
             << std::fixed << std::setprecision(6) << d.best_cost << ","
             << d.num_regions << ","
             << std::fixed << std::setprecision(3) << d.r2_coverage_pct << ","
             << std::fixed << std::setprecision(6) << d.mean_vertex_score << ","
             << d.reactivated << "\n";
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
    filename << outputDir << "/cost_tuning_sweep_" << timestamp.str() << "_summary.csv";

    std::ofstream file(filename.str());
    file << "environment,delta_label,num_regions,run,total_time_s,first_sol_iteration,"
         << "first_sol_cost,first_sol_tree_size,final_best_cost,final_tree_size,total_iterations\n";

    for(const auto& r : results)
    {
        int regions = NUM_R1_REGIONS;  // all planners compile under the same discretization
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
    result.build_delta = deltaLabel;
    result.environment = environment;
    result.run_number = runNumber;
    result.first_solution_iteration = -1;
    result.first_solution_cost = INFINITY;
    result.first_solution_tree_size = -1;
    result.final_best_cost = INFINITY;

    // Per-iteration planner-only timing: only propagate+update is inside the timed
    // window, so between-iteration host reads never inflate elapsed_time_ms.
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

        // h_minCost_ is the cumulative root-to-goal path length
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
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.num_regions       = NUM_R1_REGIONS;
        d.r2_coverage_pct   = NAN;   // KinoPaxPlus has no R2/vertexScore machinery
        d.mean_vertex_score = NAN;
        d.reactivated       = -1;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;

        // Timeout check (planner-only time)
        if(plannerMs >= maxTimeMs) break;
    }

    result.total_time_seconds = plannerMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(iterStart);
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
    result.delta_label = "KPAX";
    result.build_delta = deltaLabel;
    result.environment = environment;
    result.run_number = runNumber;
    result.first_solution_iteration = -1;
    result.first_solution_cost = INFINITY;
    result.first_solution_tree_size = -1;
    result.final_best_cost = INFINITY;

    // Per-iteration planner-only timing (diagnostics + path-cost walks excluded).
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

        // Reset pathToGoal before each iteration so we can detect new goals
        cudaMemcpy(planner.d_pathToGoal_ptr_, &zero, sizeof(int), cudaMemcpyHostToDevice);
        planner.h_pathToGoal_ = 0;

        cudaEventRecord(iterStart);
        planner.propagateFrontier(d_obstacles, numObstacles);
        planner.graph_.updateVertices();
        int oldTreeSize = planner.h_treeSize_;   // nodes before this iter's additions
        planner.updateFrontier();
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&iterMs, iterStart, iterStop);
        plannerMs += iterMs;

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
                result.first_solution_tree_size = planner.h_treeSize_;
            }
            if(pathCost < result.final_best_cost)
                result.final_best_cost = pathCost;
        }

        // --- Frontier-death diagnostics (outside the timed window) ---
        // reactivated: old tree nodes (idx < oldTreeSize) re-added to the frontier.
        // New nodes live at idx >= oldTreeSize, so they are excluded.
        int reactivated = (int)thrust::count(planner.d_frontier_.begin(),
                                             planner.d_frontier_.begin() + oldTreeSize, true);
        // R2 coverage: fraction of sub-regions ever activated (activeSubVertices != 0).
        int inactiveR2 = (int)thrust::count(planner.graph_.d_activeSubVertices_.begin(),
                                            planner.graph_.d_activeSubVertices_.end(), 0);
        float r2CoveragePct = 100.0f * float(NUM_R2_REGIONS - inactiveR2) / float(NUM_R2_REGIONS);
        // Mean Syclop vertex score across all R1 regions.
        float scoreSum  = thrust::reduce(planner.graph_.d_vertexScoreArray_.begin(),
                                         planner.graph_.d_vertexScoreArray_.end(), 0.0f);
        float meanScore = scoreSum / float(NUM_R1_REGIONS);

        IterationData d;
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.num_regions       = NUM_R1_REGIONS;
        d.r2_coverage_pct   = r2CoveragePct;
        d.mean_vertex_score = meanScore;
        d.reactivated       = reactivated;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;

        // Timeout check (planner-only time)
        if(plannerMs >= maxTimeMs) break;
    }

    result.total_time_seconds = plannerMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
    return result;
}

// ========================================================================
// Run KPAX baseline on one environment for 20 runs
// ========================================================================
void runKPAXBaseline(
    const std::string& deltaLabel,
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
    printf("KPAX BASELINE: %s | Delta: %s | %d runs\n", environment_name.c_str(), deltaLabel.c_str(), numRuns);
    printf("========================================\n");

    {
        KPAX planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkKPAX(planner, deltaLabel, environment_name, run,
                                             h_initial, h_goal, d_obstacles,
                                             numObstacles, maxIterations, maxTimeMs);
            printf("  Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
                   run + 1, numRuns, result.total_time_seconds, result.total_iterations,
                   result.final_tree_size, result.first_solution_iteration,
                   result.first_solution_cost, result.final_best_cost);
            writePerIterationCSV(result, outputDir);
            if(g_dumpViz && run == 0)
                dumpTreeCSV(planner.d_treeSamples_ptr_, planner.d_treeSamplesParentIdxs_ptr_,
                            planner.d_treeSampleCosts_ptr_, planner.h_treeSize_,
                            vizTreePath(g_vizDir, environment_name, "KPAX"));
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
            if(g_dumpViz && run == 0)
                dumpTreeCSV(planner.d_treeSamples_ptr_, planner.d_treeSamplesParentIdxs_ptr_,
                            planner.d_treeSampleCosts_ptr_, planner.h_treeSize_,
                            vizTreePath(g_vizDir, environment_name, "KinoPaxPlus"));
            all_results.push_back(result);

            if(run < numRuns - 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

// ========================================================================
// KinoPaxSTARcostprune benchmark + runner (cost-based pruning variant of
// KinoPaxSTAR; cost tracked via h_minCost_, same instrumentation).
// ========================================================================
RunResult benchmarkKinoPaxSTARcostprune(
    KinoPaxSTARcostprune& planner,
    const std::string& deltaLabel,
    const std::string& environment,
    int runNumber,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations,
    float maxTimeMs,
    float acceptCap,
    float costPruneExp,
    float costPruneFloor,
    int reactivationBlend,
    const std::string& label)
{
    // Override the planner's defaults for this run. resetPlanner (called below) does not
    // touch these, so setting them at entry holds for the run.
    planner.h_acceptCap_         = acceptCap;
    planner.h_costPruneExp_      = costPruneExp;
    planner.h_costPruneFloor_    = costPruneFloor;
    planner.h_reactivationBlend_ = reactivationBlend;
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
        int oldTreeSize = planner.h_treeSize_;   // nodes before this iter's additions
        planner.updateFrontier();
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&iterMs, iterStart, iterStop);
        plannerMs += iterMs;

        // h_minCost_ is the cumulative root-to-goal path length (KinoPaxPlus-style).
        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
        if(planner.h_minCost_ < MAX_FLOAT && result.first_solution_iteration == -1)
        {
            result.first_solution_iteration = itr;
            result.first_solution_cost      = planner.h_minCost_;
            result.first_solution_tree_size = planner.h_treeSize_;
        }
        if(planner.h_minCost_ < result.final_best_cost)
            result.final_best_cost = planner.h_minCost_;

        // --- Frontier diagnostics (outside the timed window; KinoPaxSTARcostprune uses the KPAX Graph) ---
        int reactivated = (int)thrust::count(planner.d_frontier_.begin(),
                                             planner.d_frontier_.begin() + oldTreeSize, true);
        int inactiveR2 = (int)thrust::count(planner.graph_.d_activeSubVertices_.begin(),
                                            planner.graph_.d_activeSubVertices_.end(), 0);
        float r2CoveragePct = 100.0f * float(NUM_R2_REGIONS - inactiveR2) / float(NUM_R2_REGIONS);
        float scoreSum  = thrust::reduce(planner.graph_.d_vertexScoreArray_.begin(),
                                         planner.graph_.d_vertexScoreArray_.end(), 0.0f);
        float meanScore = scoreSum / float(NUM_R1_REGIONS);

        IterationData d;
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.num_regions       = NUM_R1_REGIONS;
        d.r2_coverage_pct   = r2CoveragePct;
        d.mean_vertex_score = meanScore;
        d.reactivated       = reactivated;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        if(planner.h_propIterations_ == 0) break;

        // Timeout check (planner-only time)
        if(plannerMs >= maxTimeMs) break;
    }

    result.total_time_seconds = plannerMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
    return result;
}

void runKinoPaxSTARcostpruneUnionProbes(
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
    printf("KINOPAXSTARCOSTPRUNE UNION PROBES: %s | Delta: %s | Regions: %d\n",
           environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    // --- Union-blend probes: fmaxf(costProb, syclop) with floor 0 (see UNION_* above) ---
    for(int u = 0; u < NUM_UNION_POINTS; u++)
    {
        const float       cap   = UNION_CAPS[u];
        const std::string label = UNION_LABELS[u];

        printf("  --- UNION blend: acceptCap = %.2f, exp = %.2f, floor = %.2f (%s) ---\n",
               cap, UNION_EXP, UNION_FLOOR, label.c_str());
        KinoPaxSTARcostprune planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkKinoPaxSTARcostprune(planner, deltaLabel, environment_name, run,
                                                 h_initial, h_goal, d_obstacles,
                                                 numObstacles, maxIterations, maxTimeMs,
                                                 cap, UNION_EXP, UNION_FLOOR, 1, label);
            printf("  union cap=%.2f Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
                   cap, run + 1, numRuns, result.total_time_seconds, result.total_iterations,
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
// KinoPaxSTARNoPruneAncestor benchmark + runner (KinoPaxSTARNoPrune plus KinoPaxPlus's
// ancestor pruning; h_ancestorPrune_ selects off / node-only / memoized chain).
// ========================================================================
RunResult benchmarkKinoPaxSTARNoPruneAncestor(
    KinoPaxSTARNoPruneAncestor& planner,
    const std::string& deltaLabel,
    const std::string& environment,
    int runNumber,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations,
    float maxTimeMs,
    int ancestorPrune,
    const std::string& label)
{
    // Override the planner's default for this run. resetPlanner (called below) does not touch
    // h_ancestorPrune_, so setting it at entry holds for the run. h_ancestorTol_ and
    // h_dormancyThreshold_ stay at their ctor defaults (0.0 and 5), i.e. KinoPaxPlus's
    // strict test and its hardcoded dormancy window.
    planner.h_ancestorPrune_ = ancestorPrune;
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
        int oldTreeSize = planner.h_treeSize_;   // nodes before this iter's additions
        planner.updateFrontier();
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&iterMs, iterStart, iterStop);
        plannerMs += iterMs;

        // h_minCost_ is the cumulative root-to-goal path length (KinoPaxPlus-style).
        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
        if(planner.h_minCost_ < MAX_FLOAT && result.first_solution_iteration == -1)
        {
            result.first_solution_iteration = itr;
            result.first_solution_cost      = planner.h_minCost_;
            result.first_solution_tree_size = planner.h_treeSize_;
        }
        if(planner.h_minCost_ < result.final_best_cost)
            result.final_best_cost = planner.h_minCost_;

        // --- Frontier diagnostics (outside the timed window; KinoPaxSTARNoPruneAncestor uses the KPAX Graph) ---
        int reactivated = (int)thrust::count(planner.d_frontier_.begin(),
                                             planner.d_frontier_.begin() + oldTreeSize, true);
        int inactiveR2 = (int)thrust::count(planner.graph_.d_activeSubVertices_.begin(),
                                            planner.graph_.d_activeSubVertices_.end(), 0);
        float r2CoveragePct = 100.0f * float(NUM_R2_REGIONS - inactiveR2) / float(NUM_R2_REGIONS);
        float scoreSum  = thrust::reduce(planner.graph_.d_vertexScoreArray_.begin(),
                                         planner.graph_.d_vertexScoreArray_.end(), 0.0f);
        float meanScore = scoreSum / float(NUM_R1_REGIONS);

        IterationData d;
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.num_regions       = NUM_R1_REGIONS;
        d.r2_coverage_pct   = r2CoveragePct;
        d.mean_vertex_score = meanScore;
        d.reactivated       = reactivated;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        if(planner.h_propIterations_ == 0) break;

        // Timeout check (planner-only time)
        if(plannerMs >= maxTimeMs) break;
    }

    result.total_time_seconds = plannerMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
    return result;
}

void runKinoPaxSTARNoPruneAncestorBenchmark(
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
    printf("KINOPAXSTARNOPRUNEANCESTOR MODES: %s | Delta: %s | Regions: %d\n",
           environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    // All three ancestor-pruning modes; fresh planner instance per mode.
    for(int m = 0; m < NUM_ANCESTOR_MODES; m++)
    {
        const int         mode  = ANCESTOR_MODES[m];
        const std::string label = ANCESTOR_LABELS[m];

        printf("  --- ancestorPrune = %d (%s) ---\n", mode, label.c_str());
        KinoPaxSTARNoPruneAncestor planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkKinoPaxSTARNoPruneAncestor(planner, deltaLabel, environment_name, run,
                                                 h_initial, h_goal, d_obstacles,
                                                 numObstacles, maxIterations, maxTimeMs,
                                                 mode, label);
            printf("  mode=%d Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
                   mode, run + 1, numRuns, result.total_time_seconds, result.total_iterations,
                   result.final_tree_size, result.first_solution_iteration,
                   result.first_solution_cost, result.final_best_cost);
            writePerIterationCSV(result, outputDir);
            if(g_dumpViz && run == 0)
                dumpTreeCSV(planner.d_treeSamples_ptr_, planner.d_treeSamplesParentIdxs_ptr_,
                            planner.d_treeSampleCosts_ptr_, planner.h_treeSize_,
                            vizTreePath(g_vizDir, environment_name, label));
            all_results.push_back(result);

            if(run < numRuns - 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

// ========================================================================
// KinoPaxSTARWeightedCost benchmark + runner (weighted-sum acceptance:
// min(1, w*P_syclop + (1-w)*P_cost + P_floor), with a toggleable ancestor prune).
// ========================================================================
RunResult benchmarkKinoPaxSTARWeightedCost(
    KinoPaxSTARWeightedCost& planner,
    const std::string& deltaLabel,
    const std::string& environment,
    int runNumber,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations,
    float maxTimeMs,
    float costWeight,
    float costPruneExp,
    int ancestorPrune,
    const std::string& label)
{
    // Override the planner's defaults for this run. resetPlanner (called below) does not
    // touch these, so setting them at entry holds for the run. h_probFloor_, h_ancestorTol_ and
    // h_dormancyThreshold_ stay at their ctor defaults (EPSILON, 0.0, 5).
    planner.h_costWeight_     = costWeight;
    planner.h_costPruneExp_   = costPruneExp;
    planner.h_ancestorPrune_  = ancestorPrune;
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
        int oldTreeSize = planner.h_treeSize_;   // nodes before this iter's additions
        planner.updateFrontier();
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&iterMs, iterStart, iterStop);
        plannerMs += iterMs;

        // h_minCost_ is the cumulative root-to-goal path length (KinoPaxPlus-style).
        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
        if(planner.h_minCost_ < MAX_FLOAT && result.first_solution_iteration == -1)
        {
            result.first_solution_iteration = itr;
            result.first_solution_cost      = planner.h_minCost_;
            result.first_solution_tree_size = planner.h_treeSize_;
        }
        if(planner.h_minCost_ < result.final_best_cost)
            result.final_best_cost = planner.h_minCost_;

        // --- Frontier diagnostics (outside the timed window; KinoPaxSTARWeightedCost uses the KPAX Graph) ---
        int reactivated = (int)thrust::count(planner.d_frontier_.begin(),
                                             planner.d_frontier_.begin() + oldTreeSize, true);
        int inactiveR2 = (int)thrust::count(planner.graph_.d_activeSubVertices_.begin(),
                                            planner.graph_.d_activeSubVertices_.end(), 0);
        float r2CoveragePct = 100.0f * float(NUM_R2_REGIONS - inactiveR2) / float(NUM_R2_REGIONS);
        float scoreSum  = thrust::reduce(planner.graph_.d_vertexScoreArray_.begin(),
                                         planner.graph_.d_vertexScoreArray_.end(), 0.0f);
        float meanScore = scoreSum / float(NUM_R1_REGIONS);

        IterationData d;
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.num_regions       = NUM_R1_REGIONS;
        d.r2_coverage_pct   = r2CoveragePct;
        d.mean_vertex_score = meanScore;
        d.reactivated       = reactivated;
        result.per_iteration.push_back(d);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        if(planner.h_propIterations_ == 0) break;

        // Timeout check (planner-only time)
        if(plannerMs >= maxTimeMs) break;
    }

    result.total_time_seconds = plannerMs / 1000.0;
    result.final_tree_size    = planner.h_treeSize_;
    result.total_iterations   = itr;

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
    return result;
}

void runKinoPaxSTARWeightedCostBenchmark(
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
    printf("KINOPAXSTARWEIGHTEDCOST GRID: %s | Delta: %s | Regions: %d\n",
           environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    // Full factorial over WEIGHTS x WEIGHTED_EXPS x WEIGHTED_ANC; fresh planner per grid point.
    for(int wi = 0; wi < NUM_WEIGHTS; wi++)
    for(int ei = 0; ei < NUM_WEIGHTED_EXPS; ei++)
    for(int ai = 0; ai < NUM_WEIGHTED_ANC; ai++)
    {
        const float       w     = WEIGHTS[wi];
        const float       k     = WEIGHTED_EXPS[ei];
        const int         anc   = WEIGHTED_ANC[ai];
        const std::string label = weightedLabel(w, k, anc);

        printf("  --- w = %.2f, k = %.2f, ancestorPrune = %d (%s) ---\n",
               w, k, anc, label.c_str());
        KinoPaxSTARWeightedCost planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkKinoPaxSTARWeightedCost(planner, deltaLabel, environment_name, run,
                                                 h_initial, h_goal, d_obstacles,
                                                 numObstacles, maxIterations, maxTimeMs,
                                                 w, k, anc, label);
            printf("  w=%.2f k=%.2f anc=%d Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
                   w, k, anc, run + 1, numRuns, result.total_time_seconds, result.total_iterations,
                   result.final_tree_size, result.first_solution_iteration,
                   result.first_solution_cost, result.final_best_cost);
            writePerIterationCSV(result, outputDir);
            if(g_dumpViz && run == 0)
                dumpTreeCSV(planner.d_treeSamples_ptr_, planner.d_treeSamplesParentIdxs_ptr_,
                            planner.d_treeSampleCosts_ptr_, planner.h_treeSize_,
                            vizTreePath(g_vizDir, environment_name, label));
            all_results.push_back(result);

            if(run < numRuns - 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

int main(int argc, char* argv[])
{
    std::string deltaLabel    = (argc > 1) ? argv[1] : "unknown";
    std::string obstaclePath  = (argc > 2) ? argv[2] : "../include/config/obstacles/zigzag/obstacles.csv";
    std::string envName       = (argc > 3) ? argv[3] : "zigzag";

    // The KPAX baseline runs by default; pass --skip-baselines to omit it.
    // --dump-viz additionally dumps run-0's full tree per variant for the spatial /
    // tree-growth visualization (Data/Benchmarks/KinoPaxStarCostTuning/viz/).
    bool skipBaselines = false;
    for(int i = 4; i < argc; i++)
    {
        if(std::string(argv[i]) == "--skip-baselines")
            skipBaselines = true;
        else if(std::string(argv[i]) == "--dump-viz")
            g_dumpViz = true;
    }

    const int NUM_KPAX_RUNS        = 10;
    const int NUM_KINOPAXPLUS_RUNS = 10;   // drives the KinoPaxPlus runner
    const int NUM_KINOPAXSTARCOSTPRUNE_RUNS = 10;   // drives the KinoPaxSTARcostprune grid
    const int NUM_KINOPAXSTARNOPRUNEANCESTOR_RUNS  = 10;   // drives the KinoPaxSTARNoPruneAncestor modes
    const int NUM_KINOPAXSTARWEIGHTED_RUNS  = 10;   // drives the KinoPaxSTARWeightedCost grid
    const int MAX_ITERATIONS       = 500;
    const float MAX_TIME_MS      = 10000.0f;  // 10 second timeout

    std::string outputDir = "Data/Benchmarks/KinoPaxStarCostTuning";
    std::filesystem::create_directories(outputDir);

    printf("=======================================================\n");
    printf("    KINOPAXSTAR COST TUNING SWEEP\n");
    printf("=======================================================\n");
    printf("Delta label:    %s\n", deltaLabel.c_str());
    printf("NUM_R1_REGIONS: %d\n", NUM_R1_REGIONS);
    printf("MAX_TREE_SIZE:  %d\n", MAX_TREE_SIZE);
    printf("W_R1_LENGTH=%d  C_R1_LENGTH=%d  V_R1_LENGTH=%d\n", W_R1_LENGTH, C_R1_LENGTH, V_R1_LENGTH);
    printf("Obstacle file:  %s\n", obstaclePath.c_str());
    printf("Environment:    %s\n", envName.c_str());
    printf("Baselines:      %s (KPAX, %d runs)\n", skipBaselines ? "NO" : "YES", NUM_KPAX_RUNS);
    printf("Cost metric:    %s (COST_MODE=%d)\n", (COST_MODE == 1) ? "control effort" : "workspace path length", COST_MODE);
    printf("Dump viz:       %s\n", g_dumpViz ? "YES (run 0 per variant)" : "NO");
    printf("KinoPaxPlus:    %d runs\n", NUM_KINOPAXPLUS_RUNS);
    printf("WeightedCost:   w x k x anc = %d x %d x %d = %d points x %d runs = %d runs\n",
           NUM_WEIGHTS, NUM_WEIGHTED_EXPS, NUM_WEIGHTED_ANC,
           NUM_WEIGHTS * NUM_WEIGHTED_EXPS * NUM_WEIGHTED_ANC, NUM_KINOPAXSTARWEIGHTED_RUNS,
           NUM_WEIGHTS * NUM_WEIGHTED_EXPS * NUM_WEIGHTED_ANC * NUM_KINOPAXSTARWEIGHTED_RUNS);
    printf("KinoPaxSTARcostprune: %d union-blend reference probes x %d runs\n",
           NUM_UNION_POINTS, NUM_KINOPAXSTARCOSTPRUNE_RUNS);
    printf("KinoPaxSTARNoPruneAncestor:  %d modes (off/node/chain) x %d runs each = %d runs\n",
           NUM_ANCESTOR_MODES, NUM_KINOPAXSTARNOPRUNEANCESTOR_RUNS,
           NUM_ANCESTOR_MODES * NUM_KINOPAXSTARNOPRUNEANCESTOR_RUNS);
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

    // --- Visualization dump setup (opt-in) ---
    if(g_dumpViz)
    {
        g_vizDir = outputDir + "/viz";
        std::filesystem::create_directories(g_vizDir);
        writeVizMeta(g_vizDir + "/meta.csv", h_initial, h_goal);
        printf("[viz] --dump-viz ON: run-0 tree per variant + meta -> %s\n", g_vizDir.c_str());
    }

    std::vector<RunResult> all_results;

    // --- KPAX baseline (matched to this build's discretization) ---
    if(!skipBaselines)
    {
        runKPAXBaseline(deltaLabel, envName, h_initial, h_goal, d_obstacles, numObstacles,
                        all_results, outputDir, NUM_KPAX_RUNS, MAX_ITERATIONS, MAX_TIME_MS);
    }

    // --- KinoPaxPlus delta benchmark ---
    runKinoPaxPlusBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                            all_results, outputDir, deltaLabel, NUM_KINOPAXPLUS_RUNS, MAX_ITERATIONS, MAX_TIME_MS);

    // --- KinoPaxSTARcostprune union-blend reference probes ---
    runKinoPaxSTARcostpruneUnionProbes(envName, h_initial, h_goal, d_obstacles, numObstacles,
                            all_results, outputDir, deltaLabel, NUM_KINOPAXSTARCOSTPRUNE_RUNS, MAX_ITERATIONS, MAX_TIME_MS);

    // --- KinoPaxSTARWeightedCost: w x k x ancestor grid ---
    runKinoPaxSTARWeightedCostBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                            all_results, outputDir, deltaLabel, NUM_KINOPAXSTARWEIGHTED_RUNS, MAX_ITERATIONS, MAX_TIME_MS);

    // --- KinoPaxSTARNoPruneAncestor: all three ancestor-pruning modes ---
    runKinoPaxSTARNoPruneAncestorBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                            all_results, outputDir, deltaLabel, NUM_KINOPAXSTARNOPRUNEANCESTOR_RUNS, MAX_ITERATIONS, MAX_TIME_MS);

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
