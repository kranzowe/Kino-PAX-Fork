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
#include "planners/KinoPaxSTARTrue.cuh"
#include "planners/CountingStars.cuh"
#include <thrust/count.h>
#include <thrust/reduce.h>

// --- Visualization dump (opt-in via --dump-viz); set in main(), read by the runners. ---
static bool        g_dumpViz = false;
static std::string g_vizDir;

// ---- PAPER BENCHMARK V2 ----
//
// This file is countingstars_sweep.cu's proven-reliable harness (same four planners, same
// large/fine/tiny discretizations, same MAX_ITERATIONS/MAX_TIME_MS), minimally adapted to
// reproduce paper_benchmark.cu's fixed 4-planner headline comparison across all four
// environments -- paper_benchmark.cu (MODEL 2 / Dubins Airplane) hangs at the tiny discretization
// and the root cause is still open; this tool sidesteps that by running the same comparison on the
// harness that has actually completed every delta without hanging. MODEL has since moved twice:
// first to 2 (Dubins Airplane, matching paper_benchmark.cu's original target -- with a corrected
// C_DIM=2/V_DIM=1 dimension breakdown instead of paper_benchmark.cu's own C_DIM=0/V_DIM=3, which
// crams yaw/pitch into the velocity-shaped V_DIM=3 slot bounded to [-0.3,0.3], a real
// region-density-skew bug flagged earlier), and now to 3 (12D Non-Linear Quad). See
// run_paper_benchmark_v2.sh for the full derivation of both switches, including why Quad keeps its
// own native [0,100] workspace scale rather than being shrunk to match the others, and how
// obstacles/start/goal get scaled into it (this file's main(), right after readObstaclesFromCSV).
//
// CountingStars             bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15, cost_frac 0.75,
//                           h_hopelessGuard_ PERMANENTLY ON (v3.5) -- countingstars_sweep.cu's
//                           own on/off sweep confirmed the guard helps, and this is the re-tuned
//                           bufferSlope/bufferFloor/explore_frac/cost_frac point that followed.
//                           Replaces the earlier two-point (bf0.3/bf0.6), unguarded arm. Runs on
//                           the permanently-budgeted CountingStars (OPTIMAL nodes are limited by
//                           budget in BOTH acceptance and reactivation, neither uncapped) -- see
//                           CS_DOORBIT_GUAR / CS_DOORBIT_OPTIMAL in CountingStars.cuh.
static const float CS_BUFFER_SLOPE = 1.2f;
static const float CS_BUFFER_FLOOR = 0.4f;
static const float CS_EXPLORE_FRAC = 0.15f;
static const float CS_COST_FRAC    = 0.75f;

// v3.5: THE HOPELESS GUARD -- PERMANENTLY ON this pass, not a swept axis (h_hopelessGuard_ default
// is still `false`; this file just never runs a candidate/dormant node with it off any more). Kept
// as a named value rather than a hardcoded literal so it still flows through
// countingStarsLabel()/benchmarkCountingStars() and every label names it, the same r2-off-style
// precedent trueLabel()'s own comment describes (label every axis a fixed point was built with,
// not only the swept ones).
static const int CS_HOPELESS_GUARD = 1;

// How many iterations a run actually completes inside the 10s wall-clock cap at
// MAX_TREE_SIZE = 3,000,000 (empirical). The ramp's x = itr/fill_iters must track the REAL run
// length, not MAX_ITER: h_fillIters_ defaults to MAX_ITER (1000 in write_config()'s heredoc), but a
// run now times out around 700 iterations, well short of that -- x would never reach 1 and B would
// never reach its ramp maximum for a run's entire duration. See benchmarkCountingStars() below,
// where this is assigned to planner.h_fillIters_ before resetPlanner().
// This tool's own copy of the constant (paper_benchmark.cu and countingstars_sweep.cu each keep
// their own copy too) -- a mismatched fill_iters would make the same (slope, floor) mean a
// different ramp in a different binary.
static const int CS_RAMP_FILL_ITERS = 700;

// "CountingStars_bs120_bf30_ef100_cf800_hg0". MUST start with a name loadRuns() dispatches on.
//
//   bs   bufferSlope, round(100 x float)
//   bf   bufferFloor, round(100 x float)   -- B(x) is DERIVED from these, and goal_frontier_size is
//                                             a per-ITERATION CSV column, not a per-run constant
//   ef   explore_frac, round(1000 x float)
//   cf   cost_frac,    round(1000 x float)
//   hg   hopelessGuard, 0 or 1 (v3.5) -- CS_HOPELESS_GUARD is 1 for every point this pass, see above
//
// Kept as a function (rather than a hardcoded literal) for the same reason paper_benchmark.cu
// keeps its own copy: self-documenting, and it stays correct as bufferSlope/bufferFloor sweep.
static std::string countingStarsLabel(float bufferSlope, float bufferFloor, float exploreFrac, float costFrac,
                                      int hopelessGuard)
{
    char buf[176];
    snprintf(buf, sizeof(buf), "CountingStars_bs%d_bf%d_ef%d_cf%d_hg%d",
             (int)lroundf(100.0f * bufferSlope),
             (int)lroundf(100.0f * bufferFloor),
             (int)lroundf(1000.0f * exploreFrac),
             (int)lroundf(1000.0f * costFrac),
             hopelessGuard);
    return std::string(buf);
}

// "KinoPaxSTARTrue_cap100_anc0" / "..._anc1". Identical to paper_benchmark.cu's own trueLabel() --
// see there for why cap/anc are labeled at all even though cap never varies (r2-off-style
// precedent: label every axis a fixed point was constructed with, not just the swept ones).
static std::string trueLabel(float syclopCap, int ancestorPrune)
{
    char buf[96];
    snprintf(buf, sizeof(buf), "KinoPaxSTARTrue_cap%d_anc%d",
             (int)lroundf(100.0f * syclopCap), ancestorPrune);
    return std::string(buf);
}

struct IterationData
{
    int iteration;
    int frontier_size;
    int tree_size;
    float elapsed_time_ms;
    float best_cost;
    int reactivated;          // dormant tree nodes re-added to frontier this iter (KPAX-family planners; -1 otherwise)
    // --- normalization diagnostics ---
    float score_floor;        // Graph::h_scoreFloor_: EPSILON for legacy planners, 1/N_active for
                              // the opted-in ones. The direct evidence the floor fix is live, and
                              // the one column where KPAX and KPAXCap visibly differ. NaN if the
                              // planner has no Graph (KinoPaxPlus).
    float cost_scale;         // CleanCost's D_global = globalMeanCost - globalMinCost, the
                              // denominator in costProbExpGlobal. Compare against the per-region
                              // spreads that used to be the denominator to pick the next k range.
                              // NaN for every other planner.

    // --- CountingStars diagnostics (NaN / -1 for every other planner) ---
    //
    // Read them in this order. The first answers the claim the whole design rests on; the rest are
    // how you find out which door broke it.
    //
    //   1. IS THE BUDGET MET. budget_used against the series' goal_frontier_size, every iteration.
    //      A persistent SHORTFALL means a door is not filling its share. Nothing on the candidate
    //      side is uncapped any more (v3.4 folded OPTIMAL into CHEAPEST's own budget permanently),
    //      so a genuine OVERSHOOT here is unexpected and worth investigating directly, not a known
    //      regime the low-B points are on the grid to exercise. frontier_size says the same thing
    //      one iteration later (it is measured at the top of the next propagateFrontier).
    //   2. IS THE FRONTIER DOING MORE WORK. prop_per_node = prop_attempted / frontier_size, against
    //      KinoPaxPlus's bf, which reaches 40,000 at F = 10. THE POINT OF CONTROLLING F IS
    //      CONTROLLING THIS. If it does not move with B, B is not the lever.
    //   3. WHICH DOOR BUILT THE TREE. admitted_cost / admitted_explore / admitted_costdist /
    //      reactivated_best / reactivated_count, plus ord_cutoff and cost_cutoff_dist. A cutoff
    //      RISING over a run is expected -- regions fill, so both signals get scarce. Pinned at 0
    //      means no candidate is ever good enough on that signal and the fraction is doing nothing.
    //
    //      admitted_explore, admitted_cost AND admitted_costdist OVERLAP (v3.3: OPTIMAL now also
    //      competes for FRESHEST). admitted_opt_fresh_both and admitted_both are what make them add
    //      back up:
    //      admitted == admitted_cost + admitted_explore + admitted_costdist + admitted_floor
    //                - admitted_opt_fresh_both - admitted_both.
    //      READ admitted_cost HERE, NOT optimal_count: admitted_cost is what pass 2 actually let
    //      through the CHEAPEST cutoff (v3.4, permanent); optimal_count is pass 1's measured
    //      population at distance 0. (optimal_count - admitted_cost) is the optimal-door
    //      starvation count -- how many optimal candidates the budget did not have room for.
    //   4. BLOCK IDENTITY. frontier_repeat_size must equal the sum of the frontier's admission-time
    //      block counts after scaling, and prop_attempted / frontier_repeat_size must be EXACTLY 32
    //      on every iteration. Kernel1 is retained by construction, so below 32 is a defect. v3.3:
    //      an admission-time block count is now popcount(door), 1 or 2 -- not a swept boost size.
    int   prop_attempted;         // propagations launched this iteration, collisions included
    int   frontier_repeat_size;   // sum of the per-node block counts; x32 is the kernel1 attempt count
    // The budget's own arithmetic, as applied THIS iteration. optimal_count is pass 1's MEASURED
    // population at distance 0 -- how many candidates WANTED first claim, not how many got it
    // (that is admitted_cost, below); ord_cutoff and cost_cutoff are the two thresholds the fixed
    // shares bought.
    //
    // guaranteed_react IS GONE. It was the guarantee's PLANNED size, and it existed only because
    // v2's draw probability was a remainder that had to know it. v3's draw is a fixed share, so
    // nothing consumes the plan and reactivated_best is the guarantee's realised size, counted
    // exactly on the device.
    int   optimal_count;
    // v3.4: what pass 2 actually ADMITTED via CS_DOORBIT_OPTIMAL, through the SAME cutoff CHEAPEST
    // uses (permanent, not a toggle). Strictly less than optimal_count whenever the budget starves
    // some of them -- (optimal_count - admitted_cost) is that starvation count.
    int   admitted_cost;
    int   ord_cutoff;
    int   budget_used;
    // ---- v3 / v3.2 ----
    // B, DERIVED inside the planner every iteration from bufferSlope/bufferFloor. A column rather
    // than a per-series constant the plot script carries alongside the label, which is what v2 had
    // to do: the budget figure divides budget_used by it, and re-deriving the ramp in MATLAB would
    // put a second copy of that arithmetic somewhere it could drift. UNLIKE v3, this now genuinely
    // VARIES row to row within one run -- it was always a per-iteration column, just constant under
    // v3's fixed B, so this is the first pass where plotting it against iteration is worth a panel.
    int   goal_frontier_size;
    // The cost door's cutoff, as a DISTANCE rather than a bucket index -- the index is only
    // meaningful against the dist_max that produced it, and dist_max moves every iteration.
    //
    // cost_cutoff_dist collapsing toward dist_max / 2^21 means every candidate is landing in bucket
    // 0 and the boundary roll has degraded the door to a uniform draw among near-optimal candidates
    // -- the signal that the log bucket map has the wrong shape for this distribution.
    float cost_cutoff_dist;
    float dist_max;
    // ---- v3.1: Part B's cost arm ----
    // The reactivation cutoff, over DORMANT TREE NODES rather than candidates. Read
    // react_cutoff_dist against dist_max exactly as for the candidate door -- and note dist_max is
    // the CANDIDATE anchor, reused: a dormant node above it clamps into the top bucket, which is
    // harmless (this arm takes the SMALLEST distances) unless the cutoff itself pins there, which
    // means the budget exceeded the population below dist_max.
    float react_cutoff_dist;
    // Admissions by door, counted exactly on the device. admitted_costdist is v3's cost-distance
    // door; admitted_both is the FRESHEST/CHEAPEST overlap.
    int   admitted_explore;
    int   admitted_costdist;
    int   admitted_both;
    // v3.3: OPTIMAL's overlap with FRESHEST (OPTIMAL never overlaps CHEAPEST -- see
    // CS_DOORBIT_OPTIMAL in the header), and the admission floor's yield.
    int   admitted_opt_fresh_both;
    int   admitted_floor;
    // v3.1: PART B NOW HAS THREE ARMS, and the identity the CSV carries is
    //
    //     reactivated == reactivated_best + reactivated_cost + reactivated_count
    //
    // with the left side an independent host thrust::count over frontier bits in the pre-existing
    // tree and the right side device atomics -- so it checks all three arms for free.
    //
    //   reactivated_cost   the CHEAPEST arm, spending the whole react_frac * B budget
    //   reactivated_count  the COMPLETENESS FLOOR alone (~ react_floor * dormant_count, ~30 nodes).
    //                      It was the uniform draw through v3; if it is large here, the floor is
    //                      doing reach work it was not sized for.
    int   reactivated_cost;
    int   reactivated_count;
    // reactivated_best / frontier_size approaching 1 means the region-best guarantee IS the
    // frontier -- KinoPaxPlus's regime. Read it together with best_cost before deciding that is bad.
    int   reactivated_best;
    // Fan-out budget. block_scale < 1 means the BUFFER, not the fan-out rule, is setting how hard
    // nodes expand; block_scale near 0 means the rep >= 1 floor ate the budget and the fan-out split
    // is inert, which is a goal_frontier_size problem and no other knob will move it.
    float block_ceiling;
    float block_scale;
    // v3.5: how many candidates / dormant nodes were measured HOPELESS this iteration (own cost
    // already >= the best full-solution cost, so no descendant could ever beat it) -- see
    // h_hopelessGuard_ in CountingStars.cuh. Both are 0 for every run with hg0 (guard off).
    int   hopeless_count;
    int   hopeless_dormant_count;
};

// Blank the CountingStars-only columns. Every other planner calls this, exactly as KinoPaxPlus
// already writes NaN for score_floor / cost_scale: the plot script reads columns by name and
// tolerates NaN, so one schema serves every planner.
static void clearCountingStarsCols(IterationData& d)
{
    d.prop_attempted = -1;
    d.frontier_repeat_size = -1;
    d.optimal_count = -1;
    d.admitted_cost = -1;
    d.ord_cutoff = -1;
    d.budget_used = -1;
    d.goal_frontier_size = -1;
    d.cost_cutoff_dist = NAN;
    d.dist_max = NAN;
    d.react_cutoff_dist = NAN;
    d.admitted_explore = -1;
    d.admitted_costdist = -1;
    d.admitted_both = -1;
    d.admitted_opt_fresh_both = -1;
    d.admitted_floor = -1;
    d.reactivated_cost = -1;
    d.reactivated_count = -1;
    d.reactivated_best = -1;
    d.block_ceiling = NAN;
    d.block_scale = NAN;
    d.hopeless_count = -1;
    d.hopeless_dormant_count = -1;
}

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
    //   KPAX baseline:  {env}_KPAX_delta{build}_run{n}.csv
    //   CountingStars:  {env}_{planner label}_delta{build}_run{n}.csv, e.g. CountingStars_bs120_bf30_ef100_cf800_hg0
    //   KinoPaxSTARTrue: same form, e.g. KinoPaxSTARTrue_cap100_anc0
    //   KinoPaxPlus:    {env}_delta{label}_run{n}.csv
    // KinoPaxPlus deliberately keys on the DELTA rather than a planner name: that is what keeps
    // every discretization (large_*, fine_*, tiny_*, ...) in separate files.
    // The build label carries the cost metric (large_effort / large_length), which is a
    // compile-time property of the binary -- see COST_MODE in helper.cuh.
    if(result.delta_label == "KPAX")
        filename << outputDir << "/" << result.environment << "_KPAX_delta" << result.build_delta
                 << "_run" << result.run_number << ".csv";
    // COUNTINGSTARS AND KINOPAXSTARTRUE MUST BE IN THIS ARM. Falling through to the KinoPaxPlus
    // branch below is not a cosmetic naming problem: that branch keys on the DELTA and omits
    // build_delta entirely, so the length and effort builds write the SAME path and the second
    // silently overwrites the first.
    else if(result.delta_label.rfind("CountingStars", 0) == 0 || result.delta_label.rfind("KinoPaxSTAR", 0) == 0)
        filename << outputDir << "/" << result.environment << "_" << result.delta_label << "_delta" << result.build_delta
                 << "_run" << result.run_number << ".csv";
    else
        filename << outputDir << "/" << result.environment << "_delta" << result.delta_label
                 << "_run" << result.run_number << ".csv";

    std::ofstream file(filename.str());
    // score_floor / cost_scale / admitted_cost are appended, not inserted -- the plot script reads
    // columns by name via getCol(), which returns [] for a missing one, so older CSVs still load.
    file << "iteration,frontier_size,tree_size,elapsed_time_ms,best_cost,"
         << "reactivated,"
         << "score_floor,cost_scale,"
         << "prop_attempted,frontier_repeat_size,"
         << "optimal_count,ord_cutoff,budget_used,"
         << "goal_frontier_size,"
         << "cost_cutoff_dist,dist_max,"
         << "react_cutoff_dist,"
         << "admitted_explore,admitted_costdist,admitted_both,"
         << "admitted_opt_fresh_both,admitted_floor,"
         << "reactivated_cost,reactivated_count,reactivated_best,"
         << "block_ceiling,block_scale,"
         << "admitted_cost,"
         << "hopeless_count,hopeless_dormant_count\n";

    for(const auto& d : result.per_iteration)
    {
        file << d.iteration << ","
             << d.frontier_size << ","
             << d.tree_size << ","
             << std::fixed << std::setprecision(3) << d.elapsed_time_ms << ","
             << std::fixed << std::setprecision(6) << d.best_cost << ","
             << d.reactivated << ","
             << std::fixed << std::setprecision(9) << d.score_floor << ","
             << std::fixed << std::setprecision(6) << d.cost_scale << ","
             << d.prop_attempted << ","
             << d.frontier_repeat_size << ","
             << d.optimal_count << ","
             << d.ord_cutoff << ","
             << d.budget_used << ","
             << d.goal_frontier_size << ","
             << std::fixed << std::setprecision(9) << d.cost_cutoff_dist << ","
             << std::fixed << std::setprecision(9) << d.dist_max << ","
             << std::fixed << std::setprecision(9) << d.react_cutoff_dist << ","
             << std::fixed
             << d.admitted_explore << ","
             << d.admitted_costdist << ","
             << d.admitted_both << ","
             << d.admitted_opt_fresh_both << ","
             << d.admitted_floor << ","
             << d.reactivated_cost << ","
             << d.reactivated_count << ","
             << d.reactivated_best << ","
             << std::fixed << std::setprecision(1) << d.block_ceiling << ","
             << std::fixed << std::setprecision(4) << d.block_scale << ","
             << d.admitted_cost << ","
             << d.hopeless_count << ","
             << d.hopeless_dormant_count << "\n";
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
    filename << outputDir << "/paper_benchmark_v2_" << timestamp.str() << "_summary.csv";

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
        clearCountingStarsCols(d);
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.reactivated       = -1;
        d.score_floor       = NAN;   // KinoPaxPlus uses KinoPaxPlusRegions, not Graph
        d.cost_scale        = NAN;
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
    printf("    [KPAX diag] resetPlanner() returned\n"); fflush(stdout);

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
        if(itr <= 3) { printf("    [KPAX diag] itr=%d: propagateFrontier...\n", itr); fflush(stdout); }
        planner.propagateFrontier(d_obstacles, numObstacles);
        if(itr <= 3) { printf("    [KPAX diag] itr=%d: updateVertices...\n", itr); fflush(stdout); }
        planner.graph_.updateVertices();
        int oldTreeSize = planner.h_treeSize_;   // nodes before this iter's additions
        if(itr <= 3) { printf("    [KPAX diag] itr=%d: updateFrontier...\n", itr); fflush(stdout); }
        planner.updateFrontier();
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        cudaEventElapsedTime(&iterMs, iterStart, iterStop);
        plannerMs += iterMs;

        if(itr <= 5 || itr % 50 == 0)
            {
                printf("    [KPAX diag] itr=%d done, tree=%d, plannerMs=%.1f\n", itr, planner.h_treeSize_, plannerMs);
                fflush(stdout);
            }

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

        IterationData d;
        clearCountingStarsCols(d);
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.reactivated       = reactivated;
        d.score_floor       = planner.graph_.h_scoreFloor_;
        d.cost_scale        = NAN;
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
// KinoPaxSTARTrue -- two fixed naive points, both KPAX's exploration accept OR-fused with
// KinoPaxPlus's region-best accept, no cost shaping (h_syclopCap_ pinned at 1.0, a genuine no-op
// per the class's own comment). Only h_ancestorPrune_ varies: 0 is the pure fusion (== stock
// KinoPaxSTARNoGoalBias exactly, per KinoPaxSTARTrue's own constructor comment), 1 adds the
// cost-guarded stale-best prune on top. Identical points to paper_benchmark.cu's own
// KinoPaxSTARTrue arm -- added here to help isolate whether paper_benchmark.cu's illegal-memory-
// access bug lives in this planner, at a discretization already confirmed safe for
// KPAX/KinoPaxPlus/CountingStars.
// ========================================================================
RunResult benchmarkKinoPaxSTARTrue(
    KinoPaxSTARTrue& planner,
    const std::string& deltaLabel,
    const std::string& environment,
    int runNumber,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations,
    float maxTimeMs,
    float syclopCap,
    int ancestorPrune,
    const std::string& label)
{
    // resetPlanner does not touch h_syclopCap_/h_ancestorPrune_, so setting them at entry holds
    // for the whole run.
    planner.h_syclopCap_     = syclopCap;
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

        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
        if(planner.h_minCost_ < MAX_FLOAT && result.first_solution_iteration == -1)
        {
            result.first_solution_iteration = itr;
            result.first_solution_cost      = planner.h_minCost_;
            result.first_solution_tree_size = planner.h_treeSize_;
        }
        if(planner.h_minCost_ < result.final_best_cost)
            result.final_best_cost = planner.h_minCost_;

        // --- Frontier diagnostics (outside the timed window; KinoPaxSTARTrue uses the KPAX Graph) ---
        int reactivated = (int)thrust::count(planner.d_frontier_.begin(),
                                             planner.d_frontier_.begin() + oldTreeSize, true);

        IterationData d;
        clearCountingStarsCols(d);
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.reactivated       = reactivated;
        d.score_floor       = planner.graph_.h_scoreFloor_;
        d.cost_scale        = NAN;
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

void runKinoPaxSTARTrueBenchmark(
    const std::string& environment_name, float* h_initial, float* h_goal, float* d_obstacles, uint numObstacles,
    std::vector<RunResult>& all_results, const std::string& outputDir, const std::string& deltaLabel,
    int numRuns, int maxIterations, float maxTimeMs)
{
    // Single fixed point: cap held at its no-op default, ancestorPrune=1 (the guarded stale-best
    // prune on top of the naive fusion). Matches paper_benchmark.cu's own KinoPaxSTARTrue series
    // exactly -- see process_paper_benchmark_and_plot.m's trueCap/trueAncestorPrune constants.
    static const int   ANCESTOR_PRUNE = 1;
    static const float SYCLOP_CAP     = 1.0f;

    printf("\n========================================\n");
    printf("KINOPAXSTARTRUE: %s | Delta: %s | Regions: %d\n", environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    const std::string label = trueLabel(SYCLOP_CAP, ANCESTOR_PRUNE);

    printf("  --- syclopCap = %.2f, ancestorPrune = %d (%s) ---\n", SYCLOP_CAP, ANCESTOR_PRUNE, label.c_str());
    KinoPaxSTARTrue planner;
    for(int run = 0; run < numRuns; run++)
    {
        RunResult result = benchmarkKinoPaxSTARTrue(planner, deltaLabel, environment_name, run,
                                             h_initial, h_goal, d_obstacles, numObstacles, maxIterations, maxTimeMs,
                                             SYCLOP_CAP, ANCESTOR_PRUNE, label);
        printf("  anc=%d Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
               ANCESTOR_PRUNE, run + 1, numRuns, result.total_time_seconds,
               result.total_iterations, result.final_tree_size, result.first_solution_iteration,
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

// ========================================================================
// CountingStars v3 benchmark + runner.
// A DERIVED NODE BUDGET split by three fixed shares: explore_frac to the freshest regions,
// cost_frac to the smallest cost distances (which OPTIMAL candidates now always compete inside
// too, permanently -- v3.4), and the rest to a uniform draw. Nothing on the candidate side is
// uncapped any more; the region-best reactivation guarantee that used to be uncapped on the
// dormant-node side is also gone permanently (see CS_DOORBIT_GUAR).
// ========================================================================
RunResult benchmarkCountingStars(
    CountingStars& planner,
    const std::string& deltaLabel,
    const std::string& environment,
    int runNumber,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations,
    float maxTimeMs,
    float bufferSlope,
    float bufferFloor,
    float exploreFrac,
    float costFrac,
    int hopelessGuard,
    const std::string& label)
{
    // Override the planner's defaults for this run. resetPlanner (called below) does not touch the
    // tunables, so setting them at entry holds for the whole run.
    //
    // B IS NOT SET HERE AT ALL, not even indirectly: v3.2 recomputes it every iteration inside
    // updateFrontier() from bufferSlope/bufferFloor/h_itr_/h_fillIters_, so there is no one-time
    // derivation for these assignments to precede any more. They still have to land BEFORE
    // resetPlanner() though, exactly as h_reactFloor_/h_acceptFloor_ already rely on --
    // resetPlanner() reads none of the ramp fields itself, but updateFrontier() reads them on the
    // very first iteration of the run that follows.
    //
    // h_fillIters_ IS SET TO CS_RAMP_FILL_ITERS, NOT LEFT AT ITS MAX_ITER DEFAULT. It used to be
    // left alone deliberately -- setting it to maxIterations (this benchmark's own, much larger
    // cap) would have made the same (slope, floor) mean a different ramp than a plan() call sees.
    // But MAX_ITER itself is now a poor stand-in for "the run is over": at MAX_TREE_SIZE=3,000,000
    // and a 10s wall-clock cap, a real run only completes ~700 iterations, well short of MAX_ITER
    // (1000) -- x = itr/h_fillIters_ would never reach 1 and B would never reach its ramp maximum
    // for a run's entire duration. CS_RAMP_FILL_ITERS is that real run length instead.
    planner.h_fillIters_   = CS_RAMP_FILL_ITERS;
    planner.h_bufferSlope_ = bufferSlope;
    planner.h_bufferFloor_ = bufferFloor;
    planner.h_exploreFrac_ = exploreFrac;
    planner.h_costFrac_    = costFrac;
    // v3.5: the hopeless guard -- see h_hopelessGuard_ in CountingStars.cuh. Set alongside the
    // other tunables, before resetPlanner(), for the same reason: resetPlanner() does not touch it.
    planner.h_hopelessGuard_ = (hopelessGuard != 0);

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
        // NO graph_.updateVertices() HERE. CountingStars consumes nothing it produces -- no
        // vertexScores, no scoreFloor, no nActive, no regionCoverage -- and it is not cheap: a
        // kernel over NUM_R1_REGIONS doing NUM_R2_PER_R1 reads each, plus a reduce and a count_if,
        // every iteration. The other planners still call it because they genuinely use it.
        int oldTreeSize = planner.h_treeSize_;   // nodes before this iter's additions
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

        // --- Frontier diagnostics (outside the timed window) ---
        // reactivated counts frontier bits among the PRE-EXISTING tree, i.e. exactly Part B's
        // output. Part B has TWO live arms (cheapest, then the completeness floor; the region-best
        // guarantee that made it three is gone permanently), so the identity is
        //
        //     reactivated  ==  reactivated_best + reactivated_cost + reactivated_count
        //
        // with reactivated_best a free runtime invariant that must read exactly 0 every iteration
        // (its arm cannot fire any more). NOT reactivated_count alone, which is the completeness
        // floor. The two sides are computed independently (a thrust::count here, atomicAdds in the
        // kernel), so the sum is a free check on both live arms.
        int reactivated = (int)thrust::count(planner.d_frontier_.begin(),
                                             planner.d_frontier_.begin() + oldTreeSize, true);

        IterationData d;
        clearCountingStarsCols(d);
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.reactivated       = reactivated;
        d.score_floor       = NAN;   // no Syclop score, so no floor
        // NOT NaN any more, and that is a real column here rather than a courtesy: costScale is the
        // DENOMINATOR of a candidate's distance, and distance 0 is the top door. A scale collapsing
        // toward 0 would make the top door's test degenerate, and this is the only place that shows.
        d.cost_scale        = planner.h_costScale_;
        // CountingStars readout.
        d.prop_attempted       = (int)planner.h_propAttempted_;
        d.frontier_repeat_size = (int)planner.h_frontierRepeatSize_;
        d.optimal_count        = (int)planner.h_optimalCount_;
        d.admitted_cost        = (int)planner.h_admittedCost_;
        d.ord_cutoff           = planner.h_ordCutoff_;
        d.budget_used          = (int)planner.h_budgetUsed_;
        // B is DERIVED by the planner EVERY ITERATION now, so it is read back out of it rather than
        // echoed from the sweep's own axis -- which is what makes the column a check on the
        // derivation (does the realized ramp match slope*x+floor), not a copy of a setting.
        d.goal_frontier_size   = planner.h_goalFrontierSize_;
        d.cost_cutoff_dist     = planner.h_costCutoffDist_;
        d.dist_max             = planner.h_distMax_;
        d.react_cutoff_dist    = planner.h_reactCutoffDist_;
        d.admitted_explore     = (int)planner.h_admittedExplore_;
        d.admitted_costdist    = (int)planner.h_admittedCostDist_;
        d.admitted_both        = (int)planner.h_admittedBoth_;
        d.admitted_opt_fresh_both = (int)planner.h_admittedOptFreshBoth_;
        d.admitted_floor       = (int)planner.h_admittedFloor_;
        d.reactivated_cost     = (int)planner.h_reactivatedCost_;
        d.reactivated_count    = (int)planner.h_reactivated_;
        d.reactivated_best     = (int)planner.h_reactivatedBest_;
        d.block_ceiling        = planner.h_blockCeiling_;
        d.block_scale          = planner.h_blockScale_;
        d.hopeless_count         = (int)planner.h_hopelessCount_;
        d.hopeless_dormant_count = (int)planner.h_hopelessDormantCount_;
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
    printf("COUNTINGSTARS: %s | Delta: %s | Regions: %d\n",
           environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    // Single fixed point (CS_BUFFER_SLOPE/CS_BUFFER_FLOOR/CS_EXPLORE_FRAC/CS_COST_FRAC/
    // CS_HOPELESS_GUARD above) -- matches paper_benchmark.cu's own CountingStars series exactly.
    // B's RANGE over the run, not a single value: B(x=0) = floor, B(x=1) = slope + floor. Uses
    // CS_RAMP_FILL_ITERS, matching what planner.h_fillIters_ is actually set to below -- not
    // MAX_ITER, which is not the ramp's real denominator (see CS_RAMP_FILL_ITERS above).
    int bStart = (int)floorf(CS_BUFFER_FLOOR * float(MAX_TREE_SIZE) / float(CS_RAMP_FILL_ITERS));
    int bEnd   = (int)floorf((CS_BUFFER_SLOPE + CS_BUFFER_FLOOR) * float(MAX_TREE_SIZE) / float(CS_RAMP_FILL_ITERS));

    const std::string label = countingStarsLabel(CS_BUFFER_SLOPE, CS_BUFFER_FLOOR, CS_EXPLORE_FRAC, CS_COST_FRAC,
                                                  CS_HOPELESS_GUARD);
    printf("  --- bufferSlope = %.2f, bufferFloor = %.2f (B: %d -> %d), explore_frac = %.3f, "
           "cost_frac = %.3f, react_frac = %.3f, hopelessGuard = %d (%s) ---\n",
           CS_BUFFER_SLOPE, CS_BUFFER_FLOOR, bStart, bEnd,
           CS_EXPLORE_FRAC, CS_COST_FRAC, 1.0f - CS_EXPLORE_FRAC - CS_COST_FRAC,
           CS_HOPELESS_GUARD, label.c_str());
    CountingStars planner;
    for(int run = 0; run < numRuns; run++)
    {
        RunResult result = benchmarkCountingStars(planner, deltaLabel, environment_name, run,
                                             h_initial, h_goal, d_obstacles,
                                             numObstacles, maxIterations, maxTimeMs,
                                             CS_BUFFER_SLOPE, CS_BUFFER_FLOOR, CS_EXPLORE_FRAC, CS_COST_FRAC,
                                             CS_HOPELESS_GUARD, label);
        printf("  bs=%.1f bf=%.1f Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
               CS_BUFFER_SLOPE, CS_BUFFER_FLOOR, run + 1, numRuns, result.total_time_seconds,
               result.total_iterations, result.final_tree_size, result.first_solution_iteration,
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

int main(int argc, char* argv[])
{
    std::string deltaLabel    = (argc > 1) ? argv[1] : "unknown";
    std::string obstaclePath  = (argc > 2) ? argv[2] : "../include/config/obstacles/zigzag/obstacles.csv";
    std::string envName       = (argc > 3) ? argv[3] : "zigzag";

    // The KPAX baseline runs by default; pass --skip-baselines to omit it.
    // --dump-viz additionally dumps run-0's full tree per variant for the spatial /
    // tree-growth visualization (Data/Benchmarks/KinoPaxStarCostTuning/viz/).
    //
    // --only-<planner> runs exactly that one planner and nothing else. run_paper_benchmark_v2.sh
    // now invokes this binary once PER PLANNER (see its header for why: a true CUDA hang has no
    // in-process recovery -- the host thread that would notice is the same one stuck inside a
    // blocking CUDA call -- so isolating each planner to its own timeout-wrapped process is what
    // actually bounds a hang's blast radius). Passing none still runs the full 4-planner
    // comparison, for any manual/ad hoc invocation; passing more than one --only-* flag is an error.
    bool skipBaselines       = false;
    bool onlyKPAX            = false;
    bool onlyKinoPaxPlus     = false;
    bool onlyKinoPaxSTARTrue = false;
    bool onlyCountingStars   = false;
    for(int i = 4; i < argc; i++)
    {
        if(std::string(argv[i]) == "--skip-baselines")
            skipBaselines = true;
        else if(std::string(argv[i]) == "--dump-viz")
            g_dumpViz = true;
        else if(std::string(argv[i]) == "--only-kpax")
            onlyKPAX = true;
        else if(std::string(argv[i]) == "--only-kinopaxplus")
            onlyKinoPaxPlus = true;
        else if(std::string(argv[i]) == "--only-kinopaxstartrue")
            onlyKinoPaxSTARTrue = true;
        else if(std::string(argv[i]) == "--only-countingstars")
            onlyCountingStars = true;
    }

    enum class PlannerSelect { All, KPAX, KinoPaxPlus, KinoPaxSTARTrue, CountingStars };
    PlannerSelect selected = PlannerSelect::All;
    {
        int onlyCount = (int)onlyKPAX + (int)onlyKinoPaxPlus + (int)onlyKinoPaxSTARTrue + (int)onlyCountingStars;
        if(onlyCount > 1)
        {
            fprintf(stderr, "ERROR: pass at most one --only-* flag (got %d).\n", onlyCount);
            return 1;
        }
        if(onlyKPAX)                 selected = PlannerSelect::KPAX;
        else if(onlyKinoPaxPlus)     selected = PlannerSelect::KinoPaxPlus;
        else if(onlyKinoPaxSTARTrue) selected = PlannerSelect::KinoPaxSTARTrue;
        else if(onlyCountingStars)   selected = PlannerSelect::CountingStars;
    }
    bool runKPAX            = (selected == PlannerSelect::All || selected == PlannerSelect::KPAX);
    bool runKinoPaxPlus     = (selected == PlannerSelect::All || selected == PlannerSelect::KinoPaxPlus);
    bool runKinoPaxSTARTrue = (selected == PlannerSelect::All || selected == PlannerSelect::KinoPaxSTARTrue);
    bool runCountingStars   = (selected == PlannerSelect::All || selected == PlannerSelect::CountingStars);
    const char* selectedName =
        selected == PlannerSelect::KPAX             ? "KPAX ONLY (--only-kpax)" :
        selected == PlannerSelect::KinoPaxPlus       ? "KinoPaxPlus ONLY (--only-kinopaxplus)" :
        selected == PlannerSelect::KinoPaxSTARTrue   ? "KinoPaxSTARTrue ONLY (--only-kinopaxstartrue)" :
        selected == PlannerSelect::CountingStars     ? "CountingStars ONLY (--only-countingstars)" :
                                                        "full comparison";

    const int NUM_KPAX_RUNS        = 30;
    const int NUM_KINOPAXPLUS_RUNS = 30;   // drives the KinoPaxPlus runner
    const int NUM_TRUE_RUNS        = 30;    // drives the KinoPaxSTARTrue anc0/anc1 pair
    const int NUM_CS_RUNS          = 30;    // drives the CountingStars fixed point
    const int MAX_ITERATIONS       = 1000;
    const float MAX_TIME_MS      = 10000.0f;  // 10 second per-run timeout

    // Per-environment subfolder so each environment can be plotted independently.
    std::string outputDir = "Data/Benchmarks/PaperBenchmarkV2/" + envName;
    std::filesystem::create_directories(outputDir);

    printf("=======================================================\n");
    printf("    PAPER BENCHMARK V2\n");
    printf("=======================================================\n");
    printf("Delta label:    %s\n", deltaLabel.c_str());
    printf("NUM_R1_REGIONS: %d\n", NUM_R1_REGIONS);
    printf("MAX_TREE_SIZE:  %d\n", MAX_TREE_SIZE);
    printf("W_R1_LENGTH=%d  C_R1_LENGTH=%d  V_R1_LENGTH=%d\n", W_R1_LENGTH, C_R1_LENGTH, V_R1_LENGTH);
    printf("Obstacle file:  %s\n", obstaclePath.c_str());
    printf("Environment:    %s\n", envName.c_str());
    printf("Mode:           %s\n", selectedName);
    printf("Baselines:      %s (KPAX, %d runs)\n", (skipBaselines || !runKPAX) ? "NO" : "YES", NUM_KPAX_RUNS);
    printf("Cost metric:    %s (COST_MODE=%d)\n", (COST_MODE == 1) ? "control effort" : "workspace path length", COST_MODE);
    printf("Dump viz:       %s\n", g_dumpViz ? "YES (run 0 per variant)" : "NO");
    if(runKinoPaxPlus)
        printf("KinoPaxPlus:    %d runs\n", NUM_KINOPAXPLUS_RUNS);
    // Fixed 4-planner comparison -- no grid here (see top-of-file comment). This tool's job is
    // reproducing paper_benchmark.cu's own comparison (KPAX, KinoPaxPlus, KinoPaxSTARTrue,
    // CountingStars) at discretizations already confirmed not to hang.
    if(runKinoPaxSTARTrue)
    {
        printf("KinoPaxSTARTrue: syclopCap = %.2f (no cap), ancestorPrune = 1, %d runs\n", 1.0f, NUM_TRUE_RUNS);
    }
    if(runCountingStars)
    {
        int bStartEcho = (int)floorf(CS_BUFFER_FLOOR * float(MAX_TREE_SIZE) / float(CS_RAMP_FILL_ITERS));
        int bEndEcho   = (int)floorf((CS_BUFFER_SLOPE + CS_BUFFER_FLOOR) * float(MAX_TREE_SIZE) / float(CS_RAMP_FILL_ITERS));
        printf("CountingStars:  bufferSlope=%.2f, bufferFloor=%.2f (B: %d -> %d), explore_frac=%.2f, "
               "cost_frac=%.2f, hopelessGuard=%d (permanently ON, v3.5), %d runs\n",
               CS_BUFFER_SLOPE, CS_BUFFER_FLOOR, bStartEcho, bEndEcho, CS_EXPLORE_FRAC, CS_COST_FRAC,
               CS_HOPELESS_GUARD, NUM_CS_RUNS);
    }
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
    // Obstacle CSVs are authored as fractions of a unit [0,1] workspace -- scale into this
    // build's actual workspace bounds, exactly like h_initial/h_goal above already do. A no-op
    // for every model except Quad (W_MIN=0, W_SIZE=1.0f for Double Integrator/Dubins/Unicycle;
    // Quad's own checked-in config.h uses W_MIN=0, W_SIZE=100.0f instead).
    for(float& coord : obstacles)
    {
        coord = W_MIN + coord * W_SIZE;
    }
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
    // KPAX has a confirmed buffer-overshoot bug in propagateFrontier()'s h_propIterations_==0
    // edge case (KPAX.cu:254-271): when frontier-repeat-count exceeds remaining tree capacity,
    // the fallback launches with the full (oversized) repeat count instead of clamping to what
    // remains, so h_treeSize_ can end up exceeding MAX_TREE_SIZE. First seen as a hang under
    // Dubins Airplane (tree exploding 125->845,106 in 5 iterations), then a
    // cudaErrorIllegalAddress crash under Quad on Jetson. Rather than patch that kernel-launch
    // logic yet, tiny's own region count was reduced instead (MODEL_C_R1S/MODEL_V_R1S in
    // run_paper_benchmark_v2.sh, 592,704 -> 373,248 for Quad specifically) to test whether a smaller discretization
    // keeps the tree far enough from MAX_TREE_SIZE to never trigger the edge case at all. If
    // this experiment doesn't hold, restore the hard skip below (kept as the fallback):
    //   bool skipKPAXThisDelta = deltaLabel.rfind("tiny", 0) == 0;
    bool skipKPAXThisDelta = false;
    if(runKPAX && !skipBaselines && !skipKPAXThisDelta)
    {
        runKPAXBaseline(deltaLabel, envName, h_initial, h_goal, d_obstacles, numObstacles,
                        all_results, outputDir, NUM_KPAX_RUNS, MAX_ITERATIONS, MAX_TIME_MS);
    }
    else if(runKPAX && skipKPAXThisDelta)
    {
        printf("KPAX: SKIPPED at delta=%s (tiny excluded from KPAX -- see comment above)\n", deltaLabel.c_str());
    }

    // --- KinoPaxPlus delta benchmark ---
    if(runKinoPaxPlus)
    {
        runKinoPaxPlusBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                                all_results, outputDir, deltaLabel, NUM_KINOPAXPLUS_RUNS, MAX_ITERATIONS, MAX_TIME_MS);
    }

    // Fixed 4-planner comparison -- no grid here (see top-of-file comment). This tool's job is
    // reproducing paper_benchmark.cu's own comparison (KPAX, KinoPaxPlus, KinoPaxSTARTrue,
    // CountingStars) at discretizations already confirmed not to hang.
    if(runKinoPaxSTARTrue)
    {
        // --- KinoPaxSTARTrue: one fixed point, ancestorPrune=1 (the guarded stale-best prune on
        // top of the naive KPAX/KinoPaxPlus fusion) -- matches paper_benchmark.cu's series. ---
        runKinoPaxSTARTrueBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                                all_results, outputDir, deltaLabel, NUM_TRUE_RUNS, MAX_ITERATIONS, MAX_TIME_MS);
    }

    if(runCountingStars)
    {
        // --- CountingStars: one fixed operating point (the one paper_benchmark.cu also uses). ---
        runCountingStarsBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                                all_results, outputDir, deltaLabel, NUM_CS_RUNS, MAX_ITERATIONS, MAX_TIME_MS);
    }

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
