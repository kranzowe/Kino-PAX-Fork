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
#include "planners/KPAXCap.cuh"
#include <thrust/count.h>
#include <thrust/reduce.h>

// --- Visualization dump (opt-in via --dump-viz); set in main(), read by the runners. ---
static bool        g_dumpViz = false;
static std::string g_vizDir;

// ---- CountingStars v2 grid: NODE BUDGET x FRESHNESS SHARE x FAN-OUT ----
//
// goal_frontier_size B IS THE PRIMITIVE. The four doors fill it in priority order -- OPTIMAL
// (distance 0, uncapped), FRESHEST (explore_frac of what is left, from the least-populated
// regions), the region-best GUARANTEE, then a uniform DRAW -- so F is an INPUT and
// propagations-per-node is the output.
//
// COST ACCEPTANCE IS PERMANENT. An earlier pass carried a cost_accept toggle that removed both
// cost-driven doors (OPTIMAL and the Part B GUARANTEE) to test whether they were costing time to
// first solution. That experiment is over and the doors stay: they are what makes the search
// converge on cost at all, and without them nothing preferentially expands cheap nodes.
//
// FAN-OUT IS NOW REGION-KEYED, and that is what this pass is testing. The old rule gave every
// frontier node the same block count (the design-budget split came out at maxBlocks for everyone
// in the nominal case), so at B = 10000 / maxBlocks = 16 the planner produced ~224 candidates per
// node admitted where KPAXCap and CleanCost produce ~32 -- the bulk of the runtime gap, and it was
// the propagation kernel rather than bookkeeping. Blocks now follow REGION THINNESS, exactly as
// both ancestors do.
//
// READ frontier_repeat_size / frontier_size FIRST: that is the realised mean rep. At maxBlocks = 1
// it must be exactly 1; at 4 it should sit near 1 with a small excess from thin regions.
//
// EVERY B ON THIS GRID IS BELOW NUM_R1_REGIONS (27,000 at the coarse delta), and that is a
// deliberate narrowing rather than an oversight. Two doors are uncapped and both are bounded by
// the region count rather than by B -- the OPTIMAL door (at most one region best per region per
// iteration) and the GUARANTEE (at most one node per uncovered region) -- so B stops binding once
// nActive passes it.
//
// WHICH MEANS B BINDS EARLY IN A RUN AND THEN STOPS, at a point that moves with B: almost at once
// at 200, much later at 10000. That is not the axis going inert -- it is the axis acting exactly
// where time-to-first-solution is decided, which is the question this pass is asking. Read
// budget_used/B as a curve over iterations, not as a single number: the iteration where it
// crosses 1 IS the measurement.
//
// A LATE-RUN OVERSHOOT IS THEREFORE EXPECTED AT EVERY POINT and is not a defect. The gap between
// budget_used and B is how much of the frontier the priority doors take before the draw is offered
// anything at all.
static const int GOAL_FRONTIER_SIZES[] = {200, 2000, 6000, 10000};
static const int NUM_GOAL_FRONTIER_SIZES = sizeof(GOAL_FRONTIER_SIZES) / sizeof(GOAL_FRONTIER_SIZES[0]);

// Share of the REMAINING budget (B - optimalCount) handed to the freshness door; the rest goes to
// the guarantee and the draw. 0.01 -> 0.9 is a 90x span, wide enough that if the freshness door
// matters at all it has to show here: at 0.01 it is a rounding error against the guarantee and the
// draw, at 0.9 it takes almost everything the optimal door left.
//
// The label token is round(1000 x frac), NOT 100x -- see countingStarsLabel().
static const float EXPLORE_FRACS[] = {0.01f, 0.5f};
static const int NUM_EXPLORE_FRACS = sizeof(EXPLORE_FRACS) / sizeof(EXPLORE_FRACS[0]);

// Blocks a node receives when it lands in a region the search has barely touched
// (validVertexCounter[r] < CS_NOVEL_THRESH). Everything else -- populated regions, and both
// reactivation arms -- gets 1, matching KPAXCap and CleanCost exactly.
//
// maxBlocks = 1 IS THE REFERENCE POINT. It makes every node rep 1, which is the ancestors' steady
// state: their validVertexCounter is cumulative and gains ~32 per frontier node per iteration, so
// regions cross the threshold almost as soon as they are touched and their x15 is a one-shot burst
// on new ground, not a sustained boost. 4 is the smallest setting where the burst is doing
// anything at all, so the pair brackets "no concentration" against "a little".
//
// 16 IS OFF THE GRID NOW. With the fan-out rule region-keyed the burst only reaches nodes landing
// in thin regions, so the high settings that mattered under the old flat rule are no longer where
// the question is.
//
// Independent of B: B sets the frontier's SIZE, maxBlocks sets propagations PER NODE
// (32 * maxBlocks) for the nodes that earn the burst.
static const int MAX_BLOCKS_GRID[] = {1, 4};
static const int NUM_MAX_BLOCKS = sizeof(MAX_BLOCKS_GRID) / sizeof(MAX_BLOCKS_GRID[0]);

// Which signal the SECOND door ranks on. THE HEADLINE AXIS THIS PASS.
//
//   ORDINALITY  regionNodeCount[r] -- prefers the least-populated regions. v2's behaviour, and the
//               control arm.
//   DISTANCE    (cost - minCostsR1[r]) / costScale -- prefers candidates closest to their OWN
//               region's best.
//
// A SWEPT AXIS RATHER THAN A REPLACEMENT, and that is what makes the comparison readable. Both keys
// spend exactly the same budget by construction -- the optimal door takes its uncapped share first,
// then explore_frac of the remainder goes to the second door either way -- so any difference in
// outcome is the CHOICE OF WHICH CANDIDATES and nothing else. Running them as two branches would
// mean merging two sets of CSVs into one figure by hand.
//
// NOT greedy best-first: distance is per-region normalised, so a node in a far, expensive region
// still scores 0 if it is that region's best. Every region that received candidates contributes its
// best at distance 0, so spatial coverage survives structurally and distance only ranks the rest.
static const int SELECTION_KEYS[] = {CS_KEY_ORDINALITY, CS_KEY_DISTANCE};
static const int NUM_SELECTION_KEYS = sizeof(SELECTION_KEYS) / sizeof(SELECTION_KEYS[0]);

// ---- KinoPaxSTARCleanCost baseline point ----
// Demoted from a 21-point grid to the single well-tuned operating point, as the reference the
// CountingStars grid is read against. Same cleanLabel() format as the cost sweep, so its CSVs
// remain loadable by this plot script.
static const bool  CLEAN_BASE_R2  = false;
static const float CLEAN_BASE_W   = 0.9f;
static const float CLEAN_BASE_K   = 1.0f;
static const float CLEAN_BASE_CAP = 0.03f;

// ---- KPAXCap cap sweep ----
// Stock KPAX with a cap multiplier, which makes it the control arm for the cap itself -- the thing
// CountingStars replaces -- a swept cap is exactly the thing a count replaces.
static const float KPAXCAP_CAPS[] = {0.03f};
static const int NUM_KPAXCAP_CAPS = sizeof(KPAXCAP_CAPS) / sizeof(KPAXCAP_CAPS[0]);

// ---- The derived operating points ----
// --single-point restricts every axis to one point, for a finer-discretization pass that only needs
// the operating point so the deltas can be overlaid like with like. Each of these MUST remain a
// member of its list -- the flag selects BY VALUE, so a derived point outside the grid would run
// nothing at all. cross_check_countingstars_grid.py asserts exactly that.
static const int   CS_DERIVED_GOAL_FRONTIER = 10000;
static const int   CS_DERIVED_MAX_BLOCKS    = 4;
static const float CS_DERIVED_EXPLORE_FRAC  = 0.5f;
// This branch's default and the mode under test.
static const int   CS_DERIVED_SELECTION_KEY = CS_KEY_DISTANCE;
static const float CAP_DERIVED              = 0.03f;

static bool g_singlePoint = false;

static bool capSkip(float cap)
{
    return g_singlePoint && fabsf(cap - CAP_DERIVED) > 1e-6f;
}

// Cap-axis point counting, shared by the runners and the banner so the printed count cannot
// printed point count can never drift from the grid actually executed. The grid is a full
// factorial, so the only skip is --single-point.
// Single source of truth for the CountingStars grid's shape: the runner and the banner both call
// it, so the printed point count can never drift from the grid actually executed.
static bool countingStarsSkip(int goalFrontier, float exploreFrac, int maxBlocks, int selectionKey)
{
    // FULL FACTORIAL: 2 keys x 4 budgets x 3 fracs x 2 maxBlocks = 48 points. --single-point is the
    // only skip.
    if(!g_singlePoint) return false;
    return goalFrontier != CS_DERIVED_GOAL_FRONTIER
        || maxBlocks != CS_DERIVED_MAX_BLOCKS
        || selectionKey != CS_DERIVED_SELECTION_KEY
        || fabsf(exploreFrac - CS_DERIVED_EXPLORE_FRAC) > 1e-6f;
}

static int countingStarsPointCount()
{
    int n = 0;
    for(int ki = 0; ki < NUM_SELECTION_KEYS; ki++)
    for(int bi = 0; bi < NUM_GOAL_FRONTIER_SIZES; bi++)
    for(int mi = 0; mi < NUM_MAX_BLOCKS; mi++)
    for(int ei = 0; ei < NUM_EXPLORE_FRACS; ei++)
        if(!countingStarsSkip(GOAL_FRONTIER_SIZES[bi], EXPLORE_FRACS[ei], MAX_BLOCKS_GRID[mi],
                              SELECTION_KEYS[ki])) n++;
    return n;
}

// Points on a bare cap axis (KPAXCap) under the current --single-point setting.
static int capAxisPointCount(const float* caps, int nCaps)
{
    int n = 0;
    for(int ci = 0; ci < nCaps; ci++)
        if(!capSkip(caps[ci])) n++;
    return n;
}

// "CountingStars_B10000_f100_mb16". MUST start with a name loadRuns() dispatches on.
//
//   B    node budget, plain integer
//   f    second-door share, round(1000 x float)
//   mb   maxBlocks, plain integer
//   k    selection key, "ord" / "dist"  (same on/off-word precedent as cleanLabel's r2on/r2off)
//
// THE `_f` TOKEN IS 1000x, NOT THE 100x `_e` USED ELSEWHERE IN THIS FAMILY. It was introduced when
// the grid reached explore_frac 0.001, which rounds to the token 0 at 100x -- unreadable, and
// indistinguishable from a genuine explore_frac of 0. It is KEPT at 1000x now that 0.001 is gone,
// because changing back would make `_f100` ambiguous against the CSVs already labelled that way.
//
// `_mb` AND NOT `_b` FOR maxBlocks: `B` is already the budget, and a case-only distinction between
// two integer tokens in the same filename is a misread waiting to happen.
//
// THE `_ca` TOKEN IS GONE with the cost_accept toggle. Any CSV still carrying it is from that
// experiment and will simply not match -- which is the intended outcome, not a loss.
static std::string countingStarsLabel(int goalFrontier, float exploreFrac, int maxBlocks,
                                      int selectionKey)
{
    char buf[160];
    snprintf(buf, sizeof(buf), "CountingStars_B%d_f%d_mb%d_k%s",
             goalFrontier, (int)lroundf(1000.0f * exploreFrac), maxBlocks,
             (selectionKey == CS_KEY_DISTANCE) ? "dist" : "ord");
    return std::string(buf);
}

static std::string cleanLabel(bool r2Accept, float w, float k, float cap)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d",
             r2Accept ? "on" : "off",
             (int)lroundf(100.0f * w), (int)lroundf(100.0f * k), (int)lroundf(100.0f * cap));
    return std::string(buf);
}

static std::string kpaxCapLabel(float cap)
{
    char buf[96];
    snprintf(buf, sizeof(buf), "KPAXCap_cap%d", (int)lroundf(100.0f * cap));
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
    //      A persistent SHORTFALL means a door is not filling its share; an OVERSHOOT means the
    //      optimal count exceeded B, which is expected wherever B <= NUM_R1_REGIONS and is exactly
    //      why the low-B points are on the grid. frontier_size says the same thing one iteration
    //      later (it is measured at the top of the next propagateFrontier).
    //   2. IS THE FRONTIER DOING MORE WORK. prop_per_node = prop_attempted / frontier_size, against
    //      KinoPaxPlus's bf, which reaches 40,000 at F = 10. THE POINT OF CONTROLLING F IS
    //      CONTROLLING THIS. If it does not move with B, B is not the lever.
    //   3. WHICH DOOR BUILT THE TREE. optimal_count / admitted_explore / admitted_cost /
    //      reactivated_best / reactivated_count, plus ord_cutoff. A cutoff RISING over a run is
    //      expected -- regions fill, freshness gets scarce. Pinned at 0 means no non-optimal
    //      candidate is ever fresh enough and explore_frac is doing nothing.
    //   4. BLOCK IDENTITY. frontier_repeat_size must equal the sum of the frontier's admission-time
    //      block counts after scaling, and prop_attempted / frontier_repeat_size must be EXACTLY 32
    //      on every iteration. Kernel1 is retained by construction, so below 32 is a defect.
    int   prop_attempted;         // propagations launched this iteration, collisions included
    int   prop_valid;             // collision-free candidates the accept passes judged
    int   frontier_repeat_size;   // sum of the per-node block counts; x32 is the kernel1 attempt count
    // The budget's own arithmetic, as applied THIS iteration. optimal_count is what the top door
    // took off the budget before anything else was offered any of it; ord_cutoff is the freshness
    // threshold the remainder bought; guaranteed_react is the guarantee's PLANNED size (active
    // regions no optimal admission covered), which is what set the draw's probability -- read it
    // against reactivated_best, the realised count, to see how many the kernel skipped.
    int   optimal_count;
    int   ord_cutoff;
    int   guaranteed_react;
    int   budget_used;
    // Swept axes that are settings rather than measurements. In the data so every axis of the grid
    // is filterable without parsing filenames. -1 for non-CountingStars rows.
    int   max_blocks;
    int   selection_key;      // 0 = ordinality, 1 = distance
    // The realised distance threshold. RISING over a run means the candidate pool is drifting away
    // from the region bests -- the tree is filling with mediocre nodes. PINNED near 0 means the
    // door is admitting only near-optimal candidates and has become a second optimal door.
    // NaN on the ordinality path, where no distance threshold exists.
    float dist_cutoff;
    // Admissions by door, counted exactly on the device.
    int   admitted_explore;
    int   admitted_cost;
    int   reactivated_count;
    // reactivated_best / frontier_size approaching 1 means the region-best guarantee IS the
    // frontier -- KinoPaxPlus's regime. Read it together with best_cost before deciding that is bad.
    int   reactivated_best;
    // Fan-out budget. block_scale < 1 means the BUFFER, not the fan-out rule, is setting how hard
    // nodes expand; block_scale near 0 means the rep >= 1 floor ate the budget and the fan-out split
    // is inert, which is a goal_frontier_size problem and no other knob will move it.
    float block_ceiling;
    float block_scale;
    float global_collision_frac;
};

// Blank the CountingStars-only columns. Every other planner calls this, exactly as KinoPaxPlus
// already writes NaN for score_floor / cost_scale: the plot script reads columns by name and
// tolerates NaN, so one schema serves every planner.
static void clearCountingStarsCols(IterationData& d)
{
    d.prop_attempted = -1;
    d.prop_valid = -1;
    d.frontier_repeat_size = -1;
    d.optimal_count = -1;
    d.ord_cutoff = -1;
    d.guaranteed_react = -1;
    d.budget_used = -1;
    d.max_blocks = -1;
    d.selection_key = -1;
    d.dist_cutoff = NAN;
    d.admitted_explore = -1;
    d.admitted_cost = -1;
    d.reactivated_count = -1;
    d.reactivated_best = -1;
    d.block_ceiling = NAN;
    d.block_scale = NAN;
    d.global_collision_frac = NAN;
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
    //   KPAX baseline: {env}_KPAX_delta{build}_run{n}.csv
    //   STAR variants: {env}_{planner label}_delta{build}_run{n}.csv
    //                  e.g. KinoPaxSTARCleanCost_w90_k400_cap5, CountingStars_B10000_e10
    //   KPAXCap:       same planner-label form (KPAXCap_cap5). The "KPAX" arm above is an EXACT
    //                  match, so it cannot swallow these.
    //   KinoPaxPlus:   {env}_delta{label}_run{n}.csv
    // KinoPaxPlus deliberately keys on the DELTA rather than a planner name: that is what keeps
    // the two discretizations (large_* and fine_*) in separate files.
    // The build label carries the cost metric (large_effort / large_length), which is a
    // compile-time property of the binary -- see COST_MODE in helper.cuh.
    if(result.delta_label == "KPAX")
        filename << outputDir << "/" << result.environment << "_KPAX_delta" << result.build_delta
                 << "_run" << result.run_number << ".csv";
    // COUNTINGSTARS MUST BE IN THIS ARM. Falling through to the KinoPaxPlus branch below is not a
    // cosmetic naming problem: that branch keys on the DELTA and omits build_delta entirely, so the
    // length and effort builds write the SAME path and the second silently overwrites the first.
    else if(result.delta_label.rfind("CountingStars", 0) == 0 ||
            result.delta_label.rfind("KinoPaxSTAR", 0) == 0 || result.delta_label.rfind("KPAXCap", 0) == 0)
        filename << outputDir << "/" << result.environment << "_" << result.delta_label << "_delta" << result.build_delta
                 << "_run" << result.run_number << ".csv";
    else
        filename << outputDir << "/" << result.environment << "_delta" << result.delta_label
                 << "_run" << result.run_number << ".csv";

    std::ofstream file(filename.str());
    // score_floor / cost_scale are appended, not inserted -- the plot script reads columns by name
    // via getCol(), which returns [] for a missing one, so older CSVs still load.
    file << "iteration,frontier_size,tree_size,elapsed_time_ms,best_cost,"
         << "num_regions,r2_coverage_pct,mean_vertex_score,reactivated,"
         << "score_floor,cost_scale,"
         << "prop_attempted,prop_valid,frontier_repeat_size,"
         << "optimal_count,ord_cutoff,guaranteed_react,budget_used,max_blocks,"
         << "selection_key,dist_cutoff,"
         << "admitted_explore,admitted_cost,reactivated_count,reactivated_best,"
         << "block_ceiling,block_scale,global_collision_frac\n";

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
             << d.reactivated << ","
             << std::fixed << std::setprecision(9) << d.score_floor << ","
             << std::fixed << std::setprecision(6) << d.cost_scale << ","
             << d.prop_attempted << ","
             << d.prop_valid << ","
             << d.frontier_repeat_size << ","
             << d.optimal_count << ","
             << d.ord_cutoff << ","
             << d.guaranteed_react << ","
             << d.budget_used << ","
             << d.max_blocks << ","
             << d.selection_key << ","
             << std::fixed << std::setprecision(6) << d.dist_cutoff << ","
             << d.admitted_explore << ","
             << d.admitted_cost << ","
             << d.reactivated_count << ","
             << d.reactivated_best << ","
             << std::fixed << std::setprecision(1) << d.block_ceiling << ","
             << std::fixed << std::setprecision(4) << d.block_scale << ","
             << std::fixed << std::setprecision(6) << d.global_collision_frac << "\n";
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
    filename << outputDir << "/countingstars_sweep_" << timestamp.str() << "_summary.csv";

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
        d.num_regions       = NUM_R1_REGIONS;
        d.r2_coverage_pct   = NAN;   // KinoPaxPlus has no R2/vertexScore machinery
        d.mean_vertex_score = NAN;
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
        clearCountingStarsCols(d);
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.num_regions       = NUM_R1_REGIONS;
        d.r2_coverage_pct   = r2CoveragePct;
        d.mean_vertex_score = meanScore;
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
// KPAXCap benchmark + runner (stock KPAX with the Syclop score scaled by h_syclopCap_).
//
// Modelled on benchmarkKPAX, NOT on the CleanCost runner: KPAX-family planners carry no
// h_minCost_, so a goal is detected by zeroing d_pathToGoal_ before each iteration and walking the
// parent chain with devicePathCost() afterwards, outside the timed window.
// ========================================================================
RunResult benchmarkKPAXCap(
    KPAXCap& planner,
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
    const std::string& label)
{
    // resetPlanner does not touch h_syclopCap_, so setting it at entry holds for the run.
    planner.h_syclopCap_ = syclopCap;
    RunResult result;
    result.delta_label = label;
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

        // --- Frontier diagnostics (outside the timed window; KPAXCap uses the KPAX Graph) ---
        int reactivated = (int)thrust::count(planner.d_frontier_.begin(),
                                             planner.d_frontier_.begin() + oldTreeSize, true);
        int inactiveR2 = (int)thrust::count(planner.graph_.d_activeSubVertices_.begin(),
                                            planner.graph_.d_activeSubVertices_.end(), 0);
        float r2CoveragePct = 100.0f * float(NUM_R2_REGIONS - inactiveR2) / float(NUM_R2_REGIONS);
        float scoreSum  = thrust::reduce(planner.graph_.d_vertexScoreArray_.begin(),
                                         planner.graph_.d_vertexScoreArray_.end(), 0.0f);
        float meanScore = scoreSum / float(NUM_R1_REGIONS);

        IterationData d;
        clearCountingStarsCols(d);
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.num_regions       = NUM_R1_REGIONS;
        d.r2_coverage_pct   = r2CoveragePct;
        d.mean_vertex_score = meanScore;
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

void runKPAXCapBenchmark(
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
    printf("KPAXCAP CAP SWEEP: %s | Delta: %s | Regions: %d\n",
           environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    for(int ci = 0; ci < NUM_KPAXCAP_CAPS; ci++)
    {
        const float       cap   = KPAXCAP_CAPS[ci];
        if(capSkip(cap)) continue;
        const std::string label = kpaxCapLabel(cap);

        printf("  --- cap = %.2f (%s) ---\n", cap, label.c_str());
        KPAXCap planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkKPAXCap(planner, deltaLabel, environment_name, run,
                                                h_initial, h_goal, d_obstacles,
                                                numObstacles, maxIterations, maxTimeMs,
                                                cap, label);
            printf("  cap=%.2f Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
                   cap, run + 1, numRuns, result.total_time_seconds, result.total_iterations,
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
// CountingStars v2 benchmark + runner.
// ONE GLOBAL NODE BUDGET, filled in priority order: optimal (uncapped), then explore_frac of what
// is left to the freshest regions, then the region-best guarantee, then a uniform draw.
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
    int goalFrontier,
    float exploreFrac,
    int maxBlocks,
    int selectionKey,
    const std::string& label)
{
    // Override the planner's defaults for this run. resetPlanner (called below) does not touch the
    // tunables, so setting them at entry holds for the whole run.
    //
    // NO RAMP ENDPOINTS TO SET. v1's three counts each had a *0 / *1 pair and an unswept slope
    // hiding between them; v2's budget is a flat scalar, so what the sweep sets is exactly what the
    // planner applies on every iteration.
    planner.h_goalFrontierSize_ = goalFrontier;
    planner.h_exploreFrac_      = exploreFrac;
    planner.h_maxBlocks_        = maxBlocks;
    planner.h_selectionKey_     = selectionKey;

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
        // output -- and Part B HAS TWO ARMS, so the device-side identity is
        //
        //     reactivated  ==  reactivated_best + reactivated_count
        //
        // NOT reactivated_count alone. The two sides are computed independently (a thrust::count
        // here, atomicAdds in the kernel), so the sum is a free check on both arms of Part B.
        int reactivated = (int)thrust::count(planner.d_frontier_.begin(),
                                             planner.d_frontier_.begin() + oldTreeSize, true);
        // r2_coverage_pct from the planner's RUNNING COUNTER, not a sweep of d_activeSubVertices_.
        // Identical value; the old thrust::count was O(NUM_R2_REGIONS) EVERY ITERATION -- 2.1M
        // elements at the coarse delta and 37.9M at `tiny`.
        float r2CoveragePct = 100.0f * float(planner.h_touchedR2_) / float(NUM_R2_REGIONS);
        // NaN, not 0. A zero would read as "the scores collapsed"; there are no scores.
        float meanScore = NAN;

        IterationData d;
        clearCountingStarsCols(d);
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.num_regions       = NUM_R1_REGIONS;
        d.r2_coverage_pct   = r2CoveragePct;
        d.mean_vertex_score = meanScore;
        d.reactivated       = reactivated;
        d.score_floor       = NAN;   // no Syclop score, so no floor
        // NOT NaN any more, and that is a real column here rather than a courtesy: costScale is the
        // DENOMINATOR of a candidate's distance, and distance 0 is the top door. A scale collapsing
        // toward 0 would make the top door's test degenerate, and this is the only place that shows.
        d.cost_scale        = planner.h_costScale_;
        // CountingStars readout.
        d.prop_attempted       = (int)planner.h_propAttempted_;
        d.prop_valid           = (int)planner.h_candidatesPreGate_;
        d.frontier_repeat_size = (int)planner.h_frontierRepeatSize_;
        d.optimal_count        = (int)planner.h_optimalCount_;
        d.ord_cutoff           = planner.h_ordCutoff_;
        d.guaranteed_react     = (int)planner.h_guaranteedReact_;
        d.budget_used          = (int)planner.h_budgetUsed_;
        d.max_blocks           = planner.h_maxBlocks_;
        d.selection_key        = planner.h_selectionKey_;
        d.dist_cutoff          = planner.h_distCutoff_;
        d.admitted_explore     = (int)planner.h_admittedExplore_;
        d.admitted_cost        = (int)planner.h_admittedCost_;
        d.reactivated_count    = (int)planner.h_reactivated_;
        d.reactivated_best     = (int)planner.h_reactivatedBest_;
        d.block_ceiling        = planner.h_blockCeiling_;
        d.block_scale          = planner.h_blockScale_;
        d.global_collision_frac = planner.h_globalCollisionFrac_;
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
    printf("COUNTINGSTARS GRID: %s | Delta: %s | Regions: %d\n",
           environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    // The selection key is the OUTERMOST loop so the two arms come out of the log adjacent at
    // matched (B, frac, mb) -- the comparison this pass exists for is read pairwise.
    for(int ki = 0; ki < NUM_SELECTION_KEYS; ki++)
    for(int bi = 0; bi < NUM_GOAL_FRONTIER_SIZES; bi++)
    for(int mi = 0; mi < NUM_MAX_BLOCKS; mi++)
    for(int ei = 0; ei < NUM_EXPLORE_FRACS; ei++)
    {
        const int   selectionKey = SELECTION_KEYS[ki];
        const int   goalFrontier = GOAL_FRONTIER_SIZES[bi];
        const int   maxBlocks    = MAX_BLOCKS_GRID[mi];
        const float exploreFrac  = EXPLORE_FRACS[ei];
        const char* keyName      = (selectionKey == CS_KEY_DISTANCE) ? "dist" : "ord";

        if(countingStarsSkip(goalFrontier, exploreFrac, maxBlocks, selectionKey)) continue;

        const std::string label = countingStarsLabel(goalFrontier, exploreFrac, maxBlocks, selectionKey);

        printf("  --- key = %s, B = %d, explore_frac = %.3f, maxBlocks = %d (%s) ---\n",
               keyName, goalFrontier, exploreFrac, maxBlocks, label.c_str());
        CountingStars planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkCountingStars(planner, deltaLabel, environment_name, run,
                                                 h_initial, h_goal, d_obstacles,
                                                 numObstacles, maxIterations, maxTimeMs,
                                                 goalFrontier, exploreFrac, maxBlocks, selectionKey, label);
            printf("  k=%s B=%d f=%.3f mb=%d Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
                   keyName, goalFrontier, exploreFrac, maxBlocks,
                   run + 1, numRuns, result.total_time_seconds,
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
}

RunResult benchmarkKinoPaxSTARCleanCost(
    KinoPaxSTARCleanCost& planner,
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
    float acceptCapMul,
    bool r2SeedAccept,
    const std::string& label)
{
    // Override the planner's defaults for this run. resetPlanner (called below) does not
    // touch these, so setting them at entry holds for the run. h_probFloor_ stays at its ctor
    // default (EPSILON).
    planner.h_costWeight_    = costWeight;
    planner.h_costPruneExp_  = costPruneExp;
    planner.h_acceptCapMul_  = acceptCapMul;
    planner.h_r2SeedAccept_  = r2SeedAccept;
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

        // --- Frontier diagnostics (outside the timed window; CleanCost uses the KPAX Graph) ---
        int reactivated = (int)thrust::count(planner.d_frontier_.begin(),
                                             planner.d_frontier_.begin() + oldTreeSize, true);
        int inactiveR2 = (int)thrust::count(planner.graph_.d_activeSubVertices_.begin(),
                                            planner.graph_.d_activeSubVertices_.end(), 0);
        float r2CoveragePct = 100.0f * float(NUM_R2_REGIONS - inactiveR2) / float(NUM_R2_REGIONS);
        float scoreSum  = thrust::reduce(planner.graph_.d_vertexScoreArray_.begin(),
                                         planner.graph_.d_vertexScoreArray_.end(), 0.0f);
        float meanScore = scoreSum / float(NUM_R1_REGIONS);

        IterationData d;
        clearCountingStarsCols(d);
        d.iteration     = itr;
        d.frontier_size = planner.h_frontierSize_;
        d.tree_size     = planner.h_treeSize_;
        d.elapsed_time_ms = plannerMs;
        d.best_cost     = result.final_best_cost;
        d.num_regions       = NUM_R1_REGIONS;
        d.r2_coverage_pct   = r2CoveragePct;
        d.mean_vertex_score = meanScore;
        d.reactivated       = reactivated;
        d.score_floor       = planner.graph_.h_scoreFloor_;
        d.cost_scale        = planner.h_costScale_;
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

void runKinoPaxSTARCleanCostBenchmark(
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
    printf("KINOPAXSTARCLEANCOST BASELINE: %s | Delta: %s | Regions: %d\n",
           environment_name.c_str(), deltaLabel.c_str(), NUM_R1_REGIONS);
    printf("========================================\n");

    // ONE fixed point, not a grid: the well-tuned CleanCost operating point, present as the
    // reference the CountingStars grid is read against. The w x k x cap sweep lives in
    // kinopaxstar_cost_tuning_sweep.cu, which is unchanged and still runnable.
    {
        const bool        r2    = CLEAN_BASE_R2;
        const float       w     = CLEAN_BASE_W;
        const float       k     = CLEAN_BASE_K;
        const float       cap   = CLEAN_BASE_CAP;

        const std::string label = cleanLabel(r2, w, k, cap);

        printf("  --- r2 = %s, w = %.2f, k = %.2f, cap = %.2f (%s) ---\n",
               r2 ? "on" : "off", w, k, cap, label.c_str());
        KinoPaxSTARCleanCost planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkKinoPaxSTARCleanCost(planner, deltaLabel, environment_name, run,
                                                 h_initial, h_goal, d_obstacles,
                                                 numObstacles, maxIterations, maxTimeMs,
                                                 w, k, cap, r2, label);
            printf("  r2=%s w=%.2f k=%.2f cap=%.2f Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
                   r2 ? "on" : "off", w, k, cap, run + 1, numRuns, result.total_time_seconds, result.total_iterations,
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
    //
    // --single-point restricts every axis to its derived operating point (CountingStars at
    // B 10000, explore_frac 0.10). The finer discretizations use it: the grid proper happens at the
    // coarse delta, and the finer ones only need the operating point so the deltas can be overlaid
    // like with like.
    //
    // --only-kinopaxplus runs the KinoPaxPlus series and nothing else. The discretization is a
    // compile-time property (NUM_R1_REGIONS via config.h), so the only way to get KinoPaxPlus at a
    // second, finer delta is a second binary; this flag lets that binary reuse this file instead of
    // re-running the whole grid at a discretization nothing else in the sweep uses.
    bool skipBaselines   = false;
    bool onlyKinoPaxPlus = false;
    for(int i = 4; i < argc; i++)
    {
        if(std::string(argv[i]) == "--skip-baselines")
            skipBaselines = true;
        else if(std::string(argv[i]) == "--dump-viz")
            g_dumpViz = true;
        else if(std::string(argv[i]) == "--only-kinopaxplus")
            onlyKinoPaxPlus = true;
        else if(std::string(argv[i]) == "--single-point")
            g_singlePoint = true;
    }

    const int NUM_KPAX_RUNS        = 5;
    const int NUM_KINOPAXPLUS_RUNS = 5;   // drives the KinoPaxPlus runner
    const int NUM_CS_RUNS          = 3;    // drives the CountingStars grid
    const int NUM_CLEANCOST_RUNS   = 3;    // drives the single CleanCost baseline point
    // KPAXCap runs at the KPAX count, not the grid count: it is the control arm for the cap and is
    // read directly against the KPAX baseline, so the two want a matched noise level.
    const int NUM_KPAXCAP_RUNS     = NUM_KPAX_RUNS;
    const int MAX_ITERATIONS       = 400;
    const float MAX_TIME_MS      = 6000.0f;  // 6 second per-run timeout

    // Per-environment subfolder so house and zigzag can be plotted independently.
    std::string outputDir = "Data/Benchmarks/CountingStars/" + envName;
    std::filesystem::create_directories(outputDir);

    printf("=======================================================\n");
    printf("    COUNTINGSTARS SWEEP\n");
    printf("=======================================================\n");
    printf("Delta label:    %s\n", deltaLabel.c_str());
    printf("NUM_R1_REGIONS: %d\n", NUM_R1_REGIONS);
    printf("MAX_TREE_SIZE:  %d\n", MAX_TREE_SIZE);
    printf("W_R1_LENGTH=%d  C_R1_LENGTH=%d  V_R1_LENGTH=%d\n", W_R1_LENGTH, C_R1_LENGTH, V_R1_LENGTH);
    printf("Obstacle file:  %s\n", obstaclePath.c_str());
    printf("Environment:    %s\n", envName.c_str());
    printf("Mode:           %s\n", onlyKinoPaxPlus ? "KinoPaxPlus ONLY (--only-kinopaxplus)" : "full sweep");
    printf("Grid axis:      %s\n", g_singlePoint ? "SINGLE (--single-point, derived operating point)" : "swept");
    printf("Baselines:      %s (KPAX, %d runs)\n", (skipBaselines || onlyKinoPaxPlus) ? "NO" : "YES", NUM_KPAX_RUNS);
    printf("Cost metric:    %s (COST_MODE=%d)\n", (COST_MODE == 1) ? "control effort" : "workspace path length", COST_MODE);
    printf("Dump viz:       %s\n", g_dumpViz ? "YES (run 0 per variant)" : "NO");
    printf("KinoPaxPlus:    %d runs\n", NUM_KINOPAXPLUS_RUNS);
    if(!onlyKinoPaxPlus)
    {
        // Counted with the same predicate the runner uses, not a closed form -- the previous
        // closed form silently assumed the last WEIGHTS entry was 1.0.
        int csPoints = countingStarsPointCount();
        // THE AXES ARE PRINTED FROM THE ARRAYS, never restated as a literal. A hardcoded banner is
        // a fourth place the grid can drift, and the only one no cross-check reads.
        printf("CountingStars:  goal_frontier_size {");
        for(int i = 0; i < NUM_GOAL_FRONTIER_SIZES; i++)
            printf("%s%d", i ? ", " : "", GOAL_FRONTIER_SIZES[i]);
        printf("} x explore_frac {");
        for(int i = 0; i < NUM_EXPLORE_FRACS; i++)
            printf("%s%.3f", i ? ", " : "", EXPLORE_FRACS[i]);
        printf("} x maxBlocks {");
        for(int i = 0; i < NUM_MAX_BLOCKS; i++)
            printf("%s%d", i ? ", " : "", MAX_BLOCKS_GRID[i]);
        printf("}\n");
        printf("                selection_key {ord, dist} -- THE HEADLINE AXIS. Both keys spend the\n"
               "                SAME budget; they differ only in WHICH candidates the second door\n"
               "                picks. ord = least-populated regions (v2, the control arm); dist =\n"
               "                closest to their own region's best, ranked by an exact sort.\n"
               "                dist is NOT greedy best-first: distance is per-region normalised,\n"
               "                so every region that got candidates contributes its best at 0.\n"
               "                READ admitted_explore ord vs dist at matched (B, frac, mb): both\n"
               "                spend the same budget, so any difference is WHICH candidates.\n"
               "                B is the PRIMITIVE, not a cap: the doors fill it in priority order\n"
               "                and the budget is met by construction.\n"
               "                COST ACCEPTANCE IS PERMANENT -- the OPTIMAL door and the Part B\n"
               "                region-best GUARANTEE always run. Those two are the only UNCAPPED\n"
               "                doors and both are bounded by NUM_R1_REGIONS (%d) rather than by B,\n"
               "                so B binds only ABOVE that count; below it budget_used runs over B.\n"
               "                FAN-OUT IS REGION-KEYED: a node gets maxBlocks only if its region\n"
               "                has seen < %d valid propagations, else 1; reactivations always 1.\n"
               "                That is KPAXCap's and CleanCost's rule, ported so the runtime\n"
               "                comparison is like-for-like. maxBlocks = 1 reproduces their steady\n"
               "                state exactly (every node rep 1).\n"
               "                READ FIRST: frontier_repeat_size/frontier_size (the realised mean\n"
               "                rep -- must be 1 at maxBlocks=1), then prop_attempted/budget_used\n"
               "                (candidates per admitted node; was ~224, target ~32), then time to\n"
               "                first solution against KPAXCap and CleanCost.\n"
               "                -> %d points x %d runs = %d runs\n",
               NUM_R1_REGIONS, CS_NOVEL_THRESH, csPoints, NUM_CS_RUNS, csPoints * NUM_CS_RUNS);
        printf("CleanCost:      r2 OFF, w %.2f, k %.2f, cap %.2f = 1 point x %d runs = %d runs\n",
               CLEAN_BASE_W, CLEAN_BASE_K, CLEAN_BASE_CAP, NUM_CLEANCOST_RUNS, NUM_CLEANCOST_RUNS);
        int kcapPoints = capAxisPointCount(KPAXCAP_CAPS, NUM_KPAXCAP_CAPS);
        printf("KPAXCap:        cap = %d points x %d runs = %d runs\n",
               kcapPoints, NUM_KPAXCAP_RUNS, kcapPoints * NUM_KPAXCAP_RUNS);
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
    if(!skipBaselines && !onlyKinoPaxPlus)
    {
        runKPAXBaseline(deltaLabel, envName, h_initial, h_goal, d_obstacles, numObstacles,
                        all_results, outputDir, NUM_KPAX_RUNS, MAX_ITERATIONS, MAX_TIME_MS);
    }

    // --- KinoPaxPlus delta benchmark (the one series the --only-kinopaxplus pass runs) ---
    runKinoPaxPlusBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                            all_results, outputDir, deltaLabel, NUM_KINOPAXPLUS_RUNS, MAX_ITERATIONS, MAX_TIME_MS);

    if(!onlyKinoPaxPlus)
    {
        // --- CountingStars: goal_frontier_size x explore_frac grid (the point of this sweep) ---
        runCountingStarsBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                                all_results, outputDir, deltaLabel, NUM_CS_RUNS, MAX_ITERATIONS, MAX_TIME_MS);

        // --- KinoPaxSTARCleanCost: single baseline point ---
        runKinoPaxSTARCleanCostBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                                all_results, outputDir, deltaLabel, NUM_CLEANCOST_RUNS, MAX_ITERATIONS, MAX_TIME_MS);

        // --- KPAXCap: cap sweep (control arm for the cap, against the KPAX baseline above) ---
        runKPAXCapBenchmark(envName, h_initial, h_goal, d_obstacles, numObstacles,
                                all_results, outputDir, deltaLabel, NUM_KPAXCAP_RUNS, MAX_ITERATIONS, MAX_TIME_MS);
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
