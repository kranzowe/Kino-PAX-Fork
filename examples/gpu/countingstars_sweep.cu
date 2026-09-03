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

// ---- CountingStars v3 grid: BUFFER SHARE x FRESHNESS SHARE x CHEAPNESS SHARE ----
//
// B IS NO LONGER AN AXIS -- IT IS DERIVED. v2 swept goal_frontier_size over {200, 2000, 6000, 10000}
// with no derivation behind any of them. v3 computes it inside the planner:
//
//     B = floor(fill_frac * MAX_TREE_SIZE / MAX_ITER)
//
// "the frontier size that fills the tree exactly at MAX_ITER", scaled by fill_frac -- so the
// remaining (1 - fill_frac) of the buffer is what the uncapped OPTIMAL door is left to spend. The
// buffer and the iteration count are the only things that actually constrain a frontier size, so
// they are what B should be a function of. At this sweep's config (MAX_TREE_SIZE 3e6, MAX_ITER 300)
// the three points give B = 2500 / 5000 / 7500.
//
// READ goal_frontier_size OUT OF THE CSV rather than deriving it here. It is a column now, precisely
// so a second copy of this arithmetic does not have to live in the plot script.
static const float FILL_FRACS[] = {0.25f, 0.5f, 0.75f};
static const int NUM_FILL_FRACS = sizeof(FILL_FRACS) / sizeof(FILL_FRACS[0]);

// Share of B given to the FRESHEST door (lowest region ordinality), and to v3's new CHEAPEST door
// (smallest cost distance). Whatever the two leave goes to the uniform DRAW:
//
//     react_frac = 1 - explore_frac - cost_frac
//
// so the largest point on this grid (0.4, 0.4) still leaves the draw 0.2 and react_frac is never
// negative. cross_check_countingstars_grid.py asserts that.
//
// 0 IS ON BOTH AXES AND IS A REAL ABLATION ARM, not a degenerate case: X = 0 makes the cutoff solve
// return cutoff 0 / pBoundary 0 and the door admits exactly nothing. (0, 0) is therefore the
// internal control -- OPTIMAL plus the GUARANTEE plus a full-B uniform draw and nothing else. If no
// other point beats it, neither selection door is earning its share of the budget.
//
// The label tokens are round(1000 x frac), matching v2's `_f` convention -- see countingStarsLabel().
static const float EXPLORE_FRACS[] = {0.0f, 0.2f, 0.4f};
static const int NUM_EXPLORE_FRACS = sizeof(EXPLORE_FRACS) / sizeof(EXPLORE_FRACS[0]);

static const float COST_FRACS[] = {0.0f, 0.2f, 0.4f};
static const int NUM_COST_FRACS = sizeof(COST_FRACS) / sizeof(COST_FRACS[0]);

// Blocks a node receives when it lands in a region the search has barely touched
// (validVertexCounter[r] < CS_NOVEL_THRESH), or when it cleared BOTH selection cutoffs. Everything
// else -- populated regions, and both reactivation arms -- gets 1, matching KPAXCap and CleanCost.
//
// HELD AT 4, NOT SWEPT. The previous pass swept {1, 4} and answered the question the region-keyed
// fan-out rule was asking; v3 changes the admission rule, not the fan-out rule, and the plot script
// has exactly three style channels for the three fraction axes. Restoring it as an axis means
// finding a fourth channel or splitting the figure.
static const int CS_MAX_BLOCKS = 4;

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
static const float CS_DERIVED_FILL_FRAC    = 0.75f;
static const float CS_DERIVED_EXPLORE_FRAC = 0.2f;
static const float CS_DERIVED_COST_FRAC    = 0.4f;
static const float CAP_DERIVED             = 0.03f;

static bool g_singlePoint = false;

static bool capSkip(float cap)
{
    return g_singlePoint && fabsf(cap - CAP_DERIVED) > 1e-6f;
}

// Single source of truth for the CountingStars grid's shape: the runner and the banner both call
// it, so the printed point count can never drift from the grid actually executed.
static bool countingStarsSkip(float fillFrac, float exploreFrac, float costFrac)
{
    // FULL FACTORIAL: 3 fill x 3 explore x 3 cost = 27 points. --single-point is the only skip.
    // The two fraction axes cannot sum above 0.8 on this grid, so nothing is skipped for a negative
    // react_frac -- but cross_check_countingstars_grid.py asserts it rather than trusting the values.
    if(!g_singlePoint) return false;
    return fabsf(fillFrac - CS_DERIVED_FILL_FRAC) > 1e-6f
        || fabsf(exploreFrac - CS_DERIVED_EXPLORE_FRAC) > 1e-6f
        || fabsf(costFrac - CS_DERIVED_COST_FRAC) > 1e-6f;
}

static int countingStarsPointCount()
{
    int n = 0;
    for(int bi = 0; bi < NUM_FILL_FRACS; bi++)
    for(int ei = 0; ei < NUM_EXPLORE_FRACS; ei++)
    for(int ci = 0; ci < NUM_COST_FRACS; ci++)
        if(!countingStarsSkip(FILL_FRACS[bi], EXPLORE_FRACS[ei], COST_FRACS[ci])) n++;
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

// "CountingStars_ff75_ef200_cf400_mb4". MUST start with a name loadRuns() dispatches on.
//
//   ff   fill_frac,    round(100 x float)   -- B is DERIVED from this, and is a CSV column
//   ef   explore_frac, round(1000 x float)
//   cf   cost_frac,    round(1000 x float)
//   mb   maxBlocks, plain integer
//
// THE `_B` TOKEN IS GONE because B is no longer a setting. v2's CSVs are `_B..._f..._mb...` and
// cannot collide with this shape, so they simply stop loading -- which is the intended outcome for a
// planner whose admission rule changed, not a loss.
//
// THE FRACTION TOKENS STAY AT 1000x, matching v2's `_f`. The convention arrived when a grid reached
// 0.001 (which rounds to 0 at 100x) and is kept so a stale CSV cannot be silently read as the wrong
// series. fill_frac is 100x instead because it is a coarse axis on {0.25, 0.5, 0.75} and `ff75`
// reads as three quarters where `ff750` does not.
//
// `_mb` AND NOT `_b` FOR maxBlocks: a case-only distinction between two integer tokens in one
// filename is a misread waiting to happen. It is retained in the label although maxBlocks is fixed,
// so restoring it as an axis later does not change the filename shape again.
static std::string countingStarsLabel(float fillFrac, float exploreFrac, float costFrac, int maxBlocks)
{
    char buf[160];
    snprintf(buf, sizeof(buf), "CountingStars_ff%d_ef%d_cf%d_mb%d",
             (int)lroundf(100.0f * fillFrac),
             (int)lroundf(1000.0f * exploreFrac),
             (int)lroundf(1000.0f * costFrac), maxBlocks);
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
    //   3. WHICH DOOR BUILT THE TREE. optimal_count / admitted_explore / admitted_costdist /
    //      reactivated_best / reactivated_count, plus ord_cutoff and cost_cutoff_dist. A cutoff
    //      RISING over a run is expected -- regions fill, so both signals get scarce. Pinned at 0
    //      means no candidate is ever good enough on that signal and the fraction is doing nothing.
    //
    //      admitted_explore AND admitted_costdist OVERLAP. The two selection doors are a union over
    //      one candidate pool, so admitted_both is what makes them add back up:
    //      admitted == optimal_count + admitted_explore + admitted_costdist - admitted_both.
    //   4. BLOCK IDENTITY. frontier_repeat_size must equal the sum of the frontier's admission-time
    //      block counts after scaling, and prop_attempted / frontier_repeat_size must be EXACTLY 32
    //      on every iteration. Kernel1 is retained by construction, so below 32 is a defect.
    int   prop_attempted;         // propagations launched this iteration, collisions included
    int   prop_valid;             // collision-free candidates the accept passes judged
    int   frontier_repeat_size;   // sum of the per-node block counts; x32 is the kernel1 attempt count
    // The budget's own arithmetic, as applied THIS iteration. optimal_count is what the uncapped
    // top door took before anything else was offered any of the budget; ord_cutoff and cost_cutoff
    // are the two thresholds the fixed shares bought.
    //
    // guaranteed_react IS GONE. It was the guarantee's PLANNED size, and it existed only because
    // v2's draw probability was a remainder that had to know it. v3's draw is a fixed share, so
    // nothing consumes the plan and reactivated_best is the guarantee's realised size, counted
    // exactly on the device.
    int   optimal_count;
    int   ord_cutoff;
    int   budget_used;
    // ---- v3 ----
    // B, DERIVED inside the planner from fill_frac. A column rather than a per-series constant the
    // plot script carries alongside the label, which is what v2 had to do: the budget figure divides
    // budget_used by it, and re-deriving floor(fill * MAX_TREE_SIZE / MAX_ITER) in MATLAB would put
    // a second copy of that arithmetic somewhere it could drift.
    int   goal_frontier_size;
    // The three shares as applied. Settings rather than measurements, in the data for the same
    // reason max_blocks is: the panels can be read without parsing filenames. react_frac is derived
    // (1 - explore - cost) and is logged so a caller that oversubscribed the budget is visible.
    float fill_frac;
    float explore_frac;
    float cost_frac;
    float react_frac;
    // The cost door's cutoff. READ cost_cutoff_dist, NOT cost_cutoff: the bucket index is only
    // meaningful against the dist_max that produced it, and dist_max moves every iteration.
    //
    // cost_cutoff_dist collapsing toward dist_max / 2^21 means every candidate is landing in bucket
    // 0 and the boundary roll has degraded the door to a uniform draw among near-optimal candidates
    // -- the signal that the log bucket map has the wrong shape for this distribution.
    int   cost_cutoff;
    float cost_cutoff_dist;
    float dist_max;
    // ---- v3.1: Part B's cost arm ----
    // The reactivation cutoff, over DORMANT TREE NODES rather than candidates. Read
    // react_cutoff_dist against dist_max exactly as for the candidate door -- and note dist_max is
    // the CANDIDATE anchor, reused: a dormant node above it clamps into the top bucket, which is
    // harmless (this arm takes the SMALLEST distances) unless the cutoff itself pins there, which
    // means the budget exceeded the population below dist_max.
    //
    // dormant_count is the arm's population and the denominator that makes react_floor's yield
    // (~ react_floor * dormant_count) readable.
    int   react_cutoff;
    float react_cutoff_dist;
    int   dormant_count;
    // The completeness floor. A SETTING, and one that is deliberately not swept -- its job is to be
    // non-zero. Logged so the value behind any run is auditable.
    float react_floor;
    // A swept axis that is a setting rather than a measurement. In the data so the fan-out panel
    // can be read against the maxBlocks that produced it, without parsing filenames.
    // -1 for non-CountingStars rows.
    int   max_blocks;
    // Admissions by door, counted exactly on the device. admitted_cost is the OPTIMAL door
    // (distance 0); admitted_costdist is v3's cost-distance door; admitted_both is its overlap with
    // admitted_explore.
    int   admitted_explore;
    int   admitted_cost;
    int   admitted_costdist;
    int   admitted_both;
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
    d.budget_used = -1;
    d.goal_frontier_size = -1;
    d.fill_frac = NAN;
    d.explore_frac = NAN;
    d.cost_frac = NAN;
    d.react_frac = NAN;
    d.cost_cutoff = -1;
    d.cost_cutoff_dist = NAN;
    d.dist_max = NAN;
    d.react_cutoff = -1;
    d.react_cutoff_dist = NAN;
    d.dormant_count = -1;
    d.react_floor = NAN;
    d.max_blocks = -1;
    d.admitted_explore = -1;
    d.admitted_cost = -1;
    d.admitted_costdist = -1;
    d.admitted_both = -1;
    d.reactivated_cost = -1;
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
    //                  e.g. KinoPaxSTARCleanCost_w90_k400_cap5, CountingStars_ff75_ef200_cf400_mb4
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
         << "optimal_count,ord_cutoff,budget_used,max_blocks,"
         << "goal_frontier_size,fill_frac,explore_frac,cost_frac,react_frac,"
         << "cost_cutoff,cost_cutoff_dist,dist_max,"
         << "react_cutoff,react_cutoff_dist,dormant_count,react_floor,"
         << "admitted_explore,admitted_cost,admitted_costdist,admitted_both,"
         << "reactivated_cost,reactivated_count,reactivated_best,"
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
             << d.budget_used << ","
             << d.max_blocks << ","
             << d.goal_frontier_size << ","
             << std::fixed << std::setprecision(4) << d.fill_frac << ","
             << std::fixed << std::setprecision(4) << d.explore_frac << ","
             << std::fixed << std::setprecision(4) << d.cost_frac << ","
             << std::fixed << std::setprecision(4) << d.react_frac << ","
             << d.cost_cutoff << ","
             << std::fixed << std::setprecision(9) << d.cost_cutoff_dist << ","
             << std::fixed << std::setprecision(9) << d.dist_max << ","
             << d.react_cutoff << ","
             << std::fixed << std::setprecision(9) << d.react_cutoff_dist << ","
             << d.dormant_count << ","
             << std::scientific << std::setprecision(3) << d.react_floor << ","
             << std::fixed
             << d.admitted_explore << ","
             << d.admitted_cost << ","
             << d.admitted_costdist << ","
             << d.admitted_both << ","
             << d.reactivated_cost << ","
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
// CountingStars v3 benchmark + runner.
// A DERIVED NODE BUDGET split by three fixed shares: explore_frac to the freshest regions,
// cost_frac to the smallest cost distances, and the rest to a uniform draw -- with the optimal door
// and the region-best guarantee uncapped on top.
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
    float fillFrac,
    float exploreFrac,
    float costFrac,
    int maxBlocks,
    const std::string& label)
{
    // Override the planner's defaults for this run. resetPlanner (called below) does not touch the
    // tunables, so setting them at entry holds for the whole run.
    //
    // B IS NOT SET HERE. resetPlanner derives it from fill_frac, and the ordering is what makes that
    // work: these assignments happen BEFORE the reset, exactly as d_nodeBlocks_'s fill from
    // h_maxBlocks_ already relies on. Writing h_goalFrontierSize_ here would be overwritten.
    //
    // h_fillIters_ IS DELIBERATELY LEFT AT MAX_ITER. maxIterations below is this benchmark's own
    // cap and is usually larger, but B means "the frontier that fills the tree by the end of a run"
    // and MAX_ITER is what config.h calls the end of a run. Setting it to maxIterations would make
    // the same fill_frac mean a different B in this binary than in a plan() call.
    planner.h_fillFrac_    = fillFrac;
    planner.h_exploreFrac_ = exploreFrac;
    planner.h_costFrac_    = costFrac;
    planner.h_maxBlocks_   = maxBlocks;

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
        // output -- and v3.1 gave Part B a THIRD arm, so the identity is
        //
        //     reactivated  ==  reactivated_best + reactivated_cost + reactivated_count
        //
        // NOT reactivated_count alone, which is now only the completeness floor. The two sides are
        // computed independently (a thrust::count here, atomicAdds in the kernel), so the sum is a
        // free check on all three arms of Part B.
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
        d.budget_used          = (int)planner.h_budgetUsed_;
        d.max_blocks           = planner.h_maxBlocks_;
        // B is DERIVED by the planner, so it is read back out of it rather than echoed from the
        // sweep's own axis -- which is what makes the column a check on the derivation, not a copy.
        d.goal_frontier_size   = planner.h_goalFrontierSize_;
        d.fill_frac            = planner.h_fillFrac_;
        d.explore_frac         = planner.h_exploreFrac_;
        d.cost_frac            = planner.h_costFrac_;
        d.react_frac           = planner.h_reactFrac_;
        d.cost_cutoff          = planner.h_costCutoff_;
        d.cost_cutoff_dist     = planner.h_costCutoffDist_;
        d.dist_max             = planner.h_distMax_;
        d.react_cutoff         = planner.h_reactCutoff_;
        d.react_cutoff_dist    = planner.h_reactCutoffDist_;
        d.dormant_count        = (int)planner.h_dormantCount_;
        d.react_floor          = planner.h_reactFloor_;
        d.admitted_explore     = (int)planner.h_admittedExplore_;
        d.admitted_cost        = (int)planner.h_admittedCost_;
        d.admitted_costdist    = (int)planner.h_admittedCostDist_;
        d.admitted_both        = (int)planner.h_admittedBoth_;
        d.reactivated_cost     = (int)planner.h_reactivatedCost_;
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

    for(int bi = 0; bi < NUM_FILL_FRACS; bi++)
    for(int ei = 0; ei < NUM_EXPLORE_FRACS; ei++)
    for(int ci = 0; ci < NUM_COST_FRACS; ci++)
    {
        const float fillFrac    = FILL_FRACS[bi];
        const float exploreFrac = EXPLORE_FRACS[ei];
        const float costFrac    = COST_FRACS[ci];
        const int   maxBlocks   = CS_MAX_BLOCKS;

        if(countingStarsSkip(fillFrac, exploreFrac, costFrac)) continue;

        const std::string label = countingStarsLabel(fillFrac, exploreFrac, costFrac, maxBlocks);

        printf("  --- fill_frac = %.2f (B ~ %d), explore_frac = %.3f, cost_frac = %.3f, react_frac = %.3f (%s) ---\n",
               fillFrac, (int)floorf(fillFrac * float(MAX_TREE_SIZE) / float(MAX_ITER)),
               exploreFrac, costFrac, 1.0f - exploreFrac - costFrac, label.c_str());
        CountingStars planner;
        for(int run = 0; run < numRuns; run++)
        {
            RunResult result = benchmarkCountingStars(planner, deltaLabel, environment_name, run,
                                                 h_initial, h_goal, d_obstacles,
                                                 numObstacles, maxIterations, maxTimeMs,
                                                 fillFrac, exploreFrac, costFrac, maxBlocks, label);
            printf("  ff=%.2f ef=%.3f cf=%.3f Run %d/%d: %.3fs, %d itr, tree=%d, first_sol_itr=%d, cost=%.3f -> %.3f\n",
                   fillFrac, exploreFrac, costFrac,
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
    // fill_frac 0.75, explore_frac 0.2, cost_frac 0.4). The finer discretizations use it: the grid
    // proper happens at the coarse delta, and the finer ones only need the operating point so the
    // deltas can be overlaid like with like.
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
        printf("CountingStars:  fill_frac {");
        for(int i = 0; i < NUM_FILL_FRACS; i++)
            printf("%s%.2f", i ? ", " : "", FILL_FRACS[i]);
        printf("} x explore_frac {");
        for(int i = 0; i < NUM_EXPLORE_FRACS; i++)
            printf("%s%.3f", i ? ", " : "", EXPLORE_FRACS[i]);
        printf("} x cost_frac {");
        for(int i = 0; i < NUM_COST_FRACS; i++)
            printf("%s%.3f", i ? ", " : "", COST_FRACS[i]);
        printf("}   maxBlocks %d (held)\n", CS_MAX_BLOCKS);
        printf("                B IS DERIVED, NOT SWEPT:\n"
               "                  B = floor(fill_frac * MAX_TREE_SIZE / MAX_ITER) = ");
        for(int i = 0; i < NUM_FILL_FRACS; i++)
            printf("%s%d", i ? " / " : "", (int)floorf(FILL_FRACS[i] * float(MAX_TREE_SIZE) / float(MAX_ITER)));
        printf("\n");
        printf("                THREE FIXED SHARES OF B: explore_frac to the FRESHEST door,\n"
               "                cost_frac to v3's new CHEAPEST door (smallest cost distance, chosen\n"
               "                by a log-bucketed histogram rather than the sort that kept\n"
               "                breaking), and 1 - explore - cost to the uniform DRAW. The two\n"
               "                selection doors are a UNION over one candidate pool, so\n"
               "                admitted_explore and admitted_costdist overlap and admitted_both is\n"
               "                what makes them add back up.\n"
               "                THE OPTIMAL DOOR AND THE PART B GUARANTEE ARE UNCAPPED and spend on\n"
               "                top of those shares. Both are bounded by NUM_R1_REGIONS (%d) rather\n"
               "                than by B, so B binds only ABOVE that count; below it budget_used\n"
               "                runs over B and the two fractions steer a minority of the frontier.\n"
               "                FAN-OUT IS REGION-KEYED: a node gets maxBlocks only if its region\n"
               "                has seen < %d valid propagations, or if it cleared BOTH cutoffs;\n"
               "                else 1, and reactivations always 1.\n"
               "                READ FIRST: frontier_repeat_size/frontier_size (the realised mean\n"
               "                rep), then budget_used/goal_frontier_size as a CURVE (the iteration\n"
               "                it crosses 1 is where B stops binding), then admitted_costdist\n"
               "                against admitted_explore, then cost_cutoff_dist against dist_max --\n"
               "                a collapse toward dist_max/2^21 means every candidate is in bucket 0\n"
               "                and the cost door has degraded to a uniform draw.\n"
               "                (0, 0) IS THE INTERNAL CONTROL: both selection doors off, so the\n"
               "                frontier is optimal + guarantee + a full-B draw and nothing else.\n"
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
