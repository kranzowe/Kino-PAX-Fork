// ========================================================================
// ACCEPTANCE-REASON BREAKDOWN  (KinoPaxSTARCOMBO, against a KinoPaxSTARCleanCost reference)
//
// The tuning sweeps show THAT the tuning changes outcomes, but not WHY -- the accept kernel's
// single decision hides the mechanism. A candidate enters the frontier through one of three doors:
//
//   1. the region-best exemption      (cost <= minCostsR1[r])            both planners
//   2. the R2 seeding exemption       (virgin sub-region)                CleanCost only, off by
//                                                                        default; REMOVED in COMBO
//   3. the roll                       CleanCost: rand < cap*(w*pSyclop + (1-w)*pCost + floor)
//                                     COMBO:     rand < min(pMax, comboShape * pTargetAccept)
//
// and nothing in the normal per-iteration output distinguishes them. This runner turns on
// h_countAcceptReasons_ and dumps the split per iteration for every point of the COMBO grid, plus
// ONE low-cap CleanCost run as the reference the grid is read against.
//
// ATTRIBUTION. Door 3 is a single Bernoulli draw against a SUM, so "accepted because of X" is not a
// distinction the rule makes. Each accepted node instead splits one unit of credit in proportion to
// each term's share of that sum -- RNG-independent, and taken before any clamp or throttle, since
// those scale the terms equally and cancel in the ratio. The credit measures WHICH TERM ARGUED for
// the node, not how hard the throttle was squeezing. CleanCost splits across (syclop, cost, floor);
// COMBO splits across its three sigmoids (coverage, collision, cost).
//
// WHY IT ALSO LOGS THE BUDGET COLUMNS. The acceptance split alone cannot explain COMBO's behaviour,
// because acceptance is only half the rule -- the growth controller sets the SCALE and the fan-out.
// Two columns in particular are the ones to read first when propagate falls onto kernel2 early:
//
//   n_active     regions with samples. Part B re-activates the region best UNCONDITIONALLY, one
//                per explored region, outside h_reactFrac_ entirely -- so the frontier has a hard
//                floor at nActive. Since rep >= 1, frontierRepeatSize >= F, and kernel2 is forced
//                once 32*F > remaining no matter what repTarget does.
//   reactivated  frontier bits among the pre-existing tree, i.e. Part B's actual output.
//                reactivated / frontier_size near 1 means F is region-best dominated and no
//                acceptance tuning will move the kernel1 crossover.
//
// prop_attempted / frontier_repeat_size is the kernel1 detector: exactly 32 on the kernel1 path,
// less on kernel2. h_propIterations_ alone is NOT valid for this -- it is only assigned inside the
// kernel2 branch, so on kernel1 it holds a stale value from an earlier iteration.
//
// This is a standalone runner, not a benchmark: it drives the raw iteration loop
// (propagateFrontier -> graph_.updateVertices -> updateFrontier) the same way tree_growth_dump.cu
// does. It deliberately reports against ITERATION, never wall-clock: the counting atomics distort
// timing, which is also why h_countAcceptReasons_ defaults false so the tuning sweeps are untouched.
//
// Render with scripts/plot_accept_breakdown.m.
// ========================================================================
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <thrust/count.h>
#include "planners/KinoPaxSTARCOMBO.cuh"
#include "planners/KinoPaxSTARCleanCost.cuh"

// ---- COMBO grid ----
// MIRRORS kinopaxstar_combo_tuning_sweep.cu, including the label format, so a point here and the
// same point there carry the same name and can be read side by side. Declared locally because this
// is a standalone binary; scripts/cross_check_combo_grid.py asserts the two stay in step.
enum ComboProfile { COMBO_NONE = 0, COMBO_COV, COMBO_COL, COMBO_CST, COMBO_ALL };
static const ComboProfile PROFILES[] = {COMBO_NONE, COMBO_COV, COMBO_COL, COMBO_CST, COMBO_ALL};
static const float GAINS[]           = {0.0f, 1.0f, 4.0f, 16.0f};
static const float REACT_FRAC        = 0.1f;
static const int NUM_PROFILES        = sizeof(PROFILES) / sizeof(PROFILES[0]);
static const int NUM_GAINS           = sizeof(GAINS) / sizeof(GAINS[0]);

// ---- CleanCost reference point ----
// ONE run, at the low cap: the well-tuned operating point the COMBO grid is being compared against.
static const float CLEAN_W   = 0.9f;
static const float CLEAN_K   = 1.0f;
static const float CLEAN_CAP = 0.03f;

// COMBO_NONE owns gain 0 and nothing else; every other profile skips it. XOR, same as the sweep.
static bool comboSkip(ComboProfile p, float g)
{
    return (p == COMBO_NONE) != (g == 0.0f);
}

static const char* profileTok(ComboProfile p)
{
    switch(p)
    {
        case COMBO_COV: return "cov";
        case COMBO_COL: return "col";
        case COMBO_CST: return "cst";
        case COMBO_ALL: return "all";
        default:        return "none";
    }
}

static void profileGains(ComboProfile p, float g, float* k1, float* k2, float* k3)
{
    *k1 = 0.0f; *k2 = 0.0f; *k3 = 0.0f;
    switch(p)
    {
        case COMBO_COV: *k1 = g; break;
        case COMBO_COL: *k2 = g; break;
        case COMBO_CST: *k3 = g; break;
        case COMBO_ALL: *k1 = g; *k2 = g; *k3 = g; break;
        default: break;   // COMBO_NONE: shape == 1 identically, P = pTargetAccept uniformly
    }
}

// "KinoPaxSTARCOMBO_all_g400_rf10" -- byte-identical to the sweep's comboLabel().
static std::string comboLabel(ComboProfile p, float g, float rf)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "KinoPaxSTARCOMBO_%s_g%d_rf%d",
             profileTok(p), (int)lroundf(100.0f * g), (int)lroundf(100.0f * rf));
    return std::string(buf);
}

// "KinoPaxSTARCleanCost_r2off_w90_k100_cap3" -- same convention as the tuning sweeps.
static std::string cleanLabel(float w, float k, float cap)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "KinoPaxSTARCleanCost_r2off_w%d_k%d_cap%d",
             (int)lroundf(100.0f * w), (int)lroundf(100.0f * k), (int)lroundf(100.0f * cap));
    return std::string(buf);
}

// One schema for both planners; the fields the other one does not have are written NaN / -1, the
// same contract the tuning sweeps use so the plot script can read columns by name and skip blanks.
struct IterRow
{
    int iteration;
    unsigned long long prop_attempted;   // propagations launched, INCLUDING collisions
    unsigned long long prop_valid;       // collision-free candidates the accept kernel judged
    unsigned long long acc_min_cost;     // door 1
    unsigned long long acc_seed;         // door 2 (0 for CleanCost with r2 off; always 0 for COMBO)
    unsigned long long acc_roll;         // door 3
    unsigned long long rejected;         // prop_valid - the doors
    // credit split -- CleanCost's three terms
    double credit_syclop;
    double credit_cost;
    double credit_floor;
    // credit split -- COMBO's three sigmoids
    double credit_cov;
    double credit_col;
    double credit_cst;
    int    frontier_size;
    int    tree_size;
    float  best_cost;
    float  score_floor;
    float  cost_scale;
    // --- budget / frontier composition (COMBO fills all; CleanCost fills what it has) ---
    long long frontier_repeat_size;
    long long exempt_count;
    int    n_active;                     // regions with samples -- the frontier's hard floor
    int    reactivated;                  // Part B output: frontier bits among the existing tree
    float  mean_shape;
    float  p_target_accept;
    float  p_target_reactivate;
    float  rep_target;
    float  global_coverage;
    float  explored_mean_coverage;
    float  global_collision_frac;
};

static void blankRow(IterRow& r)
{
    r.credit_syclop = NAN; r.credit_cost = NAN; r.credit_floor = NAN;
    r.credit_cov = NAN;    r.credit_col = NAN;  r.credit_cst = NAN;
    r.frontier_repeat_size = -1;
    r.exempt_count = -1;
    r.n_active = -1;
    r.reactivated = -1;
    r.mean_shape = NAN;
    r.p_target_accept = NAN;
    r.p_target_reactivate = NAN;
    r.rep_target = NAN;
    r.global_coverage = NAN;
    r.explored_mean_coverage = NAN;
    r.global_collision_frac = NAN;
}

// ========================================================================
// Run one COMBO grid point and collect its per-iteration rows.
// ========================================================================
std::vector<IterRow> runCombo(KinoPaxSTARCOMBO& planner,
                              float kCov, float kCol, float kCst, float reactFrac,
                              float* h_initial, float* h_goal,
                              float* d_obstacles, uint numObstacles,
                              int maxIterations, int& badPartition, int& badCredit)
{
    planner.h_kCoverage_          = kCov;
    planner.h_kCollision_         = kCol;
    planner.h_kCost_              = kCst;
    planner.h_reactFrac_          = reactFrac;
    planner.h_countAcceptReasons_ = true;

    std::vector<IterRow> rows;
    planner.resetPlanner(h_initial, h_goal);

    float bestCost = INFINITY;
    int itr = 0;
    while(itr < maxIterations)
    {
        itr++;
        planner.h_itr_++;

        planner.propagateFrontier(d_obstacles, numObstacles);
        planner.graph_.updateVertices();
        int oldTreeSize = planner.h_treeSize_;   // nodes before this iteration's additions
        planner.updateFrontier();

        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
        if(planner.h_minCost_ < bestCost) bestCost = planner.h_minCost_;

        const unsigned long long* C = planner.h_acceptCounts_;
        IterRow r;
        blankRow(r);
        r.iteration      = itr;
        r.prop_attempted = planner.h_propAttempted_;
        r.prop_valid     = planner.h_candidatesPreGate_;
        r.acc_min_cost   = C[KinoPaxSTARCOMBO::ACC_MIN_COST];
        r.acc_seed       = C[KinoPaxSTARCOMBO::ACC_SEED];   // structurally 0: COMBO has no seed door
        r.acc_roll       = C[KinoPaxSTARCOMBO::ACC_ROLL];

        unsigned long long accepted = r.acc_min_cost + r.acc_seed + r.acc_roll;
        // The doors plus rejected must partition prop_valid exactly. If they do not, a candidate
        // escaped the kernel uncounted and every fraction below is meaningless.
        if(accepted > r.prop_valid) { badPartition++; r.rejected = 0; }
        else                        { r.rejected = r.prop_valid - accepted; }

        const double SC = (double)COMBO_CREDIT_SCALE;
        r.credit_cov = C[KinoPaxSTARCOMBO::ACC_CREDIT_COV] / SC;
        r.credit_col = C[KinoPaxSTARCOMBO::ACC_CREDIT_COL] / SC;
        r.credit_cst = C[KinoPaxSTARCOMBO::ACC_CREDIT_CST] / SC;

        // Each node accepted by the roll contributes exactly 1.0 of credit, so the three shares
        // must sum to acc_roll. Tolerance is one llroundf unit per node.
        double creditSum = r.credit_cov + r.credit_col + r.credit_cst;
        if(fabs(creditSum - (double)r.acc_roll) > (3.0 * (double)r.acc_roll / SC) + 1e-6) badCredit++;

        r.frontier_size = planner.h_frontierSize_;
        r.tree_size     = planner.h_treeSize_;
        r.best_cost     = bestCost;
        r.score_floor   = planner.graph_.h_scoreFloor_;
        r.cost_scale    = planner.h_costScale_;

        r.frontier_repeat_size   = (long long)planner.h_frontierRepeatSize_;
        r.exempt_count           = (long long)planner.h_exemptCount_;
        r.n_active               = planner.graph_.h_nActive_;
        r.reactivated            = (int)thrust::count(planner.d_frontier_.begin(),
                                                      planner.d_frontier_.begin() + oldTreeSize, true);
        r.mean_shape             = planner.h_meanShapePrev_;
        r.p_target_accept        = planner.h_pTargetAccept_;
        r.p_target_reactivate    = planner.h_pTargetReactivate_;
        r.rep_target             = planner.h_repTarget_;
        r.global_coverage        = planner.h_globalCoverage_;
        r.explored_mean_coverage = planner.h_exploredMeanCoverage_;
        r.global_collision_frac  = planner.h_globalCollisionFrac_;
        rows.push_back(r);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        if(planner.h_propIterations_ == 0) break;
    }
    return rows;
}

// ========================================================================
// Run the single CleanCost reference point.
// ========================================================================
std::vector<IterRow> runCleanCost(KinoPaxSTARCleanCost& planner,
                                  float w, float k, float cap,
                                  float* h_initial, float* h_goal,
                                  float* d_obstacles, uint numObstacles,
                                  int maxIterations, int& badPartition, int& badCredit)
{
    planner.h_costWeight_         = w;
    planner.h_costPruneExp_       = k;
    planner.h_acceptCapMul_       = cap;
    planner.h_countAcceptReasons_ = true;

    std::vector<IterRow> rows;
    planner.resetPlanner(h_initial, h_goal);

    float bestCost = INFINITY;
    int itr = 0;
    while(itr < maxIterations)
    {
        itr++;
        planner.h_itr_++;

        planner.propagateFrontier(d_obstacles, numObstacles);
        planner.graph_.updateVertices();
        int oldTreeSize = planner.h_treeSize_;
        planner.updateFrontier();

        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
        if(planner.h_minCost_ < bestCost) bestCost = planner.h_minCost_;

        const unsigned long long* C = planner.h_acceptCounts_;
        IterRow r;
        blankRow(r);
        r.iteration      = itr;
        r.prop_attempted = planner.h_propAttempted_;
        r.prop_valid     = planner.h_candidatesPreGate_;
        r.acc_min_cost   = C[KinoPaxSTARCleanCost::ACC_MIN_COST];
        r.acc_seed       = C[KinoPaxSTARCleanCost::ACC_SEED];
        r.acc_roll       = C[KinoPaxSTARCleanCost::ACC_ROLL];

        unsigned long long accepted = r.acc_min_cost + r.acc_seed + r.acc_roll;
        if(accepted > r.prop_valid) { badPartition++; r.rejected = 0; }
        else                        { r.rejected = r.prop_valid - accepted; }

        const double SC = (double)ACCEPT_CREDIT_SCALE;
        r.credit_syclop = C[KinoPaxSTARCleanCost::ACC_CREDIT_SYCLOP] / SC;
        r.credit_cost   = C[KinoPaxSTARCleanCost::ACC_CREDIT_COST]   / SC;
        r.credit_floor  = C[KinoPaxSTARCleanCost::ACC_CREDIT_FLOOR]  / SC;

        double creditSum = r.credit_syclop + r.credit_cost + r.credit_floor;
        if(fabs(creditSum - (double)r.acc_roll) > (3.0 * (double)r.acc_roll / SC) + 1e-6) badCredit++;

        r.frontier_size = planner.h_frontierSize_;
        r.tree_size     = planner.h_treeSize_;
        r.best_cost     = bestCost;
        r.score_floor   = planner.graph_.h_scoreFloor_;
        r.cost_scale    = planner.h_costScale_;

        // CleanCost has no growth controller, but it DOES have the two columns that explain the
        // frontier -- and comparing its frontier composition against COMBO's is the whole point of
        // running it here.
        r.frontier_repeat_size = (long long)planner.h_frontierRepeatSize_;
        r.n_active             = planner.graph_.h_nActive_;
        r.reactivated          = (int)thrust::count(planner.d_frontier_.begin(),
                                                    planner.d_frontier_.begin() + oldTreeSize, true);
        rows.push_back(r);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        if(planner.h_propIterations_ == 0) break;
    }
    return rows;
}

void writeCSV(const std::vector<IterRow>& rows, const std::string& path)
{
    std::ofstream f(path);
    // The first 15 columns are unchanged from the CleanCost-only version of this runner, and
    // everything new is APPENDED -- the plot script reads by name and tolerates a missing column,
    // so historical CSVs still load.
    f << "iteration,prop_attempted,prop_valid,acc_min_cost,acc_seed,acc_roll,rejected,"
      << "credit_syclop,credit_cost,credit_floor,"
      << "frontier_size,tree_size,best_cost,score_floor,cost_scale,"
      << "credit_cov,credit_col,credit_cst,"
      << "frontier_repeat_size,exempt_count,n_active,reactivated,"
      << "mean_shape,p_target_accept,p_target_reactivate,rep_target,"
      << "global_coverage,explored_mean_coverage,global_collision_frac\n";
    for(const auto& r : rows)
    {
        f << r.iteration << "," << r.prop_attempted << "," << r.prop_valid << ","
          << r.acc_min_cost << "," << r.acc_seed << "," << r.acc_roll << "," << r.rejected << ","
          << std::fixed << std::setprecision(6)
          << r.credit_syclop << "," << r.credit_cost << "," << r.credit_floor << ","
          << r.frontier_size << "," << r.tree_size << ","
          << std::setprecision(6) << r.best_cost << ","
          << std::setprecision(9) << r.score_floor << ","
          << std::setprecision(6) << r.cost_scale << ","
          << std::setprecision(6) << r.credit_cov << "," << r.credit_col << "," << r.credit_cst << ","
          << r.frontier_repeat_size << "," << r.exempt_count << ","
          << r.n_active << "," << r.reactivated << ","
          << std::setprecision(6) << r.mean_shape << ","
          << std::scientific << std::setprecision(6) << r.p_target_accept << ","
          << std::scientific << std::setprecision(6) << r.p_target_reactivate << ","
          << std::fixed << std::setprecision(4) << r.rep_target << ","
          << std::scientific << std::setprecision(6) << r.global_coverage << ","
          << std::fixed << std::setprecision(6) << r.explored_mean_coverage << ","
          << std::fixed << std::setprecision(6) << r.global_collision_frac << "\n";
    }
    f.close();
}

// Shared per-point reporting, so the COMBO and CleanCost arms print the same summary line.
static void reportPoint(const std::vector<IterRow>& rows, const char* creditA, const char* creditB,
                        const char* creditC, double a, double b, double c)
{
    if(rows.empty()) { printf("      (no iterations)\n"); return; }
    const IterRow& last = rows.back();
    double k1 = (last.frontier_repeat_size > 0)
                  ? (double)last.prop_attempted / (double)last.frontier_repeat_size : 0.0;
    double reactPct = (last.frontier_size > 0)
                        ? 100.0 * (double)last.reactivated / (double)last.frontier_size : 0.0;
    printf("      %d itr, tree=%d, best=%.3f | last itr: valid=%llu  min-cost=%llu  roll=%llu"
           "  rejected=%llu\n",
           (int)rows.size(), last.tree_size, last.best_cost,
           last.prop_valid, last.acc_min_cost, last.acc_roll, last.rejected);
    printf("        credit %s %.1f / %s %.1f / %s %.1f  |  prop/repeat=%.1f (32=kernel1)"
           "  nActive=%d  reactivated=%.0f%% of frontier\n",
           creditA, a, creditB, b, creditC, c, k1, last.n_active, reactPct);
}

int main(int argc, char* argv[])
{
    std::string deltaLabel   = (argc > 1) ? argv[1] : "large_length";
    std::string obstaclePath = (argc > 2) ? argv[2] : "../include/config/obstacles/house/obstacles.csv";
    std::string envName      = (argc > 3) ? argv[3] : "house";

    const int MAX_ITERATIONS = 400;

    std::string outputDir = "Data/Benchmarks/KinoPaxStarAcceptBreakdown/" + envName;
    std::filesystem::create_directories(outputDir);

    printf("=======================================================\n");
    printf("    ACCEPTANCE-REASON BREAKDOWN (COMBO + CleanCost ref)\n");
    printf("=======================================================\n");
    printf("Delta label:    %s\n", deltaLabel.c_str());
    printf("NUM_R1_REGIONS: %d\n", NUM_R1_REGIONS);
    printf("MAX_TREE_SIZE:  %d\n", MAX_TREE_SIZE);
    printf("Environment:    %s  (%s)\n", envName.c_str(), obstaclePath.c_str());
    printf("Cost metric:    %s (COST_MODE=%d)\n",
           (COST_MODE == 1) ? "control effort" : "workspace path length", COST_MODE);
    printf("Max iterations: %d   (one run per point)\n", MAX_ITERATIONS);
    printf("Reference:      CleanCost w=%.2f k=%.2f cap=%.2f (r2 off)\n", CLEAN_W, CLEAN_K, CLEAN_CAP);
    printf("=======================================================\n");

    // Model 1 [0,1]^3: (0.1,0.08,0.05) -> (0.8,0.95,0.9) -- same endpoints as the tuning sweeps.
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
    printf("Loaded %d obstacles\n\n", numObstacles);

    int badPartition = 0, badCredit = 0, points = 0;

    // --- The CleanCost reference first, so its numbers are on screen while the grid runs ---
    {
        const std::string label = cleanLabel(CLEAN_W, CLEAN_K, CLEAN_CAP);
        printf("  [reference] w=%.2f k=%.2f cap=%.2f  (%s)\n", CLEAN_W, CLEAN_K, CLEAN_CAP, label.c_str());

        KinoPaxSTARCleanCost planner;
        std::vector<IterRow> rows = runCleanCost(planner, CLEAN_W, CLEAN_K, CLEAN_CAP,
                                                 h_initial, h_goal, d_obstacles, (uint)numObstacles,
                                                 MAX_ITERATIONS, badPartition, badCredit);
        std::ostringstream fn;
        fn << outputDir << "/" << envName << "_" << label << "_delta" << deltaLabel << "_run0.csv";
        writeCSV(rows, fn.str());
        if(!rows.empty())
            reportPoint(rows, "syclop", "cost", "floor",
                        rows.back().credit_syclop, rows.back().credit_cost, rows.back().credit_floor);
        points++;
    }

    // --- The COMBO grid ---
    for(int pi = 0; pi < NUM_PROFILES; pi++)
    for(int gi = 0; gi < NUM_GAINS; gi++)
    {
        const ComboProfile prof = PROFILES[pi];
        const float        gain = GAINS[gi];
        if(comboSkip(prof, gain)) continue;

        float kCov, kCol, kCst;
        profileGains(prof, gain, &kCov, &kCol, &kCst);

        const std::string label = comboLabel(prof, gain, REACT_FRAC);
        printf("  profile=%s gain=%.2f  k=%.2f/%.2f/%.2f  (%s)\n",
               profileTok(prof), gain, kCov, kCol, kCst, label.c_str());

        KinoPaxSTARCOMBO planner;
        std::vector<IterRow> rows = runCombo(planner, kCov, kCol, kCst, REACT_FRAC,
                                             h_initial, h_goal, d_obstacles, (uint)numObstacles,
                                             MAX_ITERATIONS, badPartition, badCredit);
        std::ostringstream fn;
        fn << outputDir << "/" << envName << "_" << label << "_delta" << deltaLabel << "_run0.csv";
        writeCSV(rows, fn.str());
        if(!rows.empty())
            reportPoint(rows, "cov", "col", "cst",
                        rows.back().credit_cov, rows.back().credit_col, rows.back().credit_cst);
        points++;
    }

    cudaFree(d_obstacles);

    printf("\n=======================================================\n");
    printf("    DONE -- %d points -> %s\n", points, outputDir.c_str());
    if(badPartition || badCredit)
        printf("    *** CONSISTENCY FAILURES: partition=%d  credit=%d ***\n", badPartition, badCredit);
    else
        printf("    Consistency checks passed on every iteration of every point.\n");
    printf("=======================================================\n");
    return (badPartition || badCredit) ? 1 : 0;
}
