// ========================================================================
// ACCEPTANCE-REASON BREAKDOWN  (KinoPaxSTARCleanCost)
//
// The tuning sweeps show THAT w, k and cap change outcomes, but not WHY -- the accept kernel's
// single decision hides the mechanism. A candidate enters the frontier through one of three doors:
//
//   1. the region-best exemption      (cost <= minCostsR1[r])
//   2. the R2 seeding exemption       (claimed a virgin sub-region; OFF by default)
//   3. the weighted roll              (rand < cap * (w*pSyclop + (1-w)*pCost + probFloor))
//
// and nothing in the normal per-iteration output distinguishes them. This runner turns on
// KinoPaxSTARCleanCost::h_countAcceptReasons_ and dumps the split per iteration for every point of
// the CleanCost tuning grid, one run each.
//
// ATTRIBUTION. Door 3 is a single Bernoulli draw against a weighted SUM, so "accepted by syclop"
// versus "accepted by the cost probability" is not a distinction the rule makes. Each accepted node
// therefore splits one unit of credit in proportion to each term's share of the acceptance
// probability -- RNG-independent, and the shares are taken before both weightedAccept's min(1,.)
// clamp and acceptCapMul, since those scale the terms equally and cancel in the ratio. The credit
// measures WHICH TERM ARGUED for the node, not how hard the throttle was squeezing.
//
// This is a standalone runner, not a benchmark: it drives the raw iteration loop
// (propagateFrontier -> graph_.updateVertices -> updateFrontier) the same way tree_growth_dump.cu
// does. It deliberately reports against ITERATION, never wall-clock: the counting atomics distort
// timing, which is also why h_countAcceptReasons_ defaults false so the tuning sweep is untouched.
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
#include "planners/KinoPaxSTARCleanCost.cuh"

// ---- CleanCost tuning grid ----
// MIRRORS kinopaxstar_cost_tuning_sweep.cu. Declared locally because this is a standalone binary;
// scratchpad/cross_check_grid.py asserts the two stay in step.
static const float WEIGHTS[]       = {0.9f, 0.95f, 1.0f};
static const float WEIGHTED_EXPS[] = {0.25f, 1.0f, 16.0f};
static const float CAPS[]          = {0.03f, 0.1f, 1.0f};
static const int NUM_WEIGHTS       = sizeof(WEIGHTS) / sizeof(WEIGHTS[0]);
static const int NUM_WEIGHTED_EXPS = sizeof(WEIGHTED_EXPS) / sizeof(WEIGHTED_EXPS[0]);
static const int NUM_CAPS          = sizeof(CAPS) / sizeof(CAPS[0]);

// At w = 1 the cost term vanishes from weightedAccept -- min(1, 1*P_syclop + 0*P_cost) -- so k is
// inert there and the other rungs would be the same rule differing only by RNG stream.
static bool cleanCostSkip(float w, float k)
{
    return fabsf(w - 1.0f) < 1e-6f && fabsf(k - 1.0f) > 1e-6f;
}

// "KinoPaxSTARCleanCost_r2off_w90_k25_cap3" -- same convention as the tuning sweep, so a point here
// and the same point there carry the same name.
static std::string cleanLabel(float w, float k, float cap)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "KinoPaxSTARCleanCost_r2off_w%d_k%d_cap%d",
             (int)lroundf(100.0f * w), (int)lroundf(100.0f * k), (int)lroundf(100.0f * cap));
    return std::string(buf);
}

struct IterRow
{
    int iteration;
    unsigned long long prop_attempted;   // propagations launched, INCLUDING collisions
    unsigned long long prop_valid;       // collision-free candidates the accept kernel judged
    unsigned long long acc_min_cost;     // door 1
    unsigned long long acc_seed;         // door 2 (0 while r2 seeding is off)
    unsigned long long acc_roll;         // door 3
    unsigned long long rejected;         // prop_valid - the three doors
    double credit_syclop;                // fractional, sums with the next two to acc_roll
    double credit_cost;
    double credit_floor;                 // 0 while h_probFloor_ = 0
    int    frontier_size;
    int    tree_size;
    float  best_cost;
    float  score_floor;
    float  cost_scale;
};

// ========================================================================
// Run one grid point and collect its per-iteration rows.
// ========================================================================
std::vector<IterRow> runPoint(KinoPaxSTARCleanCost& planner,
                              float w, float k, float cap,
                              float* h_initial, float* h_goal,
                              float* d_obstacles, uint numObstacles,
                              int maxIterations, int& badPartition, int& badCredit)
{
    planner.h_costWeight_        = w;
    planner.h_costPruneExp_      = k;
    planner.h_acceptCapMul_      = cap;
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
        planner.updateFrontier();

        cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
        if(planner.h_minCost_ < bestCost) bestCost = planner.h_minCost_;

        const unsigned long long* C = planner.h_acceptCounts_;
        IterRow r;
        r.iteration      = itr;
        r.prop_attempted = planner.h_propAttempted_;
        r.prop_valid     = planner.h_candidatesPreGate_;
        r.acc_min_cost   = C[KinoPaxSTARCleanCost::ACC_MIN_COST];
        r.acc_seed       = C[KinoPaxSTARCleanCost::ACC_SEED];
        r.acc_roll       = C[KinoPaxSTARCleanCost::ACC_ROLL];

        unsigned long long accepted = r.acc_min_cost + r.acc_seed + r.acc_roll;
        // The three doors plus rejected must partition prop_valid exactly. If they do not, a
        // candidate escaped the kernel uncounted and every fraction below is meaningless.
        if(accepted > r.prop_valid) { badPartition++; r.rejected = 0; }
        else                        { r.rejected = r.prop_valid - accepted; }

        const double SC = (double)ACCEPT_CREDIT_SCALE;
        r.credit_syclop = C[KinoPaxSTARCleanCost::ACC_CREDIT_SYCLOP] / SC;
        r.credit_cost   = C[KinoPaxSTARCleanCost::ACC_CREDIT_COST]   / SC;
        r.credit_floor  = C[KinoPaxSTARCleanCost::ACC_CREDIT_FLOOR]  / SC;

        // Each accepted node contributes exactly 1.0 of credit, so the three must sum to acc_roll.
        // Tolerance is one llroundf unit per node, i.e. at most acc_roll * 3 / SC.
        double creditSum = r.credit_syclop + r.credit_cost + r.credit_floor;
        if(fabs(creditSum - (double)r.acc_roll) > (3.0 * (double)r.acc_roll / SC) + 1e-6) badCredit++;

        r.frontier_size = planner.h_frontierSize_;
        r.tree_size     = planner.h_treeSize_;
        r.best_cost     = bestCost;
        r.score_floor   = planner.graph_.h_scoreFloor_;
        r.cost_scale    = planner.h_costScale_;
        rows.push_back(r);

        if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        if(planner.h_propIterations_ == 0) break;
    }
    return rows;
}

void writeCSV(const std::vector<IterRow>& rows, const std::string& path)
{
    std::ofstream f(path);
    f << "iteration,prop_attempted,prop_valid,acc_min_cost,acc_seed,acc_roll,rejected,"
      << "credit_syclop,credit_cost,credit_floor,"
      << "frontier_size,tree_size,best_cost,score_floor,cost_scale\n";
    for(const auto& r : rows)
    {
        f << r.iteration << "," << r.prop_attempted << "," << r.prop_valid << ","
          << r.acc_min_cost << "," << r.acc_seed << "," << r.acc_roll << "," << r.rejected << ","
          << std::fixed << std::setprecision(6)
          << r.credit_syclop << "," << r.credit_cost << "," << r.credit_floor << ","
          << r.frontier_size << "," << r.tree_size << ","
          << std::setprecision(6) << r.best_cost << ","
          << std::setprecision(9) << r.score_floor << ","
          << std::setprecision(6) << r.cost_scale << "\n";
    }
    f.close();
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
    printf("    ACCEPTANCE-REASON BREAKDOWN (KinoPaxSTARCleanCost)\n");
    printf("=======================================================\n");
    printf("Delta label:    %s\n", deltaLabel.c_str());
    printf("NUM_R1_REGIONS: %d\n", NUM_R1_REGIONS);
    printf("Environment:    %s  (%s)\n", envName.c_str(), obstaclePath.c_str());
    printf("Cost metric:    %s (COST_MODE=%d)\n",
           (COST_MODE == 1) ? "control effort" : "workspace path length", COST_MODE);
    printf("Max iterations: %d   (one run per grid point)\n", MAX_ITERATIONS);
    printf("=======================================================\n");

    // Model 1 [0,1]^3: (0.1,0.08,0.05) -> (0.8,0.95,0.9) -- same endpoints as the tuning sweep.
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

    for(int wi = 0; wi < NUM_WEIGHTS; wi++)
    for(int ei = 0; ei < NUM_WEIGHTED_EXPS; ei++)
    for(int ci = 0; ci < NUM_CAPS; ci++)
    {
        const float w = WEIGHTS[wi], k = WEIGHTED_EXPS[ei], cap = CAPS[ci];
        if(cleanCostSkip(w, k)) continue;

        const std::string label = cleanLabel(w, k, cap);
        printf("  w=%.2f k=%.2f cap=%.2f  (%s)\n", w, k, cap, label.c_str());

        KinoPaxSTARCleanCost planner;
        std::vector<IterRow> rows = runPoint(planner, w, k, cap, h_initial, h_goal,
                                             d_obstacles, (uint)numObstacles,
                                             MAX_ITERATIONS, badPartition, badCredit);

        std::ostringstream fn;
        fn << outputDir << "/" << envName << "_" << label << "_delta" << deltaLabel << "_run0.csv";
        writeCSV(rows, fn.str());

        const IterRow& last = rows.back();
        printf("      %d itr, tree=%d, best=%.3f | last itr: valid=%llu  min-cost=%llu  roll=%llu"
               "  rejected=%llu  (syclop %.1f / cost %.1f)\n",
               (int)rows.size(), last.tree_size, last.best_cost,
               last.prop_valid, last.acc_min_cost, last.acc_roll, last.rejected,
               last.credit_syclop, last.credit_cost);
        points++;
    }

    cudaFree(d_obstacles);

    printf("\n=======================================================\n");
    printf("    DONE -- %d grid points -> %s\n", points, outputDir.c_str());
    if(badPartition || badCredit)
        printf("    *** CONSISTENCY FAILURES: partition=%d  credit=%d ***\n", badPartition, badCredit);
    else
        printf("    Consistency checks passed on every iteration of every point.\n");
    printf("=======================================================\n");
    return (badPartition || badCredit) ? 1 : 0;
}
