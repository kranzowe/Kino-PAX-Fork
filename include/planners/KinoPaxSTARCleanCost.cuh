#pragma once
#include "planners/Planner.cuh"
#include "graphs/Graph.cuh"
#include "collisionCheck/spatialHash.cuh"

// Fixed-point scale for the fractional acceptance-credit counters. Each accepted node splits one
// unit of credit across the acceptance terms; the shares are accumulated as
// llroundf(share * ACCEPT_CREDIT_SCALE) so the atomics are exact integer adds. Host divides back.
static const unsigned long long ACCEPT_CREDIT_SCALE = 1000000ULL;

class KinoPaxSTARCleanCost : public Planner
{
public:
    /**************************** CONSTRUCTORS ****************************/
    KinoPaxSTARCleanCost();
    ~KinoPaxSTARCleanCost();

    /****************************    METHODS    ****************************/
    void plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount) override;
    float planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount);
    void planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr);
    void resetPlanner(float* h_initial, float* h_goal);
    void propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount);
    void updateFrontier();
    void getControlPathToGoal();
    void writeExecutionTimeToCSV(double time);

    /****************************    FIELDS    ****************************/
    // --- host fields ---
    Graph graph_;
    uint h_frontierSize_, h_frontierNextSize_, h_activeBlockSize_, h_frontierRepeatSize_, h_propIterations_, h_solSetSize_;
    float h_fAccept_;
    float h_minCost_;
    // --- CleanCost tunables (set in ctor, deliberately NOT reset by resetPlanner) ---
    //   P_combined = cap * min(1, w*P_syclop + (1-w)*P_cost)
    //     P_syclop = vertexScores[xR1] + fAccept   (the full KPAX rule)
    //     P_cost   = costProbExpGlobal(...)        (local reference, GLOBAL scale)
    //   w = 1 recovers KPAX's acceptance; w = 0 is pure cost-greedy.
    float h_costWeight_;      // w in [0,1]
    float h_costPruneExp_;    // k in costProbExpGlobal
    float h_probFloor_;       // P_floor. Now 0: vertexScore already carries the Graph score floor,
                              // and a second additive floor is what made acceptance cost-blind.
    // Global cost scale, RECOMPUTED EACH ITERATION -- not a tunable. dGlobal in costProbExpGlobal:
    //     h_costScale_ = (mean cost over all valid samples) - (min cost over all regions)
    // Reduced on the host in updateFrontier from the per-region arrays the propagate kernels fill.
    float h_costScale_;
    // cap in (0,1]: a flat multiplier on the FINAL acceptance probability, applied identically at
    // the admission gate and at Part-B reactivation. 1.0 = no throttle. It replaces
    // KinoPaxSTARWeightedCost's h_acceptCap_, which capped only the propagate-time roll and so sat
    // silently upstream of w.
    //
    // NOTE cap was originally derived as ~1/h_activeBlockSize_ against an acceptance probability
    // inflated by the two EPSILON floors. With P_floor removed and the Graph floor down to
    // 1/N_active, the cost-independent part drops roughly 8x, so the derived cap rises by about the
    // same factor -- cap = 1.0 may now be correct.
    float h_acceptCapMul_;

    // R2 SUB-REGION SEEDING FREE PASS, on/off. DEFAULTS FALSE -- off is now the permanent
    // condition for this planner, after measuring the two arms head to head. When true, a candidate
    // that claimed a virgin R2 sub-region is admitted unconditionally, bypassing the weighted roll
    // entirely (KPAX's main coverage drive). When false it takes the same weighted roll as
    // everything else -- the KinoPaxSTARnoseed condition (pSeed = 0) -- so the Syclop score and the
    // cost term are the only things steering admission.
    //
    // Only the ADMISSION changes: propagate still marks activeSubVertices unconditionally, so
    // r2_coverage_pct stays a valid diagnostic in both modes (same contract as the other noseed
    // variants). The freshness flag is still recorded either way.
    //
    // A bool rather than KinoPaxSTARcostprunenoseed's float h_pSeed_ -- if the annealed middle
    // ground is wanted later, widen this to a probability and roll against it.
    bool h_r2SeedAccept_;

    // ================== ACCEPTANCE-REASON INSTRUMENTATION (diagnostic only) ==================
    // A candidate enters the frontier through one of three doors -- the region-best exemption, the
    // R2 seeding exemption, or the weighted roll -- and nothing in the normal output distinguishes
    // them. These counters do, for kinopaxstar_accept_breakdown.cu.
    //
    // OFF BY DEFAULT: the tuning sweep pays only one kernel-wide uniform branch (no divergence),
    // so its behaviour and timing are unchanged.
    bool h_countAcceptReasons_;

    // Slot layout for d_acceptCounts_ / h_acceptCounts_.
    //   MIN_COST, SEED, ROLL_ACCEPTED  -- plain node counts
    //   CREDIT_*                       -- FRACTIONAL credit, fixed-point at ACCEPT_CREDIT_SCALE
    //
    // The weighted roll is a single Bernoulli draw against w*pSyclop + (1-w)*pCost + probFloor, so
    // "accepted by syclop" is not a distinction the rule makes. Each accepted node instead splits
    // one unit of credit in proportion to each term's share, which is RNG-independent.
    //
    // Fixed-point rather than float because integer addition COMMUTES EXACTLY: with ~1e6 atomics
    // onto one address a float accumulator would be both lossy and order-dependent, which is
    // useless for a diagnostic. Range is safe: 3e6 candidates * 1e6 ~ 3e12, far inside 2^64.
    enum AcceptSlot { ACC_MIN_COST = 0, ACC_SEED, ACC_ROLL, ACC_CREDIT_SYCLOP,
                      ACC_CREDIT_COST, ACC_CREDIT_FLOOR, ACC_NUM_SLOTS };
    thrust::device_vector<unsigned long long> d_acceptCounts_;
    unsigned long long* d_acceptCounts_ptr_;
    unsigned long long  h_acceptCounts_[ACC_NUM_SLOTS];   // mirrored back each iteration when on

    // Two denominators the planner already knows and used to discard. Always recorded (free).
    //   h_propAttempted_     propagation attempts this iteration, INCLUDING collisions. Set inside
    //                        propagateFrontier because only it knows which of the two launch paths
    //                        ran; re-deriving that in the benchmark would duplicate fragile logic.
    //   h_candidatesPreGate_ collision-free candidates the accept kernel judges, captured before
    //                        the post-gate re-scan overwrites h_frontierNextSize_.
    uint h_propAttempted_;
    uint h_candidatesPreGate_;

    float* h_controlPathsToGoal_;

    // --- device fields (KPAX exploration) ---
    thrust::device_vector<bool> d_frontier_, d_frontierNext_;
    thrust::device_vector<uint> d_activeFrontierIdxs_, d_frontierScanIdx_, d_activeFrontierRepeatCount_,
      d_frontierRepeatScanIdx_, d_activeFrontierRepeatIdxs_;
    thrust::device_vector<int> d_unexploredSamplesParentIdxs_;
    thrust::device_vector<float> d_unexploredSamples_, d_goalSample_;

    // --- device fields (KinoPaxPlus optimization) ---
    // No running max: neither cost probability reads one, so the atomicMaxFloat that
    // KinoPaxSTARWeightedCost ran on every propagated sample is pure overhead here.
    // sum/cnt are still accumulated per region, but they are now inputs to the HOST-side global
    // reduction that produces h_costScale_ rather than being read per candidate in the kernels.
    thrust::device_vector<float> d_minCostsR1_, d_sumCostsR1_;   // per-region cost stats
    thrust::device_vector<int>   d_cntCostsR1_;                  // per-region sample count (mean)
    thrust::device_vector<float> d_unexploredSampleCosts_;
    thrust::device_vector<int> d_bestNodeIdxPerR1_;
    thrust::device_vector<int> d_treeXR1s_, d_frontierNextXR1s_;
    thrust::device_vector<bool> d_goalSet_;
    // Per unexplored-sample flag: this thread found its R2 sub-region unoccupied. Recorded in
    // propagate (which is where the information exists -- by gate time every sub-region touched
    // this iteration is already marked active) and consumed by the accept kernel as the KPAX
    // seeding free pass. Read-then-set, so a whole launch landing in one virgin sub-region all
    // get the pass, exactly as in KPAX.
    thrust::device_vector<bool> d_frontierNextFresh_;
    thrust::device_vector<uint> d_goalSetIdxs_, d_goalSetScanIdx_;
    thrust::device_vector<int> d_iterations_;
    thrust::device_vector<float> d_pathCosts_, d_controlPathsToGoal_;

    // --- raw pointers ---
    float *d_unexploredSamples_ptr_, *d_goalSample_ptr_, *d_unexploredSampleCosts_ptr_;
    float *d_minCostsR1_ptr_, *d_sumCostsR1_ptr_, *d_minCost_ptr_;
    int   *d_cntCostsR1_ptr_;
    float *d_pathCosts_ptr_, *d_controlPathsToGoal_ptr_;
    bool *d_frontier_ptr_, *d_frontierNext_ptr_, *d_goalSet_ptr_, *d_frontierNextFresh_ptr_;
    uint *d_activeFrontierIdxs_ptr_, *d_frontierScanIdx_ptr_, *d_activeFrontierRepeatCount_ptr_,
      *d_frontierRepeatScanIdx_ptr_, *d_activeFrontierRepeatIdxs_ptr_;
    uint *d_goalSetIdxs_ptr_, *d_goalSetScanIdx_ptr_;
    int *d_unexploredSamplesParentIdxs_ptr_, *d_treeXR1s_ptr_, *d_frontierNextXR1s_ptr_;
    int *d_bestNodeIdxPerR1_ptr_, *d_iterations_ptr_;

    // --- Spatial hash grid for collision detection ---
    SpatialHashGrid* d_spatialHashGrid_;
    SpatialHashGrid h_spatialHashGrid_;
};

/**************************** DEVICE FUNCTIONS ****************************/

/***************************/
/* PROPAGATE FRONTIER KERNEL 1 */
/***************************/
// Pure candidate producer: every collision-free propagation is recorded and marked, with NO
// acceptance decision. See the accept kernel for why the decision cannot live here.
__global__ void KinoPaxSTARCleanCost_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, bool* frontierNextFresh,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid);

/***************************/
/* PROPAGATE FRONTIER KERNEL 2 */
/***************************/
__global__ void KinoPaxSTARCleanCost_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, int iterations,
                                                   float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, bool* frontierNextFresh,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid);

/***************************/
/* ACCEPT KERNEL — the ONLY acceptance decision */
/***************************/
// Region-best and fresh-sub-region candidates are exempt; everything else is kept with
// cap * min(1, w*(vertexScore + fAccept) + (1-w)*costProbExp + P_floor).
__global__ void KinoPaxSTARCleanCost_accept_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                                  float* minCostsR1,
                                                  int* frontierNextXR1s, bool* frontierNextFresh,
                                                  float* unexploredSampleCosts,
                                                  bool* frontierNext, curandState* randomSeeds,
                                                  float* vertexScores, float fAccept,
                                                  float costWeight, float costPruneExp, float probFloor,
                                                  float acceptCapMul, float costScale, bool r2SeedAccept,
                                                  bool countReasons, unsigned long long* acceptCounts);

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
__global__ void
KinoPaxSTARCleanCost_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               uint* activeFrontierRepeatCount, int* validVertexCounter, curandState* randomSeeds,
                               float* vertexScores, float fAccept,
                               float costWeight, float costPruneExp, float probFloor, float acceptCapMul,
                               float costScale,
                               float* minCostsR1,
                               int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
                               float* minCost, float* unexploredSampleCosts, bool* goalSet,
                               int* iterations, int iteration);

/***************************/
/* GET CONTROL PATH TO GOAL KERNEL */
/***************************/
__global__ void KinoPaxSTARCleanCost_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
                                                     int* treeSamplesParentIdxs, uint* goalSetIdxs, int goalSetSize,
                                                     float* pathCosts, float* treeSampleCosts, int* iterations,
                                                     float* minCost);
