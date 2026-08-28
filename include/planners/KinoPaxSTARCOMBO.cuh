#pragma once
#include "planners/Planner.cuh"
#include "graphs/Graph.cuh"
#include "collisionCheck/spatialHash.cuh"

// Fixed-point scale for the diagnostic acceptance-credit counters AND for the mean-shape
// accumulator. Integer addition COMMUTES EXACTLY, which float addition over ~1e6 atomics onto one
// address does not -- the result would be both lossy and launch-order dependent. Host divides back.
//
// PREFIXED, and the prefix is load-bearing. This is a FILE-SCOPE name in a header, so it is not
// covered by the class-name rename that deriving a planner from another one otherwise handles --
// and the tuning sweep includes BOTH this header and KinoPaxSTARCleanCost.cuh (which declares its
// own ACCEPT_CREDIT_SCALE) in one translation unit. An unprefixed copy is a redefinition error in
// exactly the one .cu that matters, and nowhere else. Same discipline as the kernel prefixes,
// different mechanism: kernels collide at LINK time under CUDA_SEPARABLE_COMPILATION, header
// constants collide at COMPILE time in whichever TU pulls in both.
static const unsigned long long COMBO_CREDIT_SCALE = 1000000ULL;

class KinoPaxSTARCOMBO : public Planner
{
public:
    /**************************** CONSTRUCTORS ****************************/
    KinoPaxSTARCOMBO();
    ~KinoPaxSTARCOMBO();

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
    // No h_fAccept_. KPAX's reactivation boost was an additive term on P_syclop, and COMBO has no
    // P_syclop -- reactivation pressure is now set explicitly by h_reactFrac_ instead of by a
    // decaying nudge whose magnitude nobody controlled.
    float h_minCost_;

    // ==================================================================================
    // SHAPE TUNABLES -- TWO shapes, because one rule was answering two different questions.
    // Set by the benchmark before resetPlanner, which deliberately does not touch them.
    //
    //   ACCEPTANCE  P   = min(pMax, shape_accept * pTarget)     WHICH nodes join the tree
    //   FAN-OUT     rep = clamp(repTarget * shape_fanout, ...)  WHERE to spend propagation
    //
    // Cost belongs in acceptance -- it is what makes COMBO optimal. It is counter-productive in
    // fan-out: cost is cumulative root-to-node, so "cheap" means SHALLOW, and weighting fan-out by
    // cost pours propagation around the root. That is a density mechanism where KPAX's novelty rule
    // is a reach mechanism, and it is why COMBO grew a bigger tree than CleanCost while reaching a
    // first solution later. Splitting the shapes is what lets fan-out stay novelty-driven while
    // acceptance goes cost-driven.
    //
    // Each shape blends coverage against cost as the run progresses (see comboShape2 in
    // helper.cuh): weight moves from coverage to cost as u = treeSize/MAX_TREE_SIZE rises.
    //
    // Each k is a dimensionless GAIN in the sigmoid ARGUMENT, not an exponent, so 0 pins its term
    // at exactly 0.5 -- an ablation switch. With BOTH fan-out gains 0, shape_fanout is the constant
    // 0.5 and every node gets the same rep: the KinoPaxPlus/CleanCost control arm.
    //
    // HIGH FAN-OUT GAIN IS HOW SPARSITY HAPPENS. A shape averaging 0.5 cannot concentrate. At high
    // gain the sigmoid becomes a step, the shape goes bimodal, and because repTarget divides by the
    // MEASURED mean shape a top node gets 1/phi times the average -- 20x at phi = 0.05, with
    // sum(rep) unchanged. That is KPAX's 15/1 with an adaptive threshold, so h_kFan* is the
    // headline tuning axis and LOW gain is the failure mode.
    // ==================================================================================
    float h_kAccCoverage_;   // acceptance: prefer regions covered LESS than the explored average
    float h_kAccCost_;       // acceptance: prefer nodes cheaper than their own region's mean
    float h_kFanCoverage_;   // fan-out:    same two signals, independently tuned
    float h_kFanCost_;

    // Blend controls. mid sets WHERE the coverage->cost crossover happens (u == mid); the two g's
    // set HOW SHARPLY, independently for the two shapes. g CANNOT move the crossover -- (1-u)^g and
    // u^g are equal at u = 0.5 for every g -- which is why mid exists separately. At mid = 0.5 the
    // blend reduces to the plain (1-u), u crossfade.
    //
    // NOTE ON mid: u tops out at whatever fraction of MAX_TREE_SIZE a run actually reaches. Under a
    // wall-clock timeout that can be well under 0.5, in which case the cost term never takes
    // majority weight at mid = 0.5 -- lower mid is the lever if the sweep shows cost never engaging.
    float h_blendExpAccept_;   // g1
    float h_blendExpFanout_;   // g2
    float h_blendMid_;         // mid, in (0,1)

    // ==================================================================================
    // GROWTH-CONTROLLER TUNABLES -- HOW MANY get in. This is what replaces CleanCost's
    // h_acceptCapMul_, and the replacement is the point of this planner.
    //
    // A cap is a constant, but the acceptance probability that hits a given growth rate is NOT
    // constant: it is (want - exempt) / (candidates - exempt), and `candidates` falls as the tree
    // buffer fills and the fan-out is forced down. Over a run the required value rises ~5x, so no
    // single swept cap is right at both ends -- set it low and the tree starves late, set it high
    // and it over-accepts early and drops onto the slow kernel2 propagate path.
    //
    // Nothing here is a scale factor on a probability. Each knob either describes WHAT YOU WANT
    // (selectivity, schedule, reactivation share) or is a safety clamp; the probability itself is
    // derived every iteration in updateFrontier from measured quantities.
    // ==================================================================================

    // Candidates examined per node admitted. THE compute/quality knob: higher means each admitted
    // node is chosen from a larger pool (better nodes, slower iterations). Default 120 is the
    // measured ratio of a well-tuned CleanCost run (~9e5 collision-free candidates per ~7.5e3
    // admissions), so the default is calibration rather than a guess.
    float h_selectivity_;

    // Part-B reactivation budget as a fraction of the per-iteration growth target. The frontier is
    // then ~(1 + reactFrac) x the new-node count. This has never been an explicit knob before: in
    // CleanCost roughly 75% of the frontier is reactivation, and nobody chose that -- it fell out
    // of the Syclop score floor.
    float h_reactFrac_;

    // Growth schedule. wantThisIter = remaining / itersLeft under the default linear profile, i.e.
    // "fill MAX_TREE_SIZE in h_growthIters_ iterations". h_growthExp_ > 1 front-loads it (concave),
    // which is worth trying because propagation capacity SHRINKS as the tree buffer fills.
    int   h_growthIters_;
    float h_growthExp_;

    // ==================================================================================
    // SPARSE FAN-OUT. The favoured fraction is a knob you SET, not an outcome you discover.
    //
    // The previous proportional rule (rep = repTarget*shape) could not concentrate: its threshold
    // was the MEAN of each delta, both deltas are right-skewed, so most candidates sat on the
    // favourable side and the measured favoured fraction came out above 0.5 at every gain. The
    // "boost" reached ~70% of the frontier -- a mild broad lift, not KPAX's sparse 15/1.
    //
    // Now a threshold h_fanThreshold_ is tracked by FEEDBACK toward h_fanTopFrac_, and only nodes
    // above it are boosted. Feedback is the right tool HERE (unlike the acceptance budget, which is
    // solved algebraically) because the shape distribution cannot be inverted -- there is nothing
    // to solve, only something to track.
    // ==================================================================================
    float h_fanTopFrac_;   // phi: fraction of candidates to favour. 1.0 = everyone = uniform rep.
    float h_fanTauRate_;   // eta on the threshold update; ~0.2 gives a ~5-iteration lag.

    // Safety clamp, not tuning surface: the most blocks any one node may receive.
    float h_repeatMax_;
    float h_pMax_;

    // ==================================================================================
    // DERIVED PER-ITERATION SCALARS -- recomputed in updateFrontier, never set by a caller.
    // All are logged so the controller can be audited from the CSV rather than trusted.
    // ==================================================================================

    // Global cost scale for comboShape's d3: (mean cost over all valid samples) - (min over all
    // regions). Local reference, GLOBAL scale -- see costProbExpGlobal in helper.cuh.
    float h_costScale_;

    // The three global metrics comboShape measures each region against.
    //   h_globalCollisionFrac_  (sum counter - sum validCounter) / sum counter, over ALL regions.
    //                           Its complement is also nu, the collision-free fraction the
    //                           controller needs -- so nu costs nothing extra.
    //   h_exploredMeanCoverage_ mean of d_regionCoverage_ over EXPLORED regions only. Unexplored
    //                           regions contribute 0 to the numerator and nothing to the
    //                           denominator, so this lives on a useful scale.
    //   h_globalCoverage_       touched R2 sub-regions / NUM_R2_REGIONS. Diluted by the enormous
    //                           unexplored majority, so it is a tiny number and a genuinely
    //                           different quantity from the explored mean. COMPUTED AND LOGGED BUT
    //                           NOT YET CONSUMED -- reserved for the announced global-coverage
    //                           scaling. Numerically identical to the benchmark's
    //                           r2_coverage_pct/100 if this reduction ever shows up in profiling.
    float h_globalCollisionFrac_;
    float h_exploredMeanCoverage_;
    float h_globalCoverage_;

    // Mean shape over the PREVIOUS iteration's candidates -- one per shape, because the two are
    // now different functions with different distributions. Each budget divides by its own.
    //
    // WHY THEY EXIST. comboShape2 puts a NEUTRAL candidate at exactly COMBO_NEUTRAL_SHAPE, but the
    // deltas are asymmetric (bounded at +1 on the unfavourable side, unbounded on the favourable
    // side), so the realised mean is not that value and the bias would pass straight into the
    // growth rate. Measuring costs one atomic each; deriving would mean predicting the delta
    // distribution, which is exactly what the gains change.
    //
    // It is also the mechanism behind fan-out concentration: raising h_kFan* makes shape_fanout
    // bimodal, which DROPS this mean, which RAISES repTarget, which hands the few top nodes a large
    // multiple of the average -- at an unchanged sum(rep). Watch mean_shape_fanout fall as the gain
    // rises; that is the concentration working.
    //
    // POPULATIONS DIFFER. The accept mean is over ROLLED candidates only -- the population
    // pTargetAccept is divided across. The fan-out mean is over ALL candidates, exemptions
    // included, because every accepted node gets a fan-out shape whether it was rolled or exempt.
    float h_meanShapeAcceptPrev_;
    float h_meanShapeFanoutPrev_;

    // Blend state for this iteration, logged so the coverage->cost handover is visible in the CSV
    // rather than inferred. h_blendU_ = treeSize/MAX_TREE_SIZE; h_blendWCost_ = the normalised cost
    // weight of the ACCEPTANCE shape (the fan-out one differs only by g2).
    float h_blendU_;
    float h_blendWCost_;

    // The two budget scalars. SEPARATE ON PURPOSE -- one shape, two populations.
    // The gate judges ~1e6 candidates per iteration; Part B judges the whole tree, up to
    // MAX_TREE_SIZE. CleanCost gets away with one scalar only because its P is ~1e-4, so
    // reactivation is a trickle. At the P this planner needs, a shared scalar would reactivate
    // more nodes per iteration than the entire growth target and the frontier would run away.
    float h_pTargetAccept_;
    float h_pTargetReactivate_;

    // Threshold on shape_fanout separating the favoured minority from everyone else, and the
    // fraction actually measured above it last iteration. h_fanFracPrev_ is what sizes repHi --
    // using the MEASURED fraction rather than the target keeps the budget exact while the
    // threshold is still converging.
    float h_fanThreshold_;
    float h_fanFracPrev_;

    // Blocks available ABOVE the rep >= 1 floor, and what one favoured node therefore gets.
    //
    //   surplus = budgetBlocks - F        repHi = 1 + surplus / nFav
    //
    // so sum(rep) = nFav*repHi + (F - nFav) = F + surplus = budgetBlocks, EXACTLY. The floor is part
    // of the arithmetic instead of a clamp that silently invalidates it.
    //
    // WATCH h_surplusBlocks_. If it is small or negative, no fan-out rule can concentrate anything:
    // the rep >= 1 floor times the frontier has already spent the whole budget. F >= nActive because
    // Part B reactivates every region's best unconditionally, so that is where the constraint would
    // be, not in the fan-out rule.
    float h_surplusBlocks_;
    float h_repHi_;

    // Mean fan-out target. Derived from h_selectivity_, then clamped by the kernel1 ceiling:
    // propagateFrontier drops onto the slow kernel2 path when frontierRepeatSize * 32 exceeds the
    // remaining tree buffer, so capping repTarget at 0.8x that bound makes staying on kernel1 an
    // as long as it CAN be kept there. Not indefinitely: rep >= 1 is a correctness clamp, so
    // frontierRepeatSize >= F, and the unconditional region-best reactivation puts a floor under F
    // itself (F >= nActive). Kernel2 is therefore forced once 32*F > remaining regardless of
    // repTarget -- around 59% of the tree in the sweep config. Shrinking F further is a design
    // question (the region-best guarantee, or the R1 grid size), not a tuning one.
    float h_repTarget_;

    // ================== ACCEPTANCE-REASON INSTRUMENTATION ==================
    // A candidate enters through one of two doors here: the region-best exemption or the roll.
    // (CleanCost's third door, the R2 seeding free pass, is gone -- see the .cu header comment.)
    //
    // ACC_MIN_COST is ALWAYS counted, unlike in CleanCost: one atomicAdd on a kernel-uniform
    // branch, and the exempt count is a real diagnostic rather than an optional one. The
    // controller itself does NOT read it -- it counts exemptions exactly, this iteration, with a
    // count_if before the gate, because the previous iteration's value would lag and would carry
    // across resetPlanner (which zeroes neither h_acceptCounts_ nor d_acceptCounts_).
    //
    // The CREDIT_* slots stay gated behind h_countAcceptReasons_ (5 extra atomics per accepted
    // node onto hot addresses). ACC_SEED is retained and permanently 0 so the CSV schema and the
    // accept-breakdown tooling stay comparable across CleanCost and COMBO.
    bool h_countAcceptReasons_;

    // CREDIT_COV / CREDIT_COL / CREDIT_CST replace CleanCost's CREDIT_SYCLOP / COST / FLOOR: each
    // accepted node splits one unit of credit across the three sigmoid terms in proportion to each
    // term's share of the sum, which is RNG-independent. ACC_SHAPE_SUM is the fixed-point sum of
    // comboShape over every rolled candidate -- NOT a diagnostic, the controller consumes it.
    // ACC_CREDIT_COL is retained and permanently 0 -- the collision term is gone, but keeping the
    // slot keeps the CSV schema and the breakdown tooling comparable with earlier data.
    // ACC_SHAPE_SUM_* are NOT diagnostics: the controller consumes both.
    enum AcceptSlot { ACC_MIN_COST = 0, ACC_SEED, ACC_ROLL, ACC_CREDIT_COV,
                      ACC_CREDIT_COL, ACC_CREDIT_CST,
                      ACC_SHAPE_SUM_ACCEPT, ACC_SHAPE_SUM_FANOUT,
                      ACC_FAN_ABOVE, ACC_NUM_SLOTS };
    thrust::device_vector<unsigned long long> d_acceptCounts_;
    unsigned long long* d_acceptCounts_ptr_;
    unsigned long long  h_acceptCounts_[ACC_NUM_SLOTS];

    // Two denominators the planner already knows and used to discard. Always recorded (free).
    //   h_propAttempted_     propagation attempts this iteration, INCLUDING collisions.
    //   h_candidatesPreGate_ collision-free candidates the accept kernel judges, captured before
    //                        the post-gate re-scan overwrites h_frontierNextSize_. THE controller's
    //                        candidate count -- exact on both propagate paths, unlike any
    //                        reconstruction from h_propAttempted_ (whose formula differs by branch).
    uint h_propAttempted_;
    uint h_candidatesPreGate_;
    uint h_exemptCount_;        // min-cost exemptions this iteration, counted before the gate

    float* h_controlPathsToGoal_;

    // --- device fields (KPAX exploration) ---
    thrust::device_vector<bool> d_frontier_, d_frontierNext_;
    thrust::device_vector<uint> d_activeFrontierIdxs_, d_frontierScanIdx_, d_activeFrontierRepeatCount_,
      d_frontierRepeatScanIdx_, d_activeFrontierRepeatIdxs_;
    thrust::device_vector<int> d_unexploredSamplesParentIdxs_;
    thrust::device_vector<float> d_unexploredSamples_, d_goalSample_;

    // --- device fields (cost / region stats) ---
    // sum/cnt are now read PER CANDIDATE in the kernels, reversing CleanCost's decision to keep
    // them host-side only: comboShape's d3 needs the region mean, not just the region min.
    thrust::device_vector<float> d_minCostsR1_, d_sumCostsR1_;
    thrust::device_vector<int>   d_cntCostsR1_;
    thrust::device_vector<float> d_unexploredSampleCosts_;
    thrust::device_vector<int> d_bestNodeIdxPerR1_;
    thrust::device_vector<int> d_treeXR1s_, d_frontierNextXR1s_;
    thrust::device_vector<bool> d_goalSet_;

    // Per unexplored-sample FAN-OUT shape, carried from the accept kernel to Part A of the
    // update kernel -- a LATER LAUNCH, so it cannot be recomputed there (it would need a second
    // RNG-free evaluation and every input re-plumbed). Same cross-kernel carry pattern as
    // CleanCost's d_frontierNextFresh_, which this replaces.
    //
    // IT MUST BE THE FAN-OUT SHAPE, not the acceptance one -- Part A uses it only to size rep.
    // Indexed by UNEXPLORED-SAMPLE SLOT, not by compacted position: the accept kernel writes
    // [activeFrontierNextIdxs[tid]] and Part A must read [x1UnexploredIdx], never [tid].
    // Written before the min-cost early return too, or exempt nodes read a stale shape.
    thrust::device_vector<float> d_frontierNextFanoutShape_;

    thrust::device_vector<uint> d_goalSetIdxs_, d_goalSetScanIdx_;
    thrust::device_vector<int> d_iterations_;
    thrust::device_vector<float> d_pathCosts_, d_controlPathsToGoal_;

    // --- raw pointers ---
    float *d_unexploredSamples_ptr_, *d_goalSample_ptr_, *d_unexploredSampleCosts_ptr_;
    float *d_minCostsR1_ptr_, *d_sumCostsR1_ptr_, *d_minCost_ptr_;
    float *d_frontierNextFanoutShape_ptr_;
    int   *d_cntCostsR1_ptr_;
    float *d_pathCosts_ptr_, *d_controlPathsToGoal_ptr_;
    bool *d_frontier_ptr_, *d_frontierNext_ptr_, *d_goalSet_ptr_;
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
__global__ void KinoPaxSTARCOMBO_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid);

/***************************/
/* PROPAGATE FRONTIER KERNEL 2 */
/***************************/
__global__ void KinoPaxSTARCOMBO_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, int iterations,
                                                   float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid);

/***************************/
/* ACCEPT KERNEL — the ONLY acceptance decision */
/***************************/
// Region-best candidates are exempt; everything else is kept with min(pMax, shape * pTargetAccept).
// Every candidate's shape is recorded for the update kernel's fan-out, exemptions included.
__global__ void KinoPaxSTARCOMBO_accept_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                               float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                               float* regionCoverage,
                                               int* frontierNextXR1s, float* unexploredSampleCosts,
                                               bool* frontierNext, curandState* randomSeeds,
                                               float* frontierNextFanoutShape,
                                               float kAccCov, float kAccCst, float kFanCov, float kFanCst,
                                               float blendU, float blendExpAccept, float blendExpFanout, float blendMid,
                                               float costScale, float exploredMeanCoverage,
                                               float pTargetAccept, float pMax, float fanThreshold,
                                               bool countReasons, unsigned long long* acceptCounts);

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
__global__ void
KinoPaxSTARCOMBO_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               uint* activeFrontierRepeatCount, curandState* randomSeeds,
                               float* frontierNextFanoutShape,
                               float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                               float* regionCoverage,
                               float kAccCov, float kAccCst, float kFanCov, float kFanCst,
                               float blendU, float blendExpAccept, float blendExpFanout, float blendMid,
                               float costScale, float exploredMeanCoverage,
                               float pTargetReactivate, float pMax, float fanThreshold, float repHi,
                               int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
                               float* minCost, float* unexploredSampleCosts, bool* goalSet,
                               int* iterations, int iteration);

/***************************/
/* GET CONTROL PATH TO GOAL KERNEL */
/***************************/
__global__ void KinoPaxSTARCOMBO_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
                                                     int* treeSamplesParentIdxs, uint* goalSetIdxs, int goalSetSize,
                                                     float* pathCosts, float* treeSampleCosts, int* iterations,
                                                     float* minCost);
