// KinoPaxSTARCOMBO -- KinoPaxSTARCleanCost with the acceptance CAP replaced by a growth controller.
//
// WHAT CLEANCOST GOT RIGHT, AND IS KEPT VERBATIM. Propagate makes no decisions -- it is a pure
// candidate producer -- and exactly ONE acceptance rule runs, in _accept_kernel, after
// graph_.updateVertices(). That ordering is load-bearing and must not be relaxed:
//   - Region cost statistics are MID-FLIGHT inside propagate. minCostsR1 / sumCostsR1 / cntCostsR1
//     are updated by atomics from the very threads that would read them, so a probability computed
//     there would use a partial mean over whichever threads happened to land first -- two identical
//     candidates in the same region would draw different values purely from scheduling.
//   - vertexScores are one iteration STALE in propagate, since updateVertices() runs after it.
// Everything new in this planner therefore goes in the same place.
//
// WHAT CHANGES. Two things, and they are independent:
//
// 1. WHAT THE PROBABILITY IS A FUNCTION OF. comboShape (helper.cuh) replaces
//    weightedAccept(w, vertexScore + fAccept, costProbExpGlobal, floor). It is three sigmoids over
//    globally-normalized deltas -- region coverage vs the explored mean, region collision fraction
//    vs global, node cost vs its own region's mean -- renormalized so a neutral candidate returns
//    exactly 1.0. h_costWeight_ / h_costPruneExp_ / h_probFloor_ are gone; k1/k2/k3 replace them,
//    and each k = 0 is an exact ablation of its term.
//
//    NOTE this drops vertexScores from acceptance entirely, which also drops Syclop's
//    1/(1 + counterArray^2) -- the only thing that penalized an over-sampled region. Coverage (T1)
//    is the intended replacement, so watch h_exploredMeanCoverage_: coverage is cumulative and
//    monotone toward 1.0, and once it saturates T1 goes constant and that penalty is gone with it.
//    graph_.updateVertices() is still called, for d_regionCoverage_ and for score_floor logging.
//
// 2. HOW IT IS SCALED. h_acceptCapMul_ is gone. A cap is a constant, but the probability that hits
//    a given growth rate is not:
//
//        pTargetAccept = (wantThisIter - exempt) / ((candidates - exempt) * meanShape)
//
//    and `candidates` falls through a run as the tree buffer fills and the fan-out is forced down,
//    so the required value RISES ~5x. That is why every earlier variant needed a hand-swept cap and
//    why no single value was ever right. Here it is computed from measured quantities each
//    iteration -- feedforward and deadbeat, no gain to tune.
//
//    The fan-out is NOT solved the same way, because it is the predictive direction -- rep set now
//    determines the candidate pool next iteration. It is thresholded instead: a node is favoured if
//    its fan-out score exceeds mu + h_fanSigmaN_*sigma over the frontier's own score distribution,
//    and a favoured node gets h_repeatMax_ blocks while everyone else gets 1.
//
//    THAT DECISION LIVES IN propagateFrontier, NOT IN THE UPDATE KERNEL, and the placement is the
//    whole point. Immediately after findInd the frontier is compacted, so F is exact rather than
//    estimated and nFav can be COUNTED rather than predicted -- which means the block total
//
//        sum(rep) = F + (repHi - 1)*nFav
//
//    is known before a single block is launched, and repHi can be reduced until it fits under the
//    kernel1 condition. Sizing it in the update kernel could not do that: Part B's reactivation
//    rolls have not happened yet there, so both F and nFav were estimates, repHi pinned at
//    h_repeatMax_, and the block total came uncoupled from the budget it was solved against --
//    which is what dropped propagate onto the slow kernel2 path sporadically in early iterations.
//    Kernel2 is still forced once 32*F > remaining, since rep >= 1 is a correctness floor and the
//    region-best reactivation is unconditional (F >= nActive), but that is now the only route to it.
//
//    TWO budget scalars, not one. The gate judges ~1e6 candidates; Part B judges the whole tree.
//    CleanCost shares one scalar only because its P is ~1e-4; at the P this planner needs, a shared
//    scalar would reactivate more nodes per iteration than the entire growth target.
//
// WHAT IS REMOVED OUTRIGHT. The R2 sub-region seeding free pass -- h_r2SeedAccept_,
// d_frontierNextFresh_, and the accept kernel's second exemption. Propagate still MARKS
// activeSubVertices (r2_coverage_pct and d_regionCoverage_ both depend on it), but a virgin
// sub-region no longer buys admission. The ACC_SEED counter slot is retained and permanently 0 so
// the CSV schema stays comparable with CleanCost's.
//
// WHAT IS KEPT. The min-cost exemption (cost <= minCostsR1[xR1]) remains an unconditional free
// pass at both acceptance points: optimality convergence depends on every region's best node
// staying in the frontier. Exempt nodes carry a REAL fan-out score, computed the same way as every
// other candidate's, and compete for the boost on exactly the same threshold -- cost is what got
// them in the door, and it does not also buy them propagation.
//
// Opts into Graph's dynamic score floor (1/N_active rather than a fixed EPSILON); see Graph.cuh.
// Carries NO retroactive pruning.
#include "planners/KinoPaxSTARCOMBO.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"
#include <thrust/transform_reduce.h>

KinoPaxSTARCOMBO::KinoPaxSTARCOMBO()
{
    graph_ = Graph(W_SIZE);
    // Opt into the mean-share score floor (1/N_active) instead of the legacy fixed EPSILON, which
    // exceeds the score it floors by ~270x at 27k regions and caps the number of discriminated
    // regions at 1/EPSILON = 100 regardless of grid size. KPAX deliberately keeps the legacy floor
    // so it remains a fixed baseline.
    graph_.h_dynamicScoreFloor_ = true;

    // KPAX exploration vectors
    d_frontier_                    = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_frontierNext_                = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_activeFrontierIdxs_          = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_activeFrontierRepeatIdxs_    = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_unexploredSamples_           = thrust::device_vector<float>(MAX_TREE_SIZE * SAMPLE_DIM);
    d_unexploredSamplesParentIdxs_ = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_frontierScanIdx_             = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_frontierRepeatScanIdx_       = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_goalSample_                  = thrust::device_vector<float>(SAMPLE_DIM);
    d_activeFrontierRepeatCount_   = thrust::device_vector<uint>(MAX_TREE_SIZE);

    // KinoPaxPlus optimization vectors
    d_minCostsR1_             = thrust::device_vector<float>(NUM_R1_REGIONS);
    d_sumCostsR1_             = thrust::device_vector<float>(NUM_R1_REGIONS);
    d_cntCostsR1_             = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_bestNodeIdxPerR1_       = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_treeXR1s_               = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_frontierNextXR1s_       = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_unexploredSampleCosts_  = thrust::device_vector<float>(MAX_TREE_SIZE);
    d_goalSet_                = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_frontierNextFanoutShape_ = thrust::device_vector<float>(MAX_TREE_SIZE, COMBO_NEUTRAL_SHAPE);
    d_frontierFanoutScore_    = thrust::device_vector<float>(MAX_TREE_SIZE, COMBO_NEUTRAL_SHAPE);
    d_acceptCounts_           = thrust::device_vector<unsigned long long>(ACC_NUM_SLOTS, 0ULL);
    d_goalSetIdxs_            = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_goalSetScanIdx_         = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_iterations_             = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_pathCosts_              = thrust::device_vector<float>(MAX_TREE_SIZE * 3);
    d_controlPathsToGoal_     = thrust::device_vector<float>(MAX_ITER * SAMPLE_DIM);

    // Raw pointers
    d_frontier_ptr_                    = thrust::raw_pointer_cast(d_frontier_.data());
    d_frontierNext_ptr_                = thrust::raw_pointer_cast(d_frontierNext_.data());
    d_activeFrontierIdxs_ptr_          = thrust::raw_pointer_cast(d_activeFrontierIdxs_.data());
    d_activeFrontierRepeatIdxs_ptr_    = thrust::raw_pointer_cast(d_activeFrontierRepeatIdxs_.data());
    d_unexploredSamples_ptr_           = thrust::raw_pointer_cast(d_unexploredSamples_.data());
    d_unexploredSamplesParentIdxs_ptr_ = thrust::raw_pointer_cast(d_unexploredSamplesParentIdxs_.data());
    d_frontierScanIdx_ptr_             = thrust::raw_pointer_cast(d_frontierScanIdx_.data());
    d_frontierRepeatScanIdx_ptr_       = thrust::raw_pointer_cast(d_frontierRepeatScanIdx_.data());
    d_goalSample_ptr_                  = thrust::raw_pointer_cast(d_goalSample_.data());
    d_activeFrontierRepeatCount_ptr_   = thrust::raw_pointer_cast(d_activeFrontierRepeatCount_.data());

    d_minCostsR1_ptr_             = thrust::raw_pointer_cast(d_minCostsR1_.data());
    d_sumCostsR1_ptr_             = thrust::raw_pointer_cast(d_sumCostsR1_.data());
    d_cntCostsR1_ptr_             = thrust::raw_pointer_cast(d_cntCostsR1_.data());
    d_bestNodeIdxPerR1_ptr_       = thrust::raw_pointer_cast(d_bestNodeIdxPerR1_.data());
    d_treeXR1s_ptr_               = thrust::raw_pointer_cast(d_treeXR1s_.data());
    d_frontierNextXR1s_ptr_       = thrust::raw_pointer_cast(d_frontierNextXR1s_.data());
    d_unexploredSampleCosts_ptr_  = thrust::raw_pointer_cast(d_unexploredSampleCosts_.data());
    d_goalSet_ptr_                = thrust::raw_pointer_cast(d_goalSet_.data());
    d_goalSetIdxs_ptr_            = thrust::raw_pointer_cast(d_goalSetIdxs_.data());
    d_goalSetScanIdx_ptr_         = thrust::raw_pointer_cast(d_goalSetScanIdx_.data());
    d_frontierNextFanoutShape_ptr_ = thrust::raw_pointer_cast(d_frontierNextFanoutShape_.data());
    d_frontierFanoutScore_ptr_    = thrust::raw_pointer_cast(d_frontierFanoutScore_.data());
    d_acceptCounts_ptr_           = thrust::raw_pointer_cast(d_acceptCounts_.data());
    d_iterations_ptr_             = thrust::raw_pointer_cast(d_iterations_.data());
    d_pathCosts_ptr_              = thrust::raw_pointer_cast(d_pathCosts_.data());
    d_controlPathsToGoal_ptr_     = thrust::raw_pointer_cast(d_controlPathsToGoal_.data());

    cudaMalloc(&d_minCost_ptr_, sizeof(float));

    // Spatial hash grid for fast collision detection
    d_spatialHashGrid_ = createSpatialHashGrid();

    // Host buffer for the best (min-cost) reconstructed trajectory
    h_controlPathsToGoal_ = new float[MAX_ITER * SAMPLE_DIM];

    // ---- Shape tunables: WHICH candidates get in. 4.0 is the middle of the swept {1, 4, 16}
    // log grid -- gentle / near-binary / hard threshold on a dimensionless delta. 0 ablates. ----
    // Acceptance gains. Cost belongs here.
    h_kAccCoverage_ = 4.0f;
    h_kAccCost_     = 4.0f;
    // Fan-out gains. THE headline axis: low gain spreads propagation evenly (max/mean = 2x), high
    // gain makes the shape bimodal and hands the top fraction a large multiple of the average, at
    // an unchanged sum(rep). 0 on both = uniform rep, the CleanCost/KinoPaxPlus control arm.
    h_kFanCoverage_ = 4.0f;
    h_kFanCost_     = 4.0f;
    // Blend: g = transition sharpness (per shape), mid = where the crossover sits.
    h_blendExpAccept_ = 1.0f;
    h_blendExpFanout_ = 1.0f;
    h_blendMid_       = 0.5f;

    // ---- Growth-controller tunables: HOW MANY get in. See the header comment. ----
    // 120 is the MEASURED candidates-per-admission of a well-tuned CleanCost run (~9e5
    // collision-free candidates per ~7.5e3 admissions), so this default is a calibration.
    h_selectivity_ = 120.0f;
    // 10% of the growth target reactivated per iteration. CleanCost's realised value is ~75%, but
    // it was never chosen -- it fell out of the Syclop score floor.
    h_reactFrac_   = 0.1f;
    // "Fill MAX_TREE_SIZE in MAX_ITER iterations", linearly. h_growthExp_ > 1 front-loads it.
    h_growthIters_ = MAX_ITER;
    // Cumulative target fraction is s(u) = u^(1/growthExp); 1 is linear and reduces wantThisIter
    // exactly to remaining/(growthIters - itr). 2 gives sqrt(u) -- half the tree inside the first
    // quarter of the iterations.
    //
    // CONCAVE ON PURPOSE, aimed at time-to-first-solution, which needs tree REACH rather than tree
    // size. Early on the buffer is empty, repCeiling is generous and the frontier is tiny, so a
    // linear schedule throttles growth exactly when extending outward is cheapest. Note the first
    // few iterations are CAPACITY-limited rather than schedule-limited -- at iteration 1 there is
    // one frontier node -- so the schedule only starts to bind once candidates exceed
    // wantThisIter/pMax, and that is the window this opens up.
    h_growthExp_   = 2.0f;

    // Standard deviations above the frontier's MEAN fan-out score that a node must clear to be
    // favoured. Scale-free, which is the property the two earlier rules lacked: it picks the tail of
    // a right-skewed distribution without needing to know its shape, and the gains can change the
    // spread freely without moving where the step sits relative to it.
    //
    // 2.0 is the measured best from the first N sweep, which ran {0, 0.5, 1, 1.5, 2} and won at the
    // top edge -- so treat it as a lower bound on the optimum, not the optimum. N = 0 favours
    // everything above the mean, which is the majority for these right-skewed deltas: the failure
    // mode the first fan-out rule had.
    //
    // THERE IS A CEILING, AND kFan SETS IT. Past (maxScore - mu)/sigma nobody clears the threshold,
    // n_fav is 0 and every node falls back to one block. That quantity is logged as fan_n_max.
    h_fanSigmaN_ = 2.0f;

    // Blocks a FAVOURED node receives. NO ALIGNMENT CONSTRAINT: rep is a COUNT OF BLOCKS, not a
    // stride or a memory offset -- repeatInd writes rep integer entries and kernel1 launches one
    // 32-thread block per entry, so any positive integer is valid. A node at 15 gets 15 blocks x
    // 32 threads = 480 propagations.
    //
    // This is now the ACTUAL count, not a clamp on a derived repHi, so it is back to KPAX's 15. The
    // 32 it briefly held only ever mattered when the surplus form saturated -- and saturation was
    // precisely the bug: once repHi pinned at the cap, sum(rep) stopped tracking the budget it was
    // solved against. Nothing saturates now; the kernel1 ceiling reduces this value when it must,
    // and a logged rep_hi below 15 says so explicitly.
    h_repeatMax_   = 15.0f;
    // Per-candidate probability ceiling, applied ONLY in the kernels, to min(shape*pTarget, pMax)
    // -- again the invariant product, so 0.5 means "no candidate exceeds a 50% chance" whatever the
    // divisor is. NOT applied to pTarget on the host; see the pTargetAccept block in updateFrontier.
    h_pMax_        = 0.5f;

    // ---- Derived per-iteration scalars. All recomputed in updateFrontier; these are only the
    // iteration-1 seeds, before any propagation has happened. ----
    h_costScale_            = 0.0f;
    h_globalCollisionFrac_  = 0.1f;   // => nu = 0.9, the measured collision-free fraction
    h_exploredMeanCoverage_ = 0.0f;
    h_globalCoverage_       = 0.0f;
    h_meanShapeAcceptPrev_  = COMBO_NEUTRAL_SHAPE;   // neutral until the first batch is measured
    h_meanShapeFanoutPrev_  = COMBO_NEUTRAL_SHAPE;
    h_blendU_               = 0.0f;
    h_blendWCost_           = 0.0f;
    h_pTargetAccept_        = 0.0f;
    h_pTargetReactivate_    = 0.0f;
    // Fan-out state. Every one of these is overwritten by propagateFrontier before it is read --
    // they are measured, not carried -- so these are only the values the CSV would show if a run
    // somehow logged iteration 0.
    h_fanMu_                = COMBO_NEUTRAL_SHAPE;
    h_fanSigma_             = 0.0f;
    h_fanThreshold_         = COMBO_NEUTRAL_SHAPE;
    h_fanNMax_              = 0.0f;
    h_nFav_                 = 0;
    h_fanFrac_              = 0.0f;
    h_repHi_                = 1.0f;
    h_wantThisIter_         = 0.0f;
    h_blockCeiling_         = 0.0f;

    // Acceptance-reason CREDIT counting OFF (diagnostic only). ACC_MIN_COST / ACC_ROLL /
    // ACC_SHAPE_SUM are counted unconditionally -- ACC_SHAPE_SUM feeds the controller.
    h_countAcceptReasons_ = false;
    h_propAttempted_      = 0;
    h_candidatesPreGate_  = 0;
    h_exemptCount_        = 0;
    for(int i = 0; i < ACC_NUM_SLOTS; i++) h_acceptCounts_[i] = 0ULL;

    h_activeBlockSize_ = 32;

    if(VERBOSE)
        {
            printf("/* Planner Type: KinoPaxSTARCOMBO (Hybrid) */\n");
            printf("/* Number of R1 Vertices: %d */\n", NUM_R1_REGIONS);
            printf("/* Number of R2 Vertices: %d */\n", NUM_R2_REGIONS);
            printf("/***************************/\n");
        }
}

KinoPaxSTARCOMBO::~KinoPaxSTARCOMBO()
{
    destroySpatialHashGrid(d_spatialHashGrid_);
    delete[] h_controlPathsToGoal_;
}

void KinoPaxSTARCOMBO::resetPlanner(float* h_initial, float* h_goal)
{
    // KPAX exploration state
    thrust::fill(d_frontier_.begin(), d_frontier_.end(), false);
    thrust::fill(d_frontierNext_.begin(), d_frontierNext_.end(), false);
    thrust::fill(d_activeFrontierIdxs_.begin(), d_activeFrontierIdxs_.end(), 0);
    thrust::fill(d_unexploredSamples_.begin(), d_unexploredSamples_.end(), 0.0f);
    thrust::fill(d_unexploredSamplesParentIdxs_.begin(), d_unexploredSamplesParentIdxs_.end(), -1);
    thrust::fill(d_frontierScanIdx_.begin(), d_frontierScanIdx_.end(), 0);
    thrust::fill(d_frontierRepeatScanIdx_.begin(), d_frontierRepeatScanIdx_.end(), 0);
    thrust::fill(d_goalSample_.begin(), d_goalSample_.end(), 0.0f);
    // No root seed here any more. propagateFrontier now zeroes this array and assigns every count
    // itself, over the compacted frontier, so a seed written here would be overwritten before it
    // was ever read. The root still gets a generous first expansion: with a one-node frontier the
    // score spread is degenerate, which the assignment treats as "no minority to pick" and hands
    // every member h_repeatMax_ blocks, bounded by the block ceiling.
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);

    // Graph state
    thrust::fill(graph_.d_activeSubVertices_.begin(), graph_.d_activeSubVertices_.end(), false);
    thrust::fill(graph_.d_vertexScoreArray_.begin(), graph_.d_vertexScoreArray_.end(), 0.0f);
    thrust::fill(graph_.d_regionCoverage_.begin(), graph_.d_regionCoverage_.end(), 0.0f);
    thrust::fill(graph_.d_counterArray_.begin(), graph_.d_counterArray_.end(), 0);
    thrust::fill(graph_.d_validCounterArray_.begin(), graph_.d_validCounterArray_.end(), 0);
    graph_.h_nActive_ = 0;

    // Tree state
    thrust::fill(d_treeSamples_.begin(), d_treeSamples_.end(), 0.0f);
    thrust::fill(d_treeSamplesParentIdxs_.begin(), d_treeSamplesParentIdxs_.end(), -1);
    thrust::fill(d_treeSampleCosts_.begin(), d_treeSampleCosts_.end(), 0.0f);
    thrust::fill(d_frontier_.begin(), d_frontier_.begin() + 1, true);

    // KinoPaxPlus optimization state
    thrust::fill(d_minCostsR1_.begin(), d_minCostsR1_.end(), MAX_FLOAT);
    thrust::fill(d_sumCostsR1_.begin(), d_sumCostsR1_.end(), 0.0f);
    thrust::fill(d_cntCostsR1_.begin(), d_cntCostsR1_.end(), 0);
    thrust::fill(d_bestNodeIdxPerR1_.begin(), d_bestNodeIdxPerR1_.end(), -1);
    thrust::fill(d_treeXR1s_.begin(), d_treeXR1s_.end(), 0);
    thrust::fill(d_frontierNextXR1s_.begin(), d_frontierNextXR1s_.end(), 0);
    thrust::fill(d_unexploredSampleCosts_.begin(), d_unexploredSampleCosts_.end(), 0.0f);
    thrust::fill(d_goalSet_.begin(), d_goalSet_.end(), false);
    thrust::fill(d_frontierNextFanoutShape_.begin(), d_frontierNextFanoutShape_.end(), COMBO_NEUTRAL_SHAPE);
    thrust::fill(d_frontierFanoutScore_.begin(), d_frontierFanoutScore_.end(), COMBO_NEUTRAL_SHAPE);
    thrust::fill(d_iterations_.begin(), d_iterations_.end(), 0);
    thrust::fill(d_pathCosts_.begin(), d_pathCosts_.end(), 0.0f);
    thrust::fill(d_controlPathsToGoal_.begin(), d_controlPathsToGoal_.end(), 0.0f);

    h_treeSize_     = 1;
    h_itr_          = 0;
    h_costToGoal_   = 0;
    h_pathToGoal_   = 0;
    h_frontierSize_ = 0;
    h_minCost_      = MAX_FLOAT;
    h_solSetSize_   = 0;
    // Must be nonzero before iteration 1: the plan/benchmark loop breaks on
    // h_propIterations_ == 0, and propagateFrontier only assigns it on the tree-full path.
    // Adding the cost-stat members shifted the object layout and flipped this uninitialized
    // value to 0 (the sibling STAR planners survive only by lucky layout). COMBO adds more
    // members still, so this assignment matters here at least as much.
    h_propIterations_ = 1;

    // Controller state. CleanCost resets NONE of this -- only its constructor did -- so a planner
    // object reused across runs (which every benchmark does) carried the previous run's final
    // counts into iteration 1. The exempt count and the mean shape both feed the controller, so
    // that would bias the first iterations of every run after the first.
    thrust::fill(d_acceptCounts_.begin(), d_acceptCounts_.end(), 0ULL);
    for(int i = 0; i < ACC_NUM_SLOTS; i++) h_acceptCounts_[i] = 0ULL;
    h_propAttempted_        = 0;
    h_candidatesPreGate_    = 0;
    h_exemptCount_          = 0;
    h_frontierNextSize_     = 0;
    h_frontierRepeatSize_   = 0;
    h_costScale_            = 0.0f;
    h_globalCollisionFrac_  = 0.1f;   // => nu = 0.9 seed
    h_exploredMeanCoverage_ = 0.0f;
    h_globalCoverage_       = 0.0f;
    h_meanShapeAcceptPrev_  = COMBO_NEUTRAL_SHAPE;
    h_meanShapeFanoutPrev_  = COMBO_NEUTRAL_SHAPE;
    h_blendU_               = 0.0f;
    h_blendWCost_           = 0.0f;
    h_pTargetAccept_        = 0.0f;
    h_pTargetReactivate_    = 0.0f;
    // Fan-out state: all measured in propagateFrontier before first use, reset for the same reason
    // the counters are -- a planner object reused across runs must not carry the previous run's
    // final values into a CSV row for iteration 1.
    h_fanMu_                = COMBO_NEUTRAL_SHAPE;
    h_fanSigma_             = 0.0f;
    h_fanThreshold_         = COMBO_NEUTRAL_SHAPE;
    h_fanNMax_              = 0.0f;
    h_nFav_                 = 0;
    h_fanFrac_              = 0.0f;
    h_repHi_                = 1.0f;
    h_wantThisIter_         = 0.0f;
    h_blockCeiling_         = 0.0f;

    cudaMemcpy(d_treeSamples_ptr_, h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costToGoal_ptr_, &h_costToGoal_, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pathToGoal_ptr_, &h_pathToGoal_, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minCost_ptr_, &h_minCost_, sizeof(float), cudaMemcpyHostToDevice);

    initializeRandomSeeds(static_cast<unsigned int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
}

void KinoPaxSTARCOMBO::plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    resetPlanner(h_initial, h_goal);

    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            graph_.updateVertices();
            updateFrontier();

            // Run to MAX_ITER / tree-full, continuing to improve minCost (no first-solution break).
            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "KinoPaxSTARCOMBO execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void KinoPaxSTARCOMBO::planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr)
{
    double t_start = std::clock();
    resetPlanner(h_initial, h_goal);

    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            printf("Iteration: %d, Tree Size: %d, Frontier Size: %d\n", h_itr_, h_treeSize_, h_frontierSize_);
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            graph_.updateVertices();
            updateFrontier();

            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    double executionTime = (std::clock() - t_start) / (double)CLOCKS_PER_SEC;
    std::cout << "KinoPaxSTARCOMBO execution time: " << executionTime << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
}

float KinoPaxSTARCOMBO::planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    resetPlanner(h_initial, h_goal);

    // Run to MAX_ITER / tree-full, continuing to improve minCost after the first solution.
    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            graph_.updateVertices();
            updateFrontier();
            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "KinoPaxSTARCOMBO execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return h_minCost_;
}

// Nodes this iteration should add, under the growth schedule.
//
// growthExp == 1 is the linear schedule and reduces EXACTLY to remaining / itersLeft. Larger values
// front-load growth, which is worth trying because propagation capacity SHRINKS as the tree buffer
// fills -- the kernel1 ceiling is proportional to what is left.
//
// Expressed as "this iteration's share of the REMAINING schedule" rather than an absolute target so
// it stays self-correcting: an iteration that under-delivers raises every later iteration's demand
// a little, instead of dumping the whole shortfall onto the next one.
//
// DEFINED HERE, above propagateFrontier, because both it and updateFrontier need it now: the fan-out
// budget is sized in propagateFrontier from the CURRENT tree size rather than from a member set a
// generation earlier.
static inline float comboWantThisIter(float remaining, float itr, float growthIters, float growthExp)
{
    if(growthIters <= 0.0f) return remaining;
    float u0 = fminf(1.0f, fmaxf(0.0f, itr / growthIters));
    float u1 = fminf(1.0f, fmaxf(0.0f, (itr + 1.0f) / growthIters));
    float inv = (growthExp > 0.0f) ? (1.0f / growthExp) : 1.0f;
    float s0 = powf(u0, inv);
    float s1 = powf(u1, inv);
    float headroom = 1.0f - s0;
    if(headroom <= 1e-6f) return remaining;          // past the schedule: take what is left
    return remaining * fminf(1.0f, (s1 - s0) / headroom);
}

// Gathers a frontier node's fan-out score by TREE INDEX, for the mu / sigma reduction over the
// compacted frontier. ACCUMULATED IN DOUBLE, and that is not defensive: sigma comes out of
// E[s^2] - mu^2, the scores live in (0,1) so the two terms are nearly equal, and in float the
// cancellation destroys exactly the small-sigma regime the degenerate branch has to detect. Same
// class of bug as reducing the int counter arrays into a 32-bit accumulator.
struct KinoPaxSTARCOMBO_ScoreOf
{
    const float* score;
    __host__ __device__ double operator()(uint treeIdx) const { return (double)score[treeIdx]; }
};

struct KinoPaxSTARCOMBO_ScoreSqOf
{
    const float* score;
    __host__ __device__ double operator()(uint treeIdx) const
    {
        double s = (double)score[treeIdx];
        return s * s;
    }
};

// Mirrors repeatFromScore's comparison EXACTLY -- strict >, same threshold. If these two ever
// disagree, nFav stops being the count the block arithmetic was solved against and sum(rep) silently
// stops matching the ceiling it was fitted under.
struct KinoPaxSTARCOMBO_ScoreAbove
{
    const float* score;
    float        threshold;
    __host__ __device__ bool operator()(uint treeIdx) const { return score[treeIdx] > threshold; }
};

void KinoPaxSTARCOMBO::propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount)
{
    // --- Build spatial hash grid for fast collision detection ---
    updateSpatialHashGrid(d_spatialHashGrid_, d_obstacles_ptr, h_obstaclesCount);
    cudaMemcpy(&h_spatialHashGrid_, d_spatialHashGrid_, sizeof(SpatialHashGrid), cudaMemcpyDeviceToHost);

    // --- Find indices and size of frontier ---
    thrust::exclusive_scan(d_frontier_.begin(), d_frontier_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontier_[MAX_TREE_SIZE - 1]) ? ++h_frontierSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontier_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // ================================================================================
    // SPARSE FAN-OUT. Favour the nodes scoring above mu + N*sigma; give them h_repeatMax_ blocks
    // and everyone else 1.
    //
    // WHY IT IS HERE AND NOT IN THE UPDATE KERNEL. d_activeFrontierIdxs_[0, h_frontierSize_) is, at
    // this instant, the WHOLE proposed frontier -- Part A's admitted candidates, the min-cost
    // exemptions that bypassed the roll, Part B's unconditional region-bests, Part B's rolled
    // reactivations and any repair-arm node, all of them, already decided. Nowhere earlier is that
    // population knowable: Part B's reactivation rolls happen inside the update kernel itself, so
    // sizing fan-out there meant estimating both F and nFav, and both estimates were wrong in the
    // same direction early in a run.
    //
    // Being exact is what makes the block total controllable. sum(rep) = F + (repHi-1)*nFav is
    // computed from two counted quantities before any launch, so repHi can simply be reduced until
    // it fits under the kernel1 condition below instead of being clamped afterwards and hoping.
    //
    // THIS IS ALSO THE ONLY WRITER of d_activeFrontierRepeatCount_. It runs over exactly the
    // compacted frontier, so rep >= 1 holds for every frontier member BY CONSTRUCTION -- no node can
    // be left blockless (which would strand its frontier bit forever, since kernel1 clears the bit
    // from the expanding block) and no node outside the frontier can hold a count (which would make
    // repeatInd emit a slice no thread writes, fathering phantom-parented nodes at cost 0).
    // ================================================================================
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    if(h_frontierSize_ > 0)
        {
            // --- Score distribution over the realised frontier. ---
            KinoPaxSTARCOMBO_ScoreOf   scoreOf{d_frontierFanoutScore_ptr_};
            KinoPaxSTARCOMBO_ScoreSqOf scoreSqOf{d_frontierFanoutScore_ptr_};
            double sumS   = thrust::transform_reduce(d_activeFrontierIdxs_.begin(),
                                                     d_activeFrontierIdxs_.begin() + h_frontierSize_,
                                                     scoreOf, 0.0, thrust::plus<double>());
            double sumSq  = thrust::transform_reduce(d_activeFrontierIdxs_.begin(),
                                                     d_activeFrontierIdxs_.begin() + h_frontierSize_,
                                                     scoreSqOf, 0.0, thrust::plus<double>());
            double invF   = 1.0 / double(h_frontierSize_);
            double mu     = sumS * invF;
            double var    = sumSq * invF - mu * mu;   // clamped below; see the double note above
            h_fanMu_      = float(mu);
            h_fanSigma_   = float(sqrt(fmax(0.0, var)));

            // DEGENERATE SPREAD IS THE UNIFORM-REP CONTROL ARM, not an error case. With both kFan
            // gains at 0 every score is exactly COMBO_NEUTRAL_SHAPE, sigma is 0, and there is no
            // minority to pick -- so favour everyone and let the ceiling set the level. That is
            // precisely the KinoPaxPlus/CleanCost behaviour the gains are meant to be ablated
            // against, and it falls out of the same arithmetic: with nFav == F the repHi formula
            // below reduces to blockCeiling / F.
            //
            // A one-node frontier (iteration 1) lands here too, which is what gives the root its
            // opening expansion now that resetPlanner no longer seeds a count.
            //
            // The threshold is driven BELOW every score rather than left at mu, because the kernel
            // uses a strict > and would otherwise favour nobody while the host had counted everyone.
            bool degenerate = (h_fanSigma_ <= 1e-6f);
            if(degenerate)
                {
                    h_fanThreshold_ = -MAX_FLOAT;
                    h_nFav_         = h_frontierSize_;
                    h_fanNMax_      = 0.0f;
                }
            else
                {
                    h_fanThreshold_ = h_fanMu_ + h_fanSigmaN_ * h_fanSigma_;
                    KinoPaxSTARCOMBO_ScoreAbove above{d_frontierFanoutScore_ptr_, h_fanThreshold_};
                    h_nFav_ = (uint)thrust::count_if(d_activeFrontierIdxs_.begin(),
                                                     d_activeFrontierIdxs_.begin() + h_frontierSize_, above);

                    // HOW MUCH ROOM N HAS LEFT, in the same units N is set in: the best node on the
                    // frontier sits this many sigma above the mean, so any N above it favours
                    // NOBODY and fan-out silently collapses to a flat 1 block each.
                    //
                    // This is not hypothetical at high kFan. As the gain rises the shape goes
                    // bimodal with mass p at the top, so mu -> p and sigma -> sqrt(p(1-p)), and the
                    // largest usable N is sqrt((1-p)/p) -- 1.0 at p = 0.5, 2.0 at p = 0.2, 3.0 at
                    // p = 0.1. Raising kFan therefore TIGHTENS the ceiling on N, which is exactly
                    // the combination a sweep pushing both axes up will walk into.
                    double maxS = thrust::transform_reduce(d_activeFrontierIdxs_.begin(),
                                                           d_activeFrontierIdxs_.begin() + h_frontierSize_,
                                                           scoreOf, 0.0, thrust::maximum<double>());
                    h_fanNMax_ = float((maxS - mu) / double(h_fanSigma_));
                }
            h_fanFrac_ = float(h_nFav_) / float(h_frontierSize_);

            // --- Blocks per favoured node, bounded so kernel1 is RETAINED. ---
            // Two ceilings, taken as a min. selectivity*want/nu is the growth coupling: it spends
            // propagation in proportion to how many nodes the schedule actually wants, so an empty
            // buffer does not licence spending it all. 0.8*remaining is the margin against the
            // kernel1 condition below. h_activeBlockSize_, NOT a literal 32 -- the condition this is
            // solving uses the member, and hard-coding the value here silently desynchronises them.
            //
            // FLOAT CASTS ARE MANDATORY: h_treeSize_ is uint, so an overshoot wraps to ~4e9.
            float remaining = fmaxf(0.0f, float(MAX_TREE_SIZE) - float(h_treeSize_));
            float nu        = (h_globalCollisionFrac_ > 0.0f && h_globalCollisionFrac_ < 1.0f)
                                ? (1.0f - h_globalCollisionFrac_) : 0.9f;
            h_wantThisIter_ = comboWantThisIter(remaining, float(h_itr_), float(h_growthIters_), h_growthExp_);
            h_blockCeiling_ = fminf(h_selectivity_ * h_wantThisIter_ / nu, 0.8f * remaining)
                                / float(h_activeBlockSize_);

            // Solve sum(rep) = F + (repHi-1)*nFav <= blockCeiling for repHi. Both F and nFav are
            // counts, so this is an inequality the launch will actually satisfy, not a prediction.
            // Floors at 1 (every frontier node keeps its block even when the ceiling is already
            // spent) and caps at h_repeatMax_ (a favoured node needs no more than KPAX's 15).
            //
            // FLOOR, NOT ROUND. rep is an integer, and rounding to nearest would overshoot the
            // ceiling by up to 0.5*nFav blocks -- small, but it would make the identity above an
            // approximation again, and approximating this budget is exactly what used to flip
            // propagate onto kernel2. Truncating costs a favoured node at most one block.
            float bMax = (h_nFav_ > 0)
                           ? 1.0f + (h_blockCeiling_ - float(h_frontierSize_)) / float(h_nFav_)
                           : 1.0f;
            unsigned int repHi = (unsigned int)fmaxf(1.0f, fminf(h_repeatMax_, floorf(bMax)));
            // Store back the value actually used, so sum(rep) = F + (rep_hi-1)*n_fav is checkable
            // directly from the CSV rather than from a float the kernel never saw.
            h_repHi_ = float(repHi);

            KinoPaxSTARCOMBO_assignFanout_kernel<<<iDivUp(h_frontierSize_, h_blockSize_), h_blockSize_>>>(
              h_frontierSize_, d_activeFrontierIdxs_ptr_, d_frontierFanoutScore_ptr_,
              h_fanThreshold_, repHi, d_activeFrontierRepeatCount_ptr_);
        }

    // --- Build frontier repeat vector ---
    // Safety net: any position repeatInd does not write must not expose a stale index from an
    // earlier iteration/cycle. Seeding with 0 (the root) makes a missed slot degrade to a
    // redundant root expansion instead of fathering nodes from uninitialised tree slots. With a
    // consistent repeat count this fill is a no-op, since [0, h_frontierRepeatSize_) is fully written.
    thrust::fill(d_activeFrontierRepeatIdxs_.begin(), d_activeFrontierRepeatIdxs_.end(), 0);
    thrust::exclusive_scan(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), d_frontierRepeatScanIdx_.begin(), 0,
                           thrust::plus<uint>());
    repeatInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_activeFrontierIdxs_ptr_, d_activeFrontierRepeatCount_ptr_,
                                             d_frontierRepeatScanIdx_ptr_, d_activeFrontierRepeatIdxs_ptr_);
    h_frontierRepeatSize_ = d_frontierRepeatScanIdx_[MAX_TREE_SIZE - 1];
    (d_activeFrontierRepeatCount_[MAX_TREE_SIZE - 1]) ? h_frontierRepeatSize_ += d_activeFrontierRepeatCount_[MAX_TREE_SIZE - 1] : 0;

    // Cap the expanded frontier to the tree buffer: h_frontierRepeatSize_ (sum of the x15/x1
    // repeat weights) is otherwise unbounded and would overrun the MAX_TREE_SIZE-length repeat
    // index buffer and propagate grid near a full tree.
    if(h_frontierRepeatSize_ > (uint)MAX_TREE_SIZE) h_frontierRepeatSize_ = MAX_TREE_SIZE;

    if(h_frontierRepeatSize_ * h_activeBlockSize_ > (MAX_TREE_SIZE - h_treeSize_))
        {
            h_propIterations_ = std::min(int(float(MAX_TREE_SIZE - h_treeSize_) / float(h_frontierRepeatSize_)), int(h_activeBlockSize_));

            if(h_propIterations_ == 0)
                {
                    h_propIterations_   = 1;
                    h_frontierNextSize_ = MAX_TREE_SIZE - h_treeSize_;
                    thrust::fill(d_frontierNext_.begin(), d_frontierNext_.end(), false);
                }

            KinoPaxSTARCOMBO_propagateFrontier_kernel2<<<iDivUp(h_propIterations_ * h_frontierRepeatSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              h_propIterations_, graph_.d_minValueInRegion_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
              d_frontierNextXR1s_ptr_, d_unexploredSampleCosts_ptr_, h_spatialHashGrid_);

            // kernel2 launches h_frontierRepeatSize_ * h_propIterations_ threads, one candidate each.
            h_propAttempted_ = h_frontierRepeatSize_ * (uint)h_propIterations_;
        }
    else
        {
            KinoPaxSTARCOMBO_propagateFrontier_kernel1<<<iDivUp(h_frontierRepeatSize_ * h_activeBlockSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              graph_.d_minValueInRegion_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
              d_frontierNextXR1s_ptr_, d_unexploredSampleCosts_ptr_, h_spatialHashGrid_);

            // kernel1 launches one block of h_activeBlockSize_ threads per repeat entry.
            h_propAttempted_ = h_frontierRepeatSize_ * h_activeBlockSize_;
        }
}

/***************************/
/* FAN-OUT ASSIGNMENT KERNEL */
/***************************/
// One thread per COMPACTED FRONTIER ENTRY, so the frontier is covered exactly: every member is
// written once, nothing outside it is touched. Indexing by activeFrontierIdxs[tid] and not by tid is
// the whole reason this is a separate launch from repeatInd -- the score array is tree-indexed.
__global__ void KinoPaxSTARCOMBO_assignFanout_kernel(uint frontierSize, uint* activeFrontierIdxs,
                                                     float* frontierFanoutScore, float fanThreshold,
                                                     unsigned int repHi, uint* activeFrontierRepeatCount)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierSize) return;

    uint treeIdx = activeFrontierIdxs[tid];
    activeFrontierRepeatCount[treeIdx] = repeatFromScore(frontierFanoutScore[treeIdx], fanThreshold, repHi);
}

/***************************/
/* PROPAGATE FRONTIER KERNEL 1 */
/***************************/
// One Block Per Frontier Sample — CANDIDATE PRODUCER ONLY. No acceptance decision, no RNG draw:
// every collision-free sample is marked and its cost / region / sub-region freshness recorded, and
// the accept kernel decides once the region statistics and vertex scores have converged.
__global__ void KinoPaxSTARCOMBO_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid)
{
    if(blockIdx.x >= frontierSize) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= MAX_TREE_SIZE) return;

    // --- Load Frontier Sample Idx and cost into shared memory ---
    __shared__ int s_x0Idx;
    __shared__ float s_x0Cost;
    if(threadIdx.x == 0)
        {
            s_x0Idx           = activeFrontierIdxs[blockIdx.x];
            s_x0Cost          = treeSampleCosts[s_x0Idx];
            frontier[s_x0Idx] = false;
        }
    __syncthreads();

    // --- Load Frontier Sample into shared memory ---
    __shared__ float s_x0[SAMPLE_DIM];
    if(threadIdx.x < SAMPLE_DIM) s_x0[threadIdx.x] = treeSamples[s_x0Idx * SAMPLE_DIM + threadIdx.x];
    __syncthreads();

    // --- Propagate Sample ---
    float* x1                        = &unexploredSamples[tid * SAMPLE_DIM];
    unexploredSamplesParentIdxs[tid] = s_x0Idx;
    curandState randSeed             = randomSeeds[tid];
    bool valid                       = propagateAndCheckSpatialHash(s_x0, x1, &randSeed, spatialHashGrid, obstacles, obstaclesCount);
    int x1Vertex                     = getRegion(x1);
    int x1SubVertex                  = getSubRegion(x1, x1Vertex, minValueInRegion);

    // --- Update Graph statistics ---
    atomicAdd(&vertexCounter[x1Vertex], 1);
    if(valid)
        {
            atomicAdd(&validVertexCounter[x1Vertex], 1);

            // Cumulative cost from root
            float cost = s_x0Cost + edgeCost(s_x0, x1);

            // --- Region cost statistics. These are what the accept kernel reads once the launch
            // has finished; reading them HERE would see them mid-flight. ---
            if(minCostsR1[x1Vertex] > cost) atomicMinFloat(&minCostsR1[x1Vertex], cost);
            atomicAdd(&sumCostsR1[x1Vertex], cost);
            atomicAdd(&cntCostsR1[x1Vertex], 1);

            // --- R2 seeding: read-then-set, so an entire launch landing in one virgin sub-region
            // all record the pass. That is KPAX's behaviour and is deliberately preserved; using
            // atomicExch's return value instead would grant it to exactly one thread. ---
            // The read-then-set guard stays: it skips the atomicExch on the common
            // already-marked path. The FLAG is no longer recorded -- COMBO has no R2 seeding free
            // pass -- but the MARKING is still required, by r2_coverage_pct and, more importantly,
            // by graph_.d_regionCoverage_, which is the input to comboShape's coverage term.
            if(activeSubVertices[x1SubVertex] == 0) atomicExch(&activeSubVertices[x1SubVertex], 1);

            // --- Record the candidate. No acceptance decision, no RNG draw. ---
            unexploredSampleCosts[tid] = cost;
            frontierNextXR1s[tid]      = x1Vertex;
            frontierNext[tid]          = true;
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* PROPAGATE FRONTIER KERNEL 2 */
/***************************/
// Iterations mode — CANDIDATE PRODUCER ONLY (see kernel 1).
__global__ void KinoPaxSTARCOMBO_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, int iterations,
                                                   float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid)
{
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= MAX_TREE_SIZE) return;
    frontier[tid] = false;
    if(tid >= frontierSize * iterations) return;

    int activeFrontierIdx = tid / iterations;
    int x0Idx             = activeFrontierIdxs[activeFrontierIdx];
    float x0Cost          = treeSampleCosts[x0Idx];

    // --- Load Frontier Sample ---
    float* x0 = &treeSamples[x0Idx * SAMPLE_DIM];

    // --- Propagate Sample ---
    float* x1                        = &unexploredSamples[tid * SAMPLE_DIM];
    unexploredSamplesParentIdxs[tid] = x0Idx;
    curandState randSeed             = randomSeeds[tid];
    bool valid                       = propagateAndCheckSpatialHash(x0, x1, &randSeed, spatialHashGrid, obstacles, obstaclesCount);
    int x1Vertex                     = getRegion(x1);
    int x1SubVertex                  = getSubRegion(x1, x1Vertex, minValueInRegion);

    // --- Update Graph statistics ---
    atomicAdd(&vertexCounter[x1Vertex], 1);
    if(valid)
        {
            atomicAdd(&validVertexCounter[x1Vertex], 1);

            float cost = x0Cost + edgeCost(x0, x1);

            // --- Region cost statistics (see kernel 1). ---
            if(minCostsR1[x1Vertex] > cost) atomicMinFloat(&minCostsR1[x1Vertex], cost);
            atomicAdd(&sumCostsR1[x1Vertex], cost);
            atomicAdd(&cntCostsR1[x1Vertex], 1);

            // --- R2 seeding: read-then-set, KPAX semantics (see kernel 1). ---
            // The read-then-set guard stays: it skips the atomicExch on the common
            // already-marked path. The FLAG is no longer recorded -- COMBO has no R2 seeding free
            // pass -- but the MARKING is still required, by r2_coverage_pct and, more importantly,
            // by graph_.d_regionCoverage_, which is the input to comboShape's coverage term.
            if(activeSubVertices[x1SubVertex] == 0) atomicExch(&activeSubVertices[x1SubVertex], 1);

            // --- Record the candidate. No acceptance decision, no RNG draw. ---
            unexploredSampleCosts[tid] = cost;
            frontierNextXR1s[tid]      = x1Vertex;
            frontierNext[tid]          = true;
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* ACCEPT KERNEL - the ONLY acceptance decision */
/***************************/
// Runs after propagate has finished, so the region statistics are converged rather than mid-flight,
// and after graph_.updateVertices(), so d_regionCoverage_ includes this iteration's samples.
//
// One exemption, then one rule:
//     P = min(pMax, comboShape(...) * pTargetAccept)
//
// The shape is recorded for EVERY candidate, exemptions included, because the update kernel -- a
// later launch -- sizes each node's fan-out from it.
__global__ void KinoPaxSTARCOMBO_accept_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                               float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                               float* regionCoverage,
                                               int* frontierNextXR1s, float* unexploredSampleCosts,
                                               bool* frontierNext, curandState* randomSeeds,
                                               float* frontierNextFanoutShape,
                                               float kAccCov, float kAccCst, float kFanCov, float kFanCst,
                                               float blendU, float blendExpAccept, float blendExpFanout, float blendMid,
                                               float costScale, float exploredMeanCoverage,
                                               float pTargetAccept, float pMax,
                                               bool countReasons, unsigned long long* acceptCounts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierNextSize) return;

    int idx    = activeFrontierNextIdxs[tid];
    float cost = unexploredSampleCosts[idx];
    int xR1    = frontierNextXR1s[idx];
    float m    = minCostsR1[xR1];

    // --- Exemption: min-cost candidates are always inserted, so every region's best node stays in
    // the frontier. LOAD-BEARING for optimality convergence, and it must not be folded into the
    // formula: at cost == m the cost term is 1 but the SHAPE is still only (1 + T1 + T2)/DIVISOR, which
    // multiplied by a pTarget of ~5e-3 would reject the region best almost always. ---
    if(cost <= m)
        {
            // MUST be written before the return. Part A reads this slot to carry the node's score
            // into the tree-indexed array; a slot left unwritten holds the shape of whatever
            // candidate occupied it in a previous iteration.
            //
            // A REAL shape, not a placeholder. An exempt node's fan-out shape is computable without
            // a roll, and it must be the same function everyone else's is: the exemption is a
            // COST argument, and letting it also decide fan-out would hand propagation to the
            // shallow cheap nodes -- cost is cumulative root-to-node, so "cheap" means "near the
            // root". They compete for the boost on the same mu + N*sigma threshold as everything
            // else, and there are ~4.5e3 of them per iteration, so this matters.
            float d1e, d3e;
            int   cnte        = cntCostsR1[xR1];
            float r1MeanCoste = (cnte > 0) ? sumCostsR1[xR1] / (float)cnte : cost;
            comboDeltas(cost, r1MeanCoste, costScale, regionCoverage[xR1], exploredMeanCoverage, &d1e, &d3e);
            float shapeFanE = comboShape2(d1e, d3e, kFanCov, kFanCst, blendU, blendExpFanout, blendMid);
            frontierNextFanoutShape[idx] = shapeFanE;
            atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_SHAPE_SUM_FANOUT],
                      (unsigned long long)llroundf((float)COMBO_CREDIT_SCALE * shapeFanE));
            atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_MIN_COST], 1ULL);
            return;
        }

    // --- Cold-start guards live HERE, not in comboShape, because this is what holds the raw
    // arrays. Both collapse their delta to 0, i.e. a neutral 0.5 for that term. At iteration 1 the
    // tree is a single node and these statistics are otherwise meaningless. ---
    int   cnt        = cntCostsR1[xR1];
    float r1MeanCost = (cnt > 0) ? sumCostsR1[xR1] / (float)cnt : cost;

    // ONE pair of deltas, TWO shapes. The deltas are a property of the candidate; the shapes differ
    // only in their gains and blend exponent, which is exactly the separation this planner needs:
    // acceptance may lean on cost while fan-out stays novelty-driven.
    float d1, d3;
    comboDeltas(cost, r1MeanCost, costScale, regionCoverage[xR1], exploredMeanCoverage, &d1, &d3);
    float shapeAcc = comboShape2(d1, d3, kAccCov, kAccCst, blendU, blendExpAccept, blendMid);
    float shapeFan = comboShape2(d1, d3, kFanCov, kFanCst, blendU, blendExpFanout, blendMid);

    // The array carries the FAN-OUT shape: Part A reads it only to size rep.
    frontierNextFanoutShape[idx] = shapeFan;

    // --- Mean shape over the rolled candidates. NOT a diagnostic: updateFrontier divides both
    // budget scalars by the previous iteration's value, because the shape gates admission AND
    // fan-out, so per-node yield goes as shape^2 and E[shape] is not 1 for an asymmetric delta
    // distribution. Fixed-point so ~1e6 atomics onto one address commute exactly. Accumulated over
    // ROLLED candidates only, matching the population pTargetAccept is divided across. ---
    // Two means, two populations. ACCEPT is summed over ROLLED candidates only -- the population
    // pTargetAccept is divided across. FANOUT is summed over ALL candidates, exemptions included
    // (see the exempt branch above), because every admitted node gets a fan-out shape either way.
    atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_SHAPE_SUM_ACCEPT],
              (unsigned long long)llroundf((float)COMBO_CREDIT_SCALE * shapeAcc));
    atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_SHAPE_SUM_FANOUT],
              (unsigned long long)llroundf((float)COMBO_CREDIT_SCALE * shapeFan));
    // No ACC_FAN_ABOVE count here any more. It was the fed-back threshold's input and it measured
    // the WRONG POPULATION: pre-gate candidates, most of which the roll is about to reject, against
    // a threshold one iteration stale. The favoured count is h_nFav_, counted in propagateFrontier
    // over the realised frontier. The slot stays in the enum, permanently 0, for the CSV schema.

    float acceptanceProbability = fminf(shapeAcc * pTargetAccept, pMax);

    curandState seed = randomSeeds[idx];
    bool accept      = curand_uniform(&seed) < acceptanceProbability;
    randomSeeds[idx] = seed;

    if(accept)
        {
            atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_ROLL], 1ULL);

            // --- Diagnostic: split one unit of credit across the terms that argued for this node.
            // Shares are taken before the divisor and before pTargetAccept, since both scale the terms
            // equally and cancel in the ratio -- the credit measures WHICH TERM wanted the node,
            // independent of the throttle. Gated: 3 extra atomics on hot addresses. ---
            if(countReasons)
                {
                    // Credit is split across the two terms of the ACCEPTANCE shape, weighted the
                    // same way the shape weights them -- so it reports not just which signal liked
                    // the node but how much say that signal had at this point in the blend.
                    // ACC_CREDIT_COL stays 0: the collision term is gone, the slot is kept so the
                    // CSV schema matches earlier data.
                    float wCov, wCst;
                    comboBlendWeights(blendU, blendExpAccept, blendMid, &wCov, &wCst);
                    float cCov = (1.0f / (1.0f + __expf(kAccCov * d1))) * wCov;
                    float cCst = (1.0f / (1.0f + __expf(kAccCst * d3))) * wCst;
                    float ctot = cCov + cCst;
                    if(ctot > 0.0f)
                        {
                            const float SC = (float)COMBO_CREDIT_SCALE;
                            atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_CREDIT_COV],
                                      (unsigned long long)llroundf(SC * cCov / ctot));
                            atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_CREDIT_CST],
                                      (unsigned long long)llroundf(SC * cCst / ctot));
                        }
                }
        }

    if(!accept) frontierNext[idx] = false;
}

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// Part A adds this iteration's admitted candidates to the tree; Part B re-activates existing tree
// nodes. Both RECORD each node's fan-out score into frontierFanoutScore and decide nothing about
// block counts -- propagateFrontier does that next iteration, once the frontier this kernel is
// building has been compacted and can be measured instead of estimated.
//
// EVERY BRANCH THAT SETS frontier[i] = true MUST WRITE frontierFanoutScore[i]. There are four, and a
// missed one does not fail loudly: the node joins the mu/sigma reduction carrying whatever score the
// previous occupant of its tree slot had, so the threshold drifts and the wrong nodes fan out.
//
// The two parts run in one launch over disjoint index ranges: Part A owns [treeSize, treeSize +
// frontierNextSize) and Part B owns [0, treeSize), so they never contend for a node.
__global__ void
KinoPaxSTARCOMBO_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               curandState* randomSeeds,
                               float* frontierNextFanoutShape, float* frontierFanoutScore,
                               float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                               float* regionCoverage,
                               float kAccCov, float kAccCst, float kFanCov, float kFanCst,
                               float blendU, float blendExpAccept, float blendExpFanout, float blendMid,
                               float costScale, float exploredMeanCoverage,
                               float pTargetReactivate, float pMax,
                               int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
                               float* minCost, float* unexploredSampleCosts, bool* goalSet,
                               int* iterations, int iteration)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_xGoal[SAMPLE_DIM];
    if(threadIdx.x < SAMPLE_DIM) s_xGoal[threadIdx.x] = xGoal[threadIdx.x];
    __syncthreads();

    // --- Part A: Add new frontier nodes to tree ---
    if(tid < frontierNextSize)
        {
            int x1TreeIdx       = treeSize + tid;
            int x1UnexploredIdx = activeFrontierNextIdxs[tid];
            frontierNext[activeFrontierNextIdxs[tid]] = false;

            float* x1   = &unexploredSamples[x1UnexploredIdx * SAMPLE_DIM];
            int x0Idx   = unexploredSamplesParentIdxs[x1UnexploredIdx];
            float cost   = unexploredSampleCosts[x1UnexploredIdx];
            int xR1      = frontierNextXR1s[x1UnexploredIdx];

            // Transfer to tree
            treeSamplesParentIdxs[x1TreeIdx] = x0Idx;
            for(int i = 0; i < SAMPLE_DIM; i++)
                treeSamples[x1TreeIdx * SAMPLE_DIM + i] = x1[i];
            treeSampleCosts[x1TreeIdx] = cost;
            treeXR1s[x1TreeIdx]        = xR1;

            // Always add to frontier (it survived the gate)
            frontier[x1TreeIdx] = true;

            // Carry the fan-out score the accept kernel already computed into the TREE-INDEXED
            // array, where it survives until propagateFrontier reduces over the frontier.
            //
            // READ BY x1UnexploredIdx, NOT tid: the accept kernel writes its array by
            // unexplored-sample slot, so indexing by the compacted position would copy another
            // node's score. Every admitted node passed through the accept kernel -- the min-cost
            // exemptions write theirs before the early return -- so this slot is always fresh.
            frontierFanoutScore[x1TreeIdx] = frontierNextFanoutShape[x1UnexploredIdx];

            // Update best-node index if this is the new region best
            if(cost <= minCostsR1[xR1])
                atomicExch(&bestNodeIdxPerR1[xR1], x1TreeIdx);

            // Goal criteria check - accumulate goal nodes into goalSet; the min-cost
            // path is reconstructed afterwards by getControlPathToGoal.
            //
            // NO LONGER ORDER-SENSITIVE. This used to have to run after the repeat assignment,
            // because it cleared a count the assignment could otherwise resurrect -- leaving a node
            // with count>0 and frontier==false, which owns a slice of d_activeFrontierRepeatIdxs_
            // that no thread writes, so propagate expanded stale tree indices into phantom-parented
            // nodes at cost 0 that then won minCost. Counts are no longer written here at all:
            // clearing the frontier bit is sufficient, because propagateFrontier zeroes every count
            // and then writes only the compacted frontier, which a goal node is by definition not in.
            if(distance(x1, s_xGoal) < GOAL_THRESH && cost <= *minCost)
                {
                    atomicMinFloat(minCost, cost);
                    goalSet[x1TreeIdx]    = true;
                    frontier[x1TreeIdx]   = false;
                    iterations[x1TreeIdx] = iteration;
                }
        }

    // --- Part B: Re-activate existing tree nodes ---
    else if(tid < frontierNextSize + treeSize)
        {
            int treeIdx = tid - frontierNextSize;
            if(goalSet[treeIdx]) return;

            int xR1 = treeXR1s[treeIdx];

            // GUARANTEE: Best node per region is ALWAYS in the frontier -- unconditionally, with
            // no dice roll and no budget. This is KinoPaxPlus's invariant and the reason the
            // acceptance budget can be driven arbitrarily low without stalling cost improvement.
            if(treeIdx == bestNodeIdxPerR1[xR1])
                {
                    frontier[treeIdx] = true;
                    // Competes for the boost ON MERIT via a FRESHLY COMPUTED fan-out shape, against
                    // this iteration's region statistics -- not the score it was admitted with,
                    // which was measured against a different coverage distribution however many
                    // iterations ago. Handing it the neutral value instead would mean "never
                    // boosted", since the threshold sits in the tail well above 0.5; handing it the
                    // maximum would dwarf the propagation budget, because there is one of these per
                    // explored region.
                    float cb   = treeSampleCosts[treeIdx];
                    int   cntb = cntCostsR1[xR1];
                    float mcb  = (cntb > 0) ? sumCostsR1[xR1] / (float)cntb : cb;
                    float d1b, d3b;
                    comboDeltas(cb, mcb, costScale, regionCoverage[xR1], exploredMeanCoverage, &d1b, &d3b);
                    float shapeFanB = comboShape2(d1b, d3b, kFanCov, kFanCst, blendU, blendExpFanout, blendMid);
                    frontierFanoutScore[treeIdx] = shapeFanB;
                    return;
                }

            // REACTIVATION: the same comboShape the gate uses, against the REACTIVATION budget.
            // The shape is shared; the budget is not, because the two populations differ by orders
            // of magnitude (see the file header).
            if(frontier[treeIdx] == 0)
                {
                    float cost = treeSampleCosts[treeIdx];

                    int   cnt        = cntCostsR1[xR1];
                    float r1MeanCost = (cnt > 0) ? sumCostsR1[xR1] / (float)cnt : cost;

                    // Same split as the gate: the ROLL uses the acceptance shape, the fan-out uses
                    // the fan-out shape. Using one for both is exactly the conflation this planner
                    // exists to undo.
                    float d1, d3;
                    comboDeltas(cost, r1MeanCost, costScale, regionCoverage[xR1], exploredMeanCoverage, &d1, &d3);
                    float shapeAcc = comboShape2(d1, d3, kAccCov, kAccCst, blendU, blendExpAccept, blendMid);
                    float shapeFan = comboShape2(d1, d3, kFanCov, kFanCst, blendU, blendExpFanout, blendMid);

                    float reactivationProb = fminf(shapeAcc * pTargetReactivate, pMax);

                    curandState seed = randomSeeds[treeIdx];
                    if(curand_uniform(&seed) < reactivationProb)
                        {
                            frontier[treeIdx]            = true;
                            frontierFanoutScore[treeIdx] = shapeFan;
                        }
                    randomSeeds[treeIdx] = seed;
                }
            else
                {
                    // REPAIR ARM. It should now be unreachable: propagateFrontier assigns a count to
                    // every member of the compacted frontier, so kernel1 expands and clears every
                    // frontier bit and no node can arrive here still flagged true. It is kept
                    // because the kernel2 path clears frontier[tid] by THREAD index rather than by
                    // the node it expanded, which can still strand a bit.
                    //
                    // It writes a score rather than a block count -- a node in the frontier with no
                    // score would poison the mu/sigma reduction with whatever its tree slot last
                    // held. The score is computed the same way as the region-best branch's, so a
                    // repaired node competes for the boost on identical terms.
                    float cr   = treeSampleCosts[treeIdx];
                    int   cntr = cntCostsR1[xR1];
                    float mcr  = (cntr > 0) ? sumCostsR1[xR1] / (float)cntr : cr;
                    float d1r, d3r;
                    comboDeltas(cr, mcr, costScale, regionCoverage[xR1], exploredMeanCoverage, &d1r, &d3r);
                    frontierFanoutScore[treeIdx] =
                      comboShape2(d1r, d3r, kFanCov, kFanCst, blendU, blendExpFanout, blendMid);
                }
        }
}

// Predicate for the exact min-cost-exemption count. Mirrors the accept kernel's first branch
// EXACTLY -- unexploredSampleCosts[idx] <= minCostsR1[frontierNextXR1s[idx]] -- which is legitimate
// because nothing writes d_minCostsR1_ between propagate and the gate, so the answer computed here
// is the answer the gate will give. Counting the exemptions rather than lagging a counter matters:
// they bypass the roll entirely, so the growth budget has to be spent on them first.
struct KinoPaxSTARCOMBO_IsMinCostExempt
{
    const float* unexploredSampleCosts;
    const int*   frontierNextXR1s;
    const float* minCostsR1;

    __host__ __device__ bool operator()(uint idx) const
    {
        return unexploredSampleCosts[idx] <= minCostsR1[frontierNextXR1s[idx]];
    }
};

void KinoPaxSTARCOMBO::updateFrontier()
{
    // --- Find indices and size of the next frontier ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // Collision-free candidates the accept kernel is about to judge. Captured here because the
    // post-gate re-scan below overwrites h_frontierNextSize_ with the survivors.
    //
    // THIS is the controller's candidate count. Do NOT reconstruct it as frontierRepeatSize * 32 *
    // nu: h_propAttempted_ is set by two different formulas depending on which propagate path ran
    // (repeatSize * 32 on kernel1, repeatSize * propIterations on kernel2), so that product is a
    // no-op round trip in one branch and overstates by up to 32x in the other.
    h_candidatesPreGate_ = h_frontierNextSize_;

    // Zeroed EVERY iteration, not just when the credit diagnostic is on: ACC_MIN_COST, ACC_ROLL and
    // ACC_SHAPE_SUM are always written, and ACC_SHAPE_SUM feeds the controller.
    thrust::fill(d_acceptCounts_.begin(), d_acceptCounts_.end(), 0ULL);

    // ================================================================================
    // METRICS. All seven comboShape inputs, computed after propagate has filled the arrays and
    // after graph_.updateVertices() has refreshed d_regionCoverage_, and before the gate reads them.
    // ================================================================================

    // Global cost scale for comboShape's d3: (mean cost over all valid samples ever) minus (min
    // cost over all regions). Unreached regions contribute sum = 0, cnt = 0, min = MAX_FLOAT, so
    // all three reductions are correct with no masking.
    float sumAll = thrust::reduce(d_sumCostsR1_.begin(), d_sumCostsR1_.end(), 0.0f);
    int   cntAll = thrust::reduce(d_cntCostsR1_.begin(), d_cntCostsR1_.end(), 0);
    float minAll = thrust::reduce(d_minCostsR1_.begin(), d_minCostsR1_.end(), MAX_FLOAT, thrust::minimum<float>());
    h_costScale_ = (cntAll > 0 && minAll < MAX_FLOAT) ? (sumAll / (float)cntAll - minAll) : 0.0f;

    // Global collision fraction. REDUCED INTO 64-BIT: counterArray is int, and summed over every
    // region across a full run it reaches ~1e9 -- a 32-bit accumulator would silently overflow at
    // larger MAX_ITER / MAX_TREE_SIZE. The complement is also nu, the collision-free fraction the
    // fan-out budget needs, so nu costs nothing extra and is measured rather than assumed.
    long long totAll   = thrust::reduce(graph_.d_counterArray_.begin(), graph_.d_counterArray_.end(), (long long)0);
    long long validAll = thrust::reduce(graph_.d_validCounterArray_.begin(), graph_.d_validCounterArray_.end(), (long long)0);
    h_globalCollisionFrac_ = (totAll > 0) ? float(totAll - validAll) / float(totAll) : 0.0f;

    // Global coverage: touched R2 sub-regions over ALL of them. Diluted by the enormous unexplored
    // majority, so it stays tiny -- a genuinely different quantity from the explored mean below.
    // LOGGED BUT NOT YET CONSUMED by comboShape; reserved for global-coverage scaling.
    long long touchedR2 = thrust::reduce(graph_.d_activeSubVertices_.begin(), graph_.d_activeSubVertices_.end(), (long long)0);
    h_globalCoverage_   = float(touchedR2) / float(NUM_R2_REGIONS);

    // Mean coverage over EXPLORED regions only. Unexplored regions contribute 0 to the numerator
    // (computeVertexScores_kernel writes 0 on its inactive branch) and nothing to the denominator,
    // so this lives on a useful scale instead of being swamped. graph_.h_nActive_ is the count the
    // dynamic score floor already made -- reusing it is what keeps the two definitions of "active"
    // from drifting apart.
    float covSum = thrust::reduce(graph_.d_regionCoverage_.begin(), graph_.d_regionCoverage_.end(), 0.0f);
    h_exploredMeanCoverage_ = (graph_.h_nActive_ > 0) ? covSum / float(graph_.h_nActive_) : 0.0f;

    // ================================================================================
    // GROWTH CONTROLLER. See the file header. Feedforward and deadbeat: "I need N more nodes and I
    // have M candidates, so accept N/M of them." No gain to tune, nothing to damp.
    // ================================================================================

    // FLOAT CASTS ARE MANDATORY. h_itr_ and h_treeSize_ are uint, so an overshoot in either
    // subtraction wraps to ~4e9 instead of going negative.
    float remaining = fmaxf(0.0f, float(MAX_TREE_SIZE) - float(h_treeSize_));
    float wantThisIter =
      comboWantThisIter(remaining, float(h_itr_), float(h_growthIters_), h_growthExp_);

    // Exact exemption count for THIS iteration's candidate list.
    KinoPaxSTARCOMBO_IsMinCostExempt exemptPred{d_unexploredSampleCosts_ptr_, d_frontierNextXR1s_ptr_, d_minCostsR1_ptr_};
    h_exemptCount_ = (h_candidatesPreGate_ > 0)
                       ? (uint)thrust::count_if(d_activeFrontierIdxs_.begin(),
                                                d_activeFrontierIdxs_.begin() + h_candidatesPreGate_, exemptPred)
                       : 0u;

    // Acceptance budget. The exemptions are admitted whether the controller wants them or not, so
    // they come out of the target FIRST and only the remainder is divided across the roll.
    //
    // The numerator floors at 0, NOT at some P_MIN: if the exemptions alone already meet the growth
    // target then 0 is the correct answer, and a positive floor would admit P_MIN * candidates
    // extra nodes on top of a budget that is already satisfied.
    //
    // Dividing by meanShapePrev corrects for E[shape] != COMBO_NEUTRAL_SHAPE. comboShape puts a
    // neutral candidate exactly at COMBO_NEUTRAL_SHAPE, but the deltas are asymmetric -- bounded at
    // +1 on the unfavourable side, unbounded on the favourable side -- so the realised mean is not
    // the neutral value and the bias would otherwise pass straight into the growth rate. Dividing by
    // the MEASURED mean is also what makes the growth rate invariant to any rescaling of the shape.
    // Blend state for this iteration. u is how full the tree is, so the shapes slide from
    // coverage-driven to cost-driven as the run progresses. Computed here so both kernels and the
    // CSV see one value, and logged so the handover is visible rather than inferred.
    h_blendU_ = fminf(1.0f, fmaxf(0.0f, float(h_treeSize_) / float(MAX_TREE_SIZE)));
    {
        // Mirrors comboBlendWeights on the host purely for the logged diagnostic.
        float v = (h_blendMid_ > 0.0f && h_blendMid_ < 1.0f && fabsf(h_blendMid_ - 0.5f) > 1e-6f)
                    ? powf(h_blendU_, logf(0.5f) / logf(h_blendMid_)) : h_blendU_;
        float wc = powf(v, h_blendExpAccept_);
        float wv = powf(1.0f - v, h_blendExpAccept_);
        h_blendWCost_ = (wc + wv > 0.0f) ? wc / (wc + wv) : 0.5f;
    }

    float rolled = float(h_candidatesPreGate_) - float(h_exemptCount_);
    float shapeAdj = fmaxf(1e-3f, h_meanShapeAcceptPrev_);
    h_pTargetAccept_ = (rolled > 0.0f) ? fmaxf(0.0f, wantThisIter - float(h_exemptCount_)) / (rolled * shapeAdj) : 0.0f;
    // DELIBERATELY NOT CLAMPED HERE. pTarget is not a probability -- it is a probability PER UNIT
    // SHAPE, and the candidate's actual probability is min(shape*pTarget, pMax), enforced in the
    // accept kernel where the product is formed. Clamping pTarget against pMax on the host is a
    // units error, and it is scale-dependent: pTarget is divided by the measured mean shape, so
    // rescaling the shape rescales pTarget while pMax -- an absolute threshold -- does not, and the
    // clamp silently starts biting at a different physical acceptance rate. Leaving it
    // unclamped also makes the logged p_target_accept show the true DEMAND, so how deep into
    // saturation an iteration went is visible rather than hidden behind a flat line at pMax.

    // Reactivation budget, over ITS OWN population -- the whole tree, not the candidate list. A
    // single shared scalar is what would break here: at treeSize = 2e6 the acceptance budget applied
    // to the tree would reactivate more nodes per iteration than the entire growth target.
    // Unclamped for the same reason as pTargetAccept above: the kernel forms
    // min(shape*pTargetReactivate, pMax), which is where the probability actually gets bounded. At
    // iteration 1 this is ~1e3 because treeSize == 1; that is correct and harmless, since every
    // per-node product still clamps to pMax.
    h_pTargetReactivate_ = (h_treeSize_ > 0)
                             ? h_reactFrac_ * wantThisIter / float(h_treeSize_) / shapeAdj
                             : 0.0f;

    // --- THE acceptance decision: region-best candidates exempt, everything else kept with
    // min(pMax, comboShape(...) * pTargetAccept). ---
    // Guard the launch: iDivUp(0, block) is 0 blocks, which is cudaErrorInvalidConfiguration.
    if(h_frontierNextSize_ > 0)
        {
            KinoPaxSTARCOMBO_accept_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
              d_activeFrontierIdxs_ptr_, h_frontierNextSize_,
              d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
              graph_.d_regionCoverage_ptr_,
              d_frontierNextXR1s_ptr_, d_unexploredSampleCosts_ptr_,
              d_frontierNext_ptr_, d_randomSeeds_ptr_, d_frontierNextFanoutShape_ptr_,
              h_kAccCoverage_, h_kAccCost_, h_kFanCoverage_, h_kFanCost_,
              h_blendU_, h_blendExpAccept_, h_blendExpFanout_, h_blendMid_,
              h_costScale_, h_exploredMeanCoverage_,
              h_pTargetAccept_, h_pMax_,
              h_countAcceptReasons_, d_acceptCounts_ptr_);
        }

    cudaMemcpy(h_acceptCounts_, d_acceptCounts_ptr_, ACC_NUM_SLOTS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Mean shape over the candidates that actually took the roll -- the population pTargetAccept is
    // divided across. Consumed NEXT iteration; held at its previous value when nothing rolled, so a
    // barren iteration cannot reset the correction to a meaningless number.
    if(rolled > 0.0f)
        {
            float accSum  = float(h_acceptCounts_[ACC_SHAPE_SUM_ACCEPT]) / float(COMBO_CREDIT_SCALE);
            float meanAcc = accSum / rolled;
            if(meanAcc > 1e-3f) h_meanShapeAcceptPrev_ = meanAcc;
        }
    // The fan-out mean is over ALL candidates -- exemptions included, since they carry a real
    // fan-out shape too -- so its denominator is the full pre-gate count, not `rolled`.
    if(h_candidatesPreGate_ > 0)
        {
            float fanSum  = float(h_acceptCounts_[ACC_SHAPE_SUM_FANOUT]) / float(COMBO_CREDIT_SCALE);
            float meanFan = fanSum / float(h_candidatesPreGate_);
            // DIAGNOSTIC ONLY -- nothing divides by it. Kept because it is what diagnosed the first
            // fan-out rule's failure: the mean IS the favoured fraction once the shape goes bimodal,
            // and measuring it above 0.5 is what proved a threshold at each delta's MEAN (with both
            // deltas right-skewed) was favouring the majority rather than an elite.
            //
            // Note this is over PRE-GATE CANDIDATES, so it is not h_fanMu_ and the two should not be
            // compared: h_fanMu_ is over the realised frontier, which the roll has already filtered
            // toward high scores, so it sits above this by construction.
            if(meanFan > 1e-3f) h_meanShapeFanoutPrev_ = meanFan;
        }

    // --- Re-scan after the accept kernel ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Check tree capacity ---
    if(h_treeSize_ + h_frontierNextSize_ >= MAX_TREE_SIZE)
        {
            h_propIterations_ = 0;
            return;
        }

    // --- Update Frontier ---
    // NO FAN-OUT SIZING HERE ANY MORE, and its absence is the point. It used to live at this spot
    // and had to guess the frontier it was sizing for:
    //
    //   fNext = h_frontierNextSize_ + graph_.h_nActive_ + h_reactFrac_*wantThisIter
    //
    // -- Part A's exact count, plus an UPPER bound on the region-bests, plus a BUDGET for
    // reactivations that had not been rolled yet. nFav was likewise a prediction, taken from a
    // fraction measured over pre-gate candidates against the previous iteration's threshold. Both
    // estimates ran small early in a run, repHi pinned at h_repeatMax_, and the identity the whole
    // scheme rested on (sum(rep) = budgetBlocks) quietly stopped holding -- which is what dropped
    // propagate onto kernel2 in some early iterations and not others.
    //
    // propagateFrontier now does it after findInd, where F and nFav are both COUNTED. This kernel's
    // only fan-out responsibility is to record each node's score.
    KinoPaxSTARCOMBO_updateFrontier_kernel<<<iDivUp(h_frontierNextSize_ + h_treeSize_, h_blockSize_), h_blockSize_>>>(
      d_frontier_ptr_, d_frontierNext_ptr_, d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_goalSample_ptr_, h_treeSize_,
      d_unexploredSamples_ptr_, d_treeSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, d_randomSeeds_ptr_,
      d_frontierNextFanoutShape_ptr_, d_frontierFanoutScore_ptr_,
      d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
      graph_.d_regionCoverage_ptr_,
      h_kAccCoverage_, h_kAccCost_, h_kFanCoverage_, h_kFanCost_,
      h_blendU_, h_blendExpAccept_, h_blendExpFanout_, h_blendMid_,
      h_costScale_, h_exploredMeanCoverage_,
      h_pTargetReactivate_, h_pMax_,
      d_treeXR1s_ptr_, d_frontierNextXR1s_ptr_, d_bestNodeIdxPerR1_ptr_,
      d_minCost_ptr_, d_unexploredSampleCosts_ptr_, d_goalSet_ptr_,
      d_iterations_ptr_, h_itr_);

    // --- Sync goal state ---
    cudaMemcpy(&h_minCost_, d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);

    // --- Update Tree Size ---
    h_treeSize_ += h_frontierNextSize_;
}

/***************************/
/* GET CONTROL PATH TO GOAL */
/***************************/
void KinoPaxSTARCOMBO::getControlPathToGoal()
{
    thrust::exclusive_scan(d_goalSet_.begin(), d_goalSet_.end(), d_goalSetScanIdx_.begin(), 0, thrust::plus<uint>());
    h_solSetSize_ = d_goalSetScanIdx_[MAX_TREE_SIZE - 1];
    (d_goalSet_[MAX_TREE_SIZE - 1]) ? ++h_solSetSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_goalSet_ptr_, d_goalSetScanIdx_ptr_, d_goalSetIdxs_ptr_);

    if(h_solSetSize_ == 0) return;

    KinoPaxSTARCOMBO_getControlPathToGoal_kernel<<<iDivUp(h_solSetSize_, h_blockSize_), h_blockSize_>>>(
      d_controlPathsToGoal_ptr_, d_treeSamples_ptr_, d_treeSamplesParentIdxs_ptr_, d_goalSetIdxs_ptr_, h_solSetSize_,
      d_pathCosts_ptr_, d_treeSampleCosts_ptr_, d_iterations_ptr_, d_minCost_ptr_);

    cudaMemcpy(&h_minCost_, d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_controlPathsToGoal_, d_controlPathsToGoal_ptr_, MAX_ITER * SAMPLE_DIM * sizeof(float),
               cudaMemcpyDeviceToHost);
    printf("Cost to Goal: %f\n", h_minCost_);
}

/***************************/
/* GET CONTROL PATH TO GOAL KERNEL */
/***************************/
// Every goal thread records (idx, cost, iteration); only the min-cost goal reconstructs its full path.
__global__ void KinoPaxSTARCOMBO_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
                                                     int* treeSamplesParentIdxs, uint* goalSetIdxs, int goalSetSize,
                                                     float* pathCosts, float* treeSampleCosts, int* iterations,
                                                     float* minCost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= MAX_TREE_SIZE || tid >= goalSetSize) return;

    int goalIdx = goalSetIdxs[tid];
    int x0Idx   = goalIdx;
    float cost  = treeSampleCosts[goalIdx];

    int pathCostsIdx            = 3 * tid;
    pathCosts[pathCostsIdx]     = goalIdx;
    pathCosts[pathCostsIdx + 1] = cost;
    pathCosts[pathCostsIdx + 2] = iterations[goalIdx];

    if(cost != *minCost) return;
    int i = 0;
    // controlPathsToGoal holds MAX_ITER nodes; guard so a maximal-depth path can't write
    // one node past the buffer.
    while(x0Idx != -1 && i < MAX_ITER)
        {
            for(int j = 0; j < SAMPLE_DIM; j++)
                controlPathsToGoal[SAMPLE_DIM * i + j] = treeSamples[x0Idx * SAMPLE_DIM + j];
            i++;
            x0Idx = treeSamplesParentIdxs[x0Idx];
        }
}

void KinoPaxSTARCOMBO::writeExecutionTimeToCSV(double time)
{
    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/ExecutionTime");
    filename.str("");
    filename << "Data/ExecutionTime/executionTime.csv";
    writeValueToCSV(time, filename.str());
}
