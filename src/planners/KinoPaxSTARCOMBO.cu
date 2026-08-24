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
//    The fan-out follows the same logic: repTarget comes from h_selectivity_ (candidates examined
//    per node kept) and is then clamped by the kernel1 ceiling, which makes "never drop onto the
//    slow kernel2 propagate path" an INVARIANT rather than a tuning outcome.
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
// staying in the frontier. Exempt nodes get the NEUTRAL shape for fan-out, not the maximum --
// ~4.5e3 exemptions at repeatMax would be the entire propagation budget on its own.
//
// Opts into Graph's dynamic score floor (1/N_active rather than a fixed EPSILON); see Graph.cuh.
// Carries NO retroactive pruning.
#include "planners/KinoPaxSTARCOMBO.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"

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
    d_frontierNextAcceptShape_ = thrust::device_vector<float>(MAX_TREE_SIZE, 1.0f);
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
    d_frontierNextAcceptShape_ptr_ = thrust::raw_pointer_cast(d_frontierNextAcceptShape_.data());
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
    h_kCoverage_  = 4.0f;
    h_kCollision_ = 4.0f;
    h_kCost_      = 4.0f;

    // ---- Growth-controller tunables: HOW MANY get in. See the header comment. ----
    // 120 is the MEASURED candidates-per-admission of a well-tuned CleanCost run (~9e5
    // collision-free candidates per ~7.5e3 admissions), so this default is a calibration.
    h_selectivity_ = 120.0f;
    // 10% of the growth target reactivated per iteration. CleanCost's realised value is ~75%, but
    // it was never chosen -- it fell out of the Syclop score floor.
    h_reactFrac_   = 0.1f;
    // "Fill MAX_TREE_SIZE in MAX_ITER iterations", linearly. h_growthExp_ > 1 front-loads it.
    h_growthIters_ = MAX_ITER;
    h_growthExp_   = 1.0f;
    // Safety clamps. repeatMax matches the legacy binary rule's 15.
    h_repeatMax_   = 15.0f;
    h_pMax_        = 0.5f;

    // ---- Derived per-iteration scalars. All recomputed in updateFrontier; these are only the
    // iteration-1 seeds, before any propagation has happened. ----
    h_costScale_            = 0.0f;
    h_globalCollisionFrac_  = 0.1f;   // => nu = 0.9, the measured collision-free fraction
    h_exploredMeanCoverage_ = 0.0f;
    h_globalCoverage_       = 0.0f;
    h_meanShapePrev_        = 1.0f;   // neutral until the first batch has been measured
    h_pTargetAccept_        = 0.0f;
    h_pTargetReactivate_    = 0.0f;
    h_repTarget_            = 1.0f;

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
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.begin() + 1, 5);

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
    thrust::fill(d_frontierNextAcceptShape_.begin(), d_frontierNextAcceptShape_.end(), 1.0f);
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
    h_meanShapePrev_        = 1.0f;
    h_pTargetAccept_        = 0.0f;
    h_pTargetReactivate_    = 0.0f;
    h_repTarget_            = 1.0f;

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
                                               int* counterArray, int* validCounterArray, float* regionCoverage,
                                               int* frontierNextXR1s, float* unexploredSampleCosts,
                                               bool* frontierNext, curandState* randomSeeds,
                                               float* frontierNextAcceptShape,
                                               float kCoverage, float kCollision, float kCost,
                                               float costScale, float exploredMeanCoverage, float globalCollisionFrac,
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
    // formula: at cost == m the cost term is 1 but the SHAPE is still only (1 + T1 + T2)/1.5, which
    // multiplied by a pTarget of ~5e-3 would reject the region best almost always. ---
    if(cost <= m)
        {
            // MUST be written before the return. Part A reads this slot to size the node's
            // fan-out; a slot left unwritten holds the shape of whatever candidate occupied it in a
            // previous iteration. NEUTRAL rather than maximum on purpose: exemptions run to ~4.5e3
            // per iteration, and 4.5e3 nodes at repeatMax would be the whole propagation budget.
            frontierNextAcceptShape[idx] = 1.0f;
            atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_MIN_COST], 1ULL);
            return;
        }

    // --- Cold-start guards live HERE, not in comboShape, because this is what holds the raw
    // arrays. Both collapse their delta to 0, i.e. a neutral 0.5 for that term. At iteration 1 the
    // tree is a single node and these statistics are otherwise meaningless. ---
    int   cnt        = cntCostsR1[xR1];
    float r1MeanCost = (cnt > 0) ? sumCostsR1[xR1] / (float)cnt : cost;
    int   tot        = counterArray[xR1];
    float r1CollFrac = (tot > 0) ? (float)(tot - validCounterArray[xR1]) / (float)tot : globalCollisionFrac;

    float t1, t2, t3;
    comboTerms(cost, r1MeanCost, costScale, regionCoverage[xR1], exploredMeanCoverage,
               r1CollFrac, globalCollisionFrac, kCoverage, kCollision, kCost, &t1, &t2, &t3);
    float shape = (t1 + t2 + t3) * (1.0f / 1.5f);

    frontierNextAcceptShape[idx] = shape;

    // --- Mean shape over the rolled candidates. NOT a diagnostic: updateFrontier divides both
    // budget scalars by the previous iteration's value, because the shape gates admission AND
    // fan-out, so per-node yield goes as shape^2 and E[shape] is not 1 for an asymmetric delta
    // distribution. Fixed-point so ~1e6 atomics onto one address commute exactly. Accumulated over
    // ROLLED candidates only, matching the population pTargetAccept is divided across. ---
    atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_SHAPE_SUM],
              (unsigned long long)llroundf((float)COMBO_CREDIT_SCALE * shape));

    float acceptanceProbability = fminf(shape * pTargetAccept, pMax);

    curandState seed = randomSeeds[idx];
    bool accept      = curand_uniform(&seed) < acceptanceProbability;
    randomSeeds[idx] = seed;

    if(accept)
        {
            atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_ROLL], 1ULL);

            // --- Diagnostic: split one unit of credit across the terms that argued for this node.
            // Shares are taken before the /1.5 and before pTargetAccept, since both scale the terms
            // equally and cancel in the ratio -- the credit measures WHICH TERM wanted the node,
            // independent of the throttle. Gated: 3 extra atomics on hot addresses. ---
            if(countReasons)
                {
                    float tot3 = t1 + t2 + t3;
                    if(tot3 > 0.0f)
                        {
                            const float SC = (float)COMBO_CREDIT_SCALE;
                            atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_CREDIT_COV],
                                      (unsigned long long)llroundf(SC * t1 / tot3));
                            atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_CREDIT_COL],
                                      (unsigned long long)llroundf(SC * t2 / tot3));
                            atomicAdd(&acceptCounts[KinoPaxSTARCOMBO::ACC_CREDIT_CST],
                                      (unsigned long long)llroundf(SC * t3 / tot3));
                        }
                }
        }

    if(!accept) frontierNext[idx] = false;
}

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// Part A adds this iteration's admitted candidates to the tree; Part B re-activates existing tree
// nodes. Both size their fan-out from the SAME comboShape the gate used, via repeatFromShape --
// replacing KPAX's binary "validVertexCounter < 10 ? 15 : 1" at Part A and a flat 1 at Part B.
//
// The two parts run in one launch over disjoint index ranges: Part A owns [treeSize, treeSize +
// frontierNextSize) and Part B owns [0, treeSize), so they never contend for a node.
__global__ void
KinoPaxSTARCOMBO_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               uint* activeFrontierRepeatCount, curandState* randomSeeds,
                               float* frontierNextAcceptShape,
                               float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                               int* counterArray, int* validCounterArray, float* regionCoverage,
                               float kCoverage, float kCollision, float kCost,
                               float costScale, float exploredMeanCoverage, float globalCollisionFrac,
                               float pTargetReactivate, float pMax, float repTarget, float repeatMax,
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

            // Fan-out from the shape the accept kernel already computed for this candidate.
            // INDEXED BY x1UnexploredIdx, NOT tid: the accept kernel writes the array by
            // unexplored-sample slot, so indexing by the compacted position would read another
            // node's shape. Every accepted node passed through the accept kernel -- including the
            // min-cost exemptions, which write a neutral 1.0 before their early return -- so this
            // slot is always fresh.
            activeFrontierRepeatCount[x1TreeIdx] =
              repeatFromShape(frontierNextAcceptShape[x1UnexploredIdx], repTarget, repeatMax);

            // Update best-node index if this is the new region best
            if(cost <= minCostsR1[xR1])
                atomicExch(&bestNodeIdxPerR1[xR1], x1TreeIdx);

            // Goal criteria check - accumulate goal nodes into goalSet; the min-cost
            // path is reconstructed afterwards by getControlPathToGoal.
            //
            // MUST STAY LAST, after the repeat assignment above: it clears the count, and running
            // it in the other order would let the fan-out write resurrect a nonzero count.
            if(distance(x1, s_xGoal) < GOAL_THRESH && cost <= *minCost)
                {
                    atomicMinFloat(minCost, cost);
                    goalSet[x1TreeIdx]    = true;
                    frontier[x1TreeIdx]   = false;
                    // The repeat count MUST be cleared with the frontier flag. repeatInd expands
                    // activeFrontierRepeatCount over the compacted frontier list, which findInd
                    // builds from frontier==true only. A node left with count>0 but frontier==false
                    // owns a slice of d_activeFrontierRepeatIdxs_ that no thread ever writes, yet
                    // h_frontierRepeatSize_ still spans it -- so propagateFrontier reads whatever
                    // stale tree indices that slice held (from a prior, much larger tree) and
                    // expands from not-yet-created nodes whose cost still reads 0. That produced
                    // phantom-parented, artificially cheap nodes that then won minCost.
                    activeFrontierRepeatCount[x1TreeIdx] = 0;
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
                    // NEUTRAL shape, not the maximum. There is one of these per explored region --
                    // up to NUM_R1_REGIONS of them -- so repeatMax here would dwarf the entire
                    // propagation budget on its own. It is still a promotion over CleanCost's flat
                    // 1, which gave the region best the SMALLEST fan-out of any node in the tree.
                    activeFrontierRepeatCount[treeIdx] = repeatFromShape(1.0f, repTarget, repeatMax);
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
                    int   tot        = counterArray[xR1];
                    float r1CollFrac = (tot > 0) ? (float)(tot - validCounterArray[xR1]) / (float)tot
                                                 : globalCollisionFrac;

                    float shape = comboShape(cost, r1MeanCost, costScale, regionCoverage[xR1], exploredMeanCoverage,
                                             r1CollFrac, globalCollisionFrac, kCoverage, kCollision, kCost);

                    float reactivationProb = fminf(shape * pTargetReactivate, pMax);

                    curandState seed = randomSeeds[treeIdx];
                    if(curand_uniform(&seed) < reactivationProb)
                        {
                            frontier[treeIdx]                  = true;
                            activeFrontierRepeatCount[treeIdx] = repeatFromShape(shape, repTarget, repeatMax);
                        }
                    randomSeeds[treeIdx] = seed;
                }
            else
                {
                    // REPAIR ARM -- absent from CleanCost, which is why a stuck frontier bit there
                    // is permanent. activeFrontierRepeatCount is zeroed wholesale every iteration,
                    // and the branch above is the only place a node in [0, treeSize) gets a new
                    // count -- but its guard is frontier == 0. So a node still flagged
                    // frontier == true at this point (i.e. propagate did not expand it, because it
                    // held no block) leaves this kernel with count 0 as well. repeatInd then emits
                    // no block for it again, kernel1 clears the frontier bit only from the
                    // expanding block, and the bit is stuck true forever: it inflates
                    // h_frontierSize_ and this same guard rejects the node on every future
                    // iteration. Handing it a block is what breaks the cycle.
                    activeFrontierRepeatCount[treeIdx] = repeatFromShape(1.0f, repTarget, repeatMax);
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

// Nodes this iteration should add, under the growth schedule.
//
// growthExp == 1 is the linear schedule and reduces EXACTLY to remaining / itersLeft. Larger values
// front-load growth, which is worth trying because propagation capacity SHRINKS as the tree buffer
// fills -- the kernel1 ceiling is proportional to what is left.
//
// Expressed as "this iteration's share of the REMAINING schedule" rather than an absolute target so
// it stays self-correcting: an iteration that under-delivers raises every later iteration's demand
// a little, instead of dumping the whole shortfall onto the next one.
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
    // Dividing by meanShapePrev corrects for E[shape] != 1. The /1.5 in comboShape makes the shape
    // 1.0 at the NEUTRAL point, but the deltas are asymmetric -- bounded at +1 on the unfavourable
    // side, unbounded on the favourable side -- so the realised mean is not 1 and the bias would
    // otherwise pass straight through into the growth rate.
    float rolled = float(h_candidatesPreGate_) - float(h_exemptCount_);
    float shapeAdj = fmaxf(1e-3f, h_meanShapePrev_);
    h_pTargetAccept_ = (rolled > 0.0f) ? fmaxf(0.0f, wantThisIter - float(h_exemptCount_)) / (rolled * shapeAdj) : 0.0f;
    h_pTargetAccept_ = fminf(h_pTargetAccept_, h_pMax_);

    // Reactivation budget, over ITS OWN population -- the whole tree, not the candidate list. A
    // single shared scalar is what would break here: at treeSize = 2e6 the acceptance budget applied
    // to the tree would reactivate more nodes per iteration than the entire growth target.
    // The pMax clamp is load-bearing at iteration 1, where treeSize == 1 and the raw ratio is ~1e3.
    h_pTargetReactivate_ = (h_treeSize_ > 0)
                             ? fminf(h_reactFrac_ * wantThisIter / float(h_treeSize_) / shapeAdj, h_pMax_)
                             : 0.0f;

    // --- THE acceptance decision: region-best candidates exempt, everything else kept with
    // min(pMax, comboShape(...) * pTargetAccept). ---
    // Guard the launch: iDivUp(0, block) is 0 blocks, which is cudaErrorInvalidConfiguration.
    if(h_frontierNextSize_ > 0)
        {
            KinoPaxSTARCOMBO_accept_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
              d_activeFrontierIdxs_ptr_, h_frontierNextSize_,
              d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
              graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_, graph_.d_regionCoverage_ptr_,
              d_frontierNextXR1s_ptr_, d_unexploredSampleCosts_ptr_,
              d_frontierNext_ptr_, d_randomSeeds_ptr_, d_frontierNextAcceptShape_ptr_,
              h_kCoverage_, h_kCollision_, h_kCost_,
              h_costScale_, h_exploredMeanCoverage_, h_globalCollisionFrac_,
              h_pTargetAccept_, h_pMax_,
              h_countAcceptReasons_, d_acceptCounts_ptr_);
        }

    cudaMemcpy(h_acceptCounts_, d_acceptCounts_ptr_, ACC_NUM_SLOTS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Mean shape over the candidates that actually took the roll -- the population pTargetAccept is
    // divided across. Consumed NEXT iteration; held at its previous value when nothing rolled, so a
    // barren iteration cannot reset the correction to a meaningless number.
    if(rolled > 0.0f)
        {
            float shapeSum = float(h_acceptCounts_[ACC_SHAPE_SUM]) / float(COMBO_CREDIT_SCALE);
            float meanShape = shapeSum / rolled;
            if(meanShape > 1e-3f) h_meanShapePrev_ = meanShape;
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

    // --- Fan-out budget for the NEXT propagate. Computed here, after the post-gate re-scan,
    // because the frontier it has to size is the one this kernel is about to build:
    //   F_next = the nodes Part A will admit + the nodes Part B is budgeted to reactivate.
    // Using h_frontierSize_ instead -- the frontier propagate ALREADY consumed -- would be off by a
    // whole generation, and during growth that ratio is 5-30x.
    //
    // repWanted comes from selectivity (candidates examined per node kept). repCeiling is the
    // kernel1 condition (frontierRepeatSize * 32 <= MAX_TREE_SIZE - treeSize) solved for the mean
    // repeat, with a 0.8 margin -- taken as a MIN, not as the target. Filling the ceiling would
    // spend ~0.8x the whole remaining buffer on propagation every iteration purely because the
    // buffer happens to be empty, which is 2-3x the work for the same growth.
    //
    // Because the ceiling is enforced here, staying on the kernel1 path is an INVARIANT rather than
    // a tuning outcome -- until repTarget hits its floor of 1, which for a steady frontier is past
    // ~88% of the tree. It also keeps sum(rep) far below MAX_TREE_SIZE, so repeatInd never
    // truncates (truncation would silently drop the HIGHEST tree indices, i.e. the newest nodes).
    // nu is MEASURED, not estimated. 1 - globalCollisionFrac is the run-cumulative collision-free
    // fraction, already reduced from the counter arrays above, so it costs nothing extra and is
    // smoother than a single-iteration ratio (which the measured data says is stationary anyway).
    // 0.9 is only the degenerate-case fallback.
    float nu = (h_globalCollisionFrac_ > 0.0f && h_globalCollisionFrac_ < 1.0f)
                 ? (1.0f - h_globalCollisionFrac_) : 0.9f;
    float fNext = fmaxf(1.0f, float(h_frontierNextSize_) + h_reactFrac_ * wantThisIter);
    // h_meanShapePrev_ was just refreshed above, so this is THIS iteration's mean -- which is the
    // right one, because the shapes multiplying repTarget belong to the candidates this update
    // kernel is about to place. It is a mean over ROLLED candidates, while Part A's population is
    // exemptions (shape exactly 1.0) plus rolled-AND-ACCEPTED ones, which skew high because a high
    // shape is what got them accepted. Net effect is a bounded overshoot of maybe 20% in sum(rep);
    // the 0.8 margin in repCeiling absorbs it, and prop_attempted / frontier_repeat_size is logged
    // so the realised value is visible rather than assumed. Measuring it exactly would cost another
    // atomic to answer a question the clamp already contains.
    float denom = 32.0f * fNext * fmaxf(1e-3f, h_meanShapePrev_);
    float repWanted  = (h_selectivity_ * wantThisIter / nu) / denom;
    float repCeiling = 0.8f * remaining / denom;
    h_repTarget_ = fminf(h_repeatMax_, fmaxf(1.0f, fminf(repWanted, repCeiling)));

    // --- Update Frontier ---
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    KinoPaxSTARCOMBO_updateFrontier_kernel<<<iDivUp(h_frontierNextSize_ + h_treeSize_, h_blockSize_), h_blockSize_>>>(
      d_frontier_ptr_, d_frontierNext_ptr_, d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_goalSample_ptr_, h_treeSize_,
      d_unexploredSamples_ptr_, d_treeSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, d_activeFrontierRepeatCount_ptr_, d_randomSeeds_ptr_,
      d_frontierNextAcceptShape_ptr_,
      d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
      graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_, graph_.d_regionCoverage_ptr_,
      h_kCoverage_, h_kCollision_, h_kCost_,
      h_costScale_, h_exploredMeanCoverage_, h_globalCollisionFrac_,
      h_pTargetReactivate_, h_pMax_, h_repTarget_, h_repeatMax_,
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
