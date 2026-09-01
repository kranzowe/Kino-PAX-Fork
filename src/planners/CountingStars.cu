// CountingStars v2 -- KinoPAX*, with ONE GLOBAL NODE BUDGET filled in priority order.
//
// WHAT CHANGED FROM v1, AND WHY IT IS AN INVERSION RATHER THAN A TWEAK. v1 admitted by PER-REGION
// QUOTAS (explore_count, cost_count, react_count) and the global frontier size F was whatever those
// happened to produce. That is backwards for the thing that actually matters: GPU throughput is a
// function of frontier size, so frontier size should be the INPUT.
//
// v2 makes goal_frontier_size B the PRIMITIVE, tunable to whatever this GPU is fast at, and the
// doors fill it in priority order. The budget is met BY CONSTRUCTION, not tracked -- the same
// discipline that already makes the block ceiling work, where F and the frontier's block demand are
// both counted before the launch so the total is known before a single block runs.
//
// This also retires the pattern that has failed three times in this line: steering a GLOBAL quantity
// through PER-REGION knobs with feedback. COMBO fed back its fan-out threshold; COMBO fed back its
// surplus repHi; a throughput controller over v1's three counts would have been the third. Each one
// normalised, and each one therefore handed its gain straight back.
//
// WHAT IT KEEPS FROM THE STAR LINE. Propagate makes no admission decision; the accept passes run
// after it, once the region statistics have converged. That ordering is load-bearing and must not be
// relaxed: minCostsR1 / sumCostsR1 / cntCostsR1 are updated by atomics from the very threads that
// would read them, so a decision taken inside propagate would see a partial mean and two identical
// candidates would draw different answers purely from scheduling.
//
// Propagate does still do COUNTING -- candidate counts, R2 cell claims. That is not a relaxation of
// the rule: counting with atomics is exact and order-independent, and the rule is about STATISTICS
// being mid-flight. Nothing reads minCostsR1 until the launch has finished.
//
// THE DOORS, IN PRIORITY ORDER.
//
//   1. OPTIMAL    distance 0, i.e. cost <= minCostsR1[r]. UNCAPPED, and it has first claim every
//                 iteration -- a stronger optimality guarantee than v1's region-best reactivation,
//                 which only put a region's best back AFTER the fact. Safe uncapped while
//                 B > NUM_R1_REGIONS, since NUM_R1_REGIONS is the ceiling on how many nodes can be
//                 a region best in one iteration.
//   2. FRESHEST   explore_frac of the REMAINING budget, taken from the least-populated regions.
//   3. GUARANTEE  each active region's best node, if no optimal admission already covered it.
//   4. DRAW       uniform over the rest of the tree, filling whatever the budget has left.
//
// NEITHER SELECTION NEEDS A SORT, AND THAT IS NOT A PERFORMANCE ARGUMENT -- IT IS A STRUCTURAL ONE.
// distance 0 is a THRESHOLD, not an order: it is exactly v1's `cost <= minCostsR1[r]`. Top-X-freshest
// IS an order, but ordinality is a small non-negative integer, so a HISTOGRAM plus an exclusive scan
// gives the exact cutoff in two O(n) atomic passes, reusing atomics that are already everywhere in
// this file. (For the record, a sort would also have been affordable: thrust::sort_by_key dispatches
// to CUB radix sort at ~1-2 G keys/s on Pascal, so 1e5-3e5 candidates is 0.05-0.3 ms against a ~15 ms
// iteration. It is not here because it is not needed, not because it is slow.)
//
// ORDINALITY IS PER-REGION, NOT PER-CANDIDATE. Every candidate in region r shares regionNodeCount[r],
// so "freshest" means "from the least-populated region". That is the novelty signal we want and it
// costs a single read -- no per-candidate counter, no extra array. Ties break arbitrarily, which is
// what the boundary roll resolves.
//
// TWO ACCEPT PASSES, and the split is what makes the budget exact. The histogram must be COMPLETE
// before the cutoff is known, and the cutoff must be known before anything is admitted. Pass 1
// measures and stamps no door; pass 2 decides and is the only door writer.
//
// THE R2 DOOR IS GONE; THE R2 MARKING SURVIVES. Novelty is ordinality now, so no door reads a
// sub-cell. The claim is kept purely to feed h_touchedR2_ so r2_coverage_pct stays comparable with
// the KPAX-family baselines at O(1) -- the alternative is a thrust::count over d_activeSubVertices_
// every iteration, which is 2.1M elements at the coarse delta and 37.9M at `tiny`.
//
// THE R2 MAPPING IS FIXED HERE AND ONLY HERE. Graph.cu's initializeRegions_kernel does not invert
// getRegion, so its min-corners are wrong and every R2 identity built on them is scrambled. This
// planner carries a corrected copy so the coverage metric counts the right cells; Graph.cu is left
// alone so the existing baselines stay comparable. See the header, and check_region_math.py.
//
// Opts into Graph's dynamic score floor (1/N_active rather than a fixed EPSILON); see Graph.cuh.
// Carries NO retroactive pruning.
#include "planners/CountingStars.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>

CountingStars::CountingStars()
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
    d_doorCounts_             = thrust::device_vector<unsigned long long>(CS_NUM_DOOR_SLOTS, 0ULL);
    d_touchedR2Count_         = thrust::device_vector<uint>(1, 0u);

    // CountingStars' OWN min-corner table -- see the header for why Graph.cu's is not usable.
    d_minCornerCS_            = thrust::device_vector<float>(NUM_R1_REGIONS * STATE_DIM);

    // Per-R1, reset every iteration.
    d_regionCovered_          = thrust::device_vector<bool>(NUM_R1_REGIONS, false);
    d_candCounts_             = thrust::device_vector<int>(NUM_R1_REGIONS);
    // Per-R1, cumulative. THE ORDINALITY SOURCE: how many nodes this region has ever taken.
    d_regionNodeCount_        = thrust::device_vector<int>(NUM_R1_REGIONS);
    // The freshness histogram and the optimal count that sizes the budget it spends. Both are read
    // back to the host between the two accept passes, which is the only device->host round trip the
    // admission path adds over v1: 256 ints plus one uint.
    d_ordHistogram_           = thrust::device_vector<int>(CS_ORD_BUCKETS, 0);
    d_optimalCount_           = thrust::device_vector<uint>(1, 0u);
    // Per-node, tree-indexed, written once at admission.
    d_nodeBlocks_             = thrust::device_vector<int>(MAX_TREE_SIZE, 1);
    d_nodeDoor_               = thrust::device_vector<int>(MAX_TREE_SIZE, CS_DOOR_NONE);
    // Per-candidate, unexplored-sample-slot indexed.
    d_candDistance_           = thrust::device_vector<float>(MAX_TREE_SIZE, 0.0f);
    d_candDoor_               = thrust::device_vector<int>(MAX_TREE_SIZE, CS_DOOR_NONE);

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
    d_doorCounts_ptr_             = thrust::raw_pointer_cast(d_doorCounts_.data());
    d_touchedR2Count_ptr_         = thrust::raw_pointer_cast(d_touchedR2Count_.data());
    d_minCornerCS_ptr_            = thrust::raw_pointer_cast(d_minCornerCS_.data());
    d_regionCovered_ptr_          = thrust::raw_pointer_cast(d_regionCovered_.data());
    d_candCounts_ptr_             = thrust::raw_pointer_cast(d_candCounts_.data());
    d_regionNodeCount_ptr_        = thrust::raw_pointer_cast(d_regionNodeCount_.data());
    d_ordHistogram_ptr_           = thrust::raw_pointer_cast(d_ordHistogram_.data());
    d_optimalCount_ptr_           = thrust::raw_pointer_cast(d_optimalCount_.data());
    d_nodeBlocks_ptr_             = thrust::raw_pointer_cast(d_nodeBlocks_.data());
    d_nodeDoor_ptr_               = thrust::raw_pointer_cast(d_nodeDoor_.data());
    d_candDistance_ptr_           = thrust::raw_pointer_cast(d_candDistance_.data());
    d_candDoor_ptr_               = thrust::raw_pointer_cast(d_candDoor_.data());
    d_iterations_ptr_             = thrust::raw_pointer_cast(d_iterations_.data());
    d_pathCosts_ptr_              = thrust::raw_pointer_cast(d_pathCosts_.data());
    d_controlPathsToGoal_ptr_     = thrust::raw_pointer_cast(d_controlPathsToGoal_.data());

    cudaMalloc(&d_minCost_ptr_, sizeof(float));

    // The corrected R1 min-corner table. Computed ONCE here, exactly as Graph does for its own --
    // the corners are a pure function of the discretisation, not of the run. Everything that would
    // otherwise pass graph_.d_minValueInRegion_ passes this instead; see the header for why.
    CountingStars_initializeRegions_kernel<<<iDivUp(NUM_R1_REGIONS, h_blockSize_), h_blockSize_>>>(d_minCornerCS_ptr_);

    // Spatial hash grid for fast collision detection
    d_spatialHashGrid_ = createSpatialHashGrid();

    // Host buffer for the best (min-cost) reconstructed trajectory
    h_controlPathsToGoal_ = new float[MAX_ITER * SAMPLE_DIM];

    // ================================================================================
    // THE BUDGET. Two numbers, and they are the entire tuning surface. See the header for why the
    // budget is the primitive and the per-region quotas are gone.
    // ================================================================================

    // B -- nodes in one iteration's frontier. THE HEADLINE KNOB, and the sweep's headline axis,
    // because its whole purpose is finding where this GPU is fast. The default sits between the
    // two regimes this planner is read against: far above KinoPaxPlus's F ~ 10, far below a
    // frontier pinned at nActive.
    h_goalFrontierSize_ = 10000;

    // Share of the REMAINING budget (B - optimalCount) given to freshness. Kept equal to the
    // sweep's derived operating point (CS_DERIVED_EXPLORE_FRAC_ON), so a standalone plan() run and
    // a --single-point sweep pass are the same planner.
    h_exploreFrac_ = 0.1f;

    // Cost acceptance ON by default: the planner's normal behaviour, and the arm the toggle is
    // read against. See the header for why switching it off has to remove BOTH cost-driven doors.
    h_costAccept_ = true;

    // ---- Fan-out. Blocks a node gets are decided at admission; see the header for the rule. ----
    // rep is a plain COUNT OF BLOCKS with no alignment constraint -- repeatInd writes rep integer
    // entries and kernel1 launches one 32-thread block per entry, so a node at 16 gets
    // 16 x 32 = 512 propagations. blockBudget = maxBlocks * B follows from it.
    //
    // SWEPT, and independent of B: while the fan-out split is non-binding this IS
    // propagations-per-node, where B is frontier size. 16 matches the sweep's derived point.
    h_maxBlocks_   = 16;

    // ---- Derived per-iteration scalars. All recomputed before they are read; these are only the
    // values the CSV would show if a run somehow logged iteration 0. ----
    h_optimalCount_        = 0;   // stays 0 for the whole run when h_costAccept_ is false
    h_ordCutoff_           = 0;
    h_pBoundary_           = 0.0f;
    h_guaranteedReact_     = 0;
    h_budgetUsed_          = 0;
    h_admittedExplore_     = 0;
    h_admittedCost_        = 0;
    h_reactivated_         = 0;
    h_reactivatedBest_     = 0;
    h_blockCeiling_        = 0.0f;
    h_blockScale_          = 1.0f;
    h_globalCollisionFrac_ = 0.1f;
    h_costScale_           = 0.0f;
    h_touchedR2_           = 0;
    h_propAttempted_       = 0;
    h_candidatesPreGate_   = 0;
    for(int i = 0; i < CS_NUM_DOOR_SLOTS; i++) h_doorCounts_[i] = 0ULL;
    for(int i = 0; i < CS_ORD_BUCKETS; i++) h_ordHistogram_[i] = 0;

    h_activeBlockSize_ = 32;

    if(VERBOSE)
        {
            printf("/* Planner Type: CountingStars v2 (global budget) */\n");
            printf("/* Number of R1 Vertices: %d */\n", NUM_R1_REGIONS);
            printf("/* Number of R2 Vertices: %d */\n", NUM_R2_REGIONS);
            printf("/***************************/\n");
        }
}

CountingStars::~CountingStars()
{
    destroySpatialHashGrid(d_spatialHashGrid_);
    delete[] h_controlPathsToGoal_;
}

void CountingStars::resetPlanner(float* h_initial, float* h_goal)
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
    // No root seed here. propagateFrontier zeroes this array and assigns every count itself over
    // the compacted frontier, so a seed written here would be overwritten before it was read. The
    // root still opens wide: d_nodeBlocks_ is filled with h_maxBlocks_ below.
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
    // Region node counts are CUMULATIVE over a run, so this is the one place they are cleared.
    // Carrying them across a reset would start every region already looking full, and the freshness
    // door would admit nothing from iteration 1 onward.
    thrust::fill(d_regionNodeCount_.begin(), d_regionNodeCount_.end(), 0);
    thrust::fill(d_regionCovered_.begin(), d_regionCovered_.end(), false);
    thrust::fill(d_candCounts_.begin(), d_candCounts_.end(), 0);
    thrust::fill(d_ordHistogram_.begin(), d_ordHistogram_.end(), 0);
    thrust::fill(d_optimalCount_.begin(), d_optimalCount_.end(), 0u);
    // maxBlocks, not 1: the root is admitted by no door, so nothing else would ever write its count.
    thrust::fill(d_nodeBlocks_.begin(), d_nodeBlocks_.end(), h_maxBlocks_);
    thrust::fill(d_nodeDoor_.begin(), d_nodeDoor_.end(), CS_DOOR_NONE);
    thrust::fill(d_candDistance_.begin(), d_candDistance_.end(), 0.0f);
    thrust::fill(d_candDoor_.begin(), d_candDoor_.end(), CS_DOOR_NONE);
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
    // value to 0 (the sibling STAR planners survive only by lucky layout).
    h_propIterations_ = 1;

    // Every derived scalar is reset, not just the ones the constructor happened to set. CleanCost
    // reset NONE of these, so a planner object reused across runs -- which every benchmark does --
    // carried the previous run's final values into iteration 1.
    thrust::fill(d_doorCounts_.begin(), d_doorCounts_.end(), 0ULL);
    for(int i = 0; i < CS_NUM_DOOR_SLOTS; i++) h_doorCounts_[i] = 0ULL;
    for(int i = 0; i < CS_ORD_BUCKETS; i++) h_ordHistogram_[i] = 0;
    h_propAttempted_        = 0;
    h_candidatesPreGate_    = 0;
    h_frontierNextSize_     = 0;
    h_frontierRepeatSize_   = 0;
    h_globalCollisionFrac_  = 0.1f;
    h_costScale_            = 0.0f;
    thrust::fill(d_touchedR2Count_.begin(), d_touchedR2Count_.end(), 0u);
    h_touchedR2_            = 0;
    h_optimalCount_         = 0;
    h_ordCutoff_            = 0;
    h_pBoundary_            = 0.0f;
    h_guaranteedReact_      = 0;
    h_budgetUsed_           = 0;
    h_admittedExplore_      = 0;
    h_admittedCost_         = 0;
    h_reactivated_          = 0;
    h_reactivatedBest_      = 0;
    h_blockCeiling_         = 0.0f;
    h_blockScale_           = 1.0f;

    cudaMemcpy(d_treeSamples_ptr_, h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costToGoal_ptr_, &h_costToGoal_, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pathToGoal_ptr_, &h_pathToGoal_, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minCost_ptr_, &h_minCost_, sizeof(float), cudaMemcpyHostToDevice);

    initializeRandomSeeds(static_cast<unsigned int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
}

void CountingStars::plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
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
    std::cout << "CountingStars execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CountingStars::planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr)
{
    double t_start = std::clock();
    resetPlanner(h_initial, h_goal);

    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            printf("Iteration: %d, Tree Size: %d, Frontier Size: %d\n", h_itr_, h_treeSize_, h_frontierSize_);
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            updateFrontier();

            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    double executionTime = (std::clock() - t_start) / (double)CLOCKS_PER_SEC;
    std::cout << "CountingStars execution time: " << executionTime << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
}

float CountingStars::planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
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
            updateFrontier();
            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "CountingStars execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return h_minCost_;
}

// Gathers a frontier node's admission-time block count by TREE INDEX, so the host can total the
// frontier's demand before launching anything and shrink it if the buffer cannot take it. Reduced
// into 64-bit: the frontier can be ~1e5 nodes at up to maxBlocks each, and this is summed every
// iteration -- the same class of overflow the graph's int counter arrays have.
struct CountingStars_BlocksOf
{
    const int* nodeBlocks;
    __host__ __device__ long long operator()(uint treeIdx) const { return (long long)nodeBlocks[treeIdx]; }
};

// Counts the regions the GUARANTEE will have to cover: an active region (one that has a best node)
// whose best was not already re-admitted through the optimal door this iteration.
//
// WHY THE HOST NEEDS THIS BEFORE THE LAUNCH. The draw's probability is
// (B - admitted - guaranteed) / treeSize, so `guaranteed` has to be known before updateFrontier
// runs -- it cannot be discovered inside the kernel that spends it. That is why regionCovered is
// written by accept pass 2 rather than by Part A: pass 2 is the last point at which an optimal
// admission is decided, and it is before this reduction.
struct CountingStars_UncoveredBest
{
    const int*  bestNodeIdxPerR1;
    const bool* regionCovered;
    __host__ __device__ int operator()(int r) const
    {
        return (bestNodeIdxPerR1[r] >= 0 && !regionCovered[r]) ? 1 : 0;
    }
};

void CountingStars::propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount)
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
    // FAN-OUT. Every node already carries the block count it was admitted with; this step only
    // totals the frontier's demand, shrinks it if the buffer cannot take it, and writes the counts.
    //
    // WHY IT IS HERE. d_activeFrontierIdxs_[0, h_frontierSize_) is, at this instant, the WHOLE
    // frontier -- optimal admissions, freshness admissions and reactivations alike -- so the total
    // is known before a single block is launched. That makes the ceiling something to SOLVE against
    // rather than clamp after the fact.
    //
    // THIS IS ALSO THE ONLY WRITER of d_activeFrontierRepeatCount_. It runs over exactly the
    // compacted frontier, so rep >= 1 holds for every member BY CONSTRUCTION -- no node can be left
    // blockless (which would strand its frontier bit forever, since kernel1 clears the bit from the
    // expanding block) and no node outside the frontier can hold a count (which would make repeatInd
    // emit a slice no thread writes, fathering phantom-parented nodes at cost 0). A goal node clears
    // its own frontier bit and is therefore absent here, which is why Part A needs no count clearing
    // and no ordering constraint.
    // ================================================================================
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    if(h_frontierSize_ > 0)
        {
            // Blocks the frontier is asking for, and the blocks the buffer can give.
            //
            // FLOAT CASTS ARE MANDATORY: h_treeSize_ is uint, so an overshoot wraps to ~4e9. The
            // 0.8 is the margin against the kernel1 condition below; h_activeBlockSize_, never a
            // literal 32, because that condition uses the member and hard-coding the value here
            // would silently desynchronise them.
            CountingStars_BlocksOf blocksOf{d_nodeBlocks_ptr_};
            long long wantBlocks = thrust::transform_reduce(d_activeFrontierIdxs_.begin(),
                                                            d_activeFrontierIdxs_.begin() + h_frontierSize_,
                                                            blocksOf, (long long)0, thrust::plus<long long>());

            float remaining = fmaxf(0.0f, float(MAX_TREE_SIZE) - float(h_treeSize_));
            h_blockCeiling_ = 0.8f * remaining / float(h_activeBlockSize_);

            // Shrink the BOOST, never the floor. Every frontier node keeps its one block whatever
            // happens, so only the excess above F is scalable:
            //
            //   sum(rep) = F + scale * (wantBlocks - F)  <=  blockCeiling
            //
            // scale == 1 means the ceiling did not bind. Below 1 says the BUFFER, not the fan-out
            // rule, is setting how hard nodes expand -- and a scale near 0 means F itself has eaten
            // the budget, which is a goal_frontier_size problem and no other knob will move it.
            float excess  = float(wantBlocks) - float(h_frontierSize_);
            float allowed = h_blockCeiling_ - float(h_frontierSize_);
            h_blockScale_ = (excess > 0.0f) ? fminf(1.0f, fmaxf(0.0f, allowed / excess)) : 1.0f;

            CountingStars_assignFanout_kernel<<<iDivUp(h_frontierSize_, h_blockSize_), h_blockSize_>>>(
              h_frontierSize_, d_activeFrontierIdxs_ptr_, d_nodeBlocks_ptr_, h_blockScale_,
              d_activeFrontierRepeatCount_ptr_);
        }

    // --- Per-iteration per-region counters, zeroed before propagate fills them. ---
    thrust::fill(d_candCounts_.begin(), d_candCounts_.end(), 0);

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

    // Cap the expanded frontier to the tree buffer: h_frontierRepeatSize_ (sum of the per-node
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

            CountingStars_propagateFrontier_kernel2<<<iDivUp(h_propIterations_ * h_frontierRepeatSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              h_propIterations_, d_minCornerCS_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
              d_frontierNextXR1s_ptr_, d_candDoor_ptr_,
              d_candCounts_ptr_, d_touchedR2Count_ptr_,
              d_unexploredSampleCosts_ptr_, h_spatialHashGrid_);

            // kernel2 launches h_frontierRepeatSize_ * h_propIterations_ threads, one candidate each.
            h_propAttempted_ = h_frontierRepeatSize_ * (uint)h_propIterations_;
        }
    else
        {
            CountingStars_propagateFrontier_kernel1<<<iDivUp(h_frontierRepeatSize_ * h_activeBlockSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              d_minCornerCS_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
              d_frontierNextXR1s_ptr_, d_candDoor_ptr_,
              d_candCounts_ptr_, d_touchedR2Count_ptr_,
              d_unexploredSampleCosts_ptr_, h_spatialHashGrid_);

            // kernel1 launches one block of h_activeBlockSize_ threads per repeat entry.
            h_propAttempted_ = h_frontierRepeatSize_ * h_activeBlockSize_;
        }
}

/***************************/
/* FAN-OUT ASSIGNMENT KERNEL */
/***************************/
// One thread per COMPACTED FRONTIER ENTRY, so the frontier is covered exactly: every member is
// written once, nothing outside it is touched. Indexing by activeFrontierIdxs[tid] and not by tid is
// the whole reason this is a separate launch from repeatInd -- nodeBlocks is tree-indexed.
__global__ void CountingStars_assignFanout_kernel(uint frontierSize, uint* activeFrontierIdxs,
                                                  int* nodeBlocks, float scale,
                                                  uint* activeFrontierRepeatCount)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierSize) return;

    uint treeIdx = activeFrontierIdxs[tid];
    int  want    = nodeBlocks[treeIdx];
    if(want < 1) want = 1;

    // Scale the BOOST, not the floor, and FLOOR the result rather than rounding. Rounding to
    // nearest would overshoot the ceiling by up to half a block per favoured node -- small, but it
    // would turn sum(rep) <= blockCeiling back into an approximation, and approximating this budget
    // is what used to flip propagate onto the slow kernel2 path.
    unsigned int rep = 1u + (unsigned int)floorf(float(want - 1) * scale);
    activeFrontierRepeatCount[treeIdx] = (rep >= 1u) ? rep : 1u;
}

/***************************/
/* R1 MIN-CORNER INITIALISATION */
/***************************/
// The exact inverse of getRegion's encode:
//
//   r1 = wRegion * C_R1_LENGTH^C_DIM * V_R1_LENGTH^V_DIM + aRegion * V_R1_LENGTH^V_DIM + vRegion
//
// so the groups strip off in reverse significance -- velocity first, then attitude, and whatever
// remains is workspace. Graph.cu's version reads them in the opposite order AND uses hardcoded
// exponents (C_R1_LENGTH^2, V_R1_LENGTH^1) where the encode uses C_DIM and V_DIM, which collapses
// NUM_R1_REGIONS regions onto far fewer distinct corners. Written entirely in config macros so it
// stays correct at any discretisation; scripts/check_region_math.py proves it is a bijection.
//
// The WITHIN-group digit order below is the same as Graph.cu's and was never wrong: getRegion builds
// each group with axis 0 as the most significant digit.
__global__ void CountingStars_initializeRegions_kernel(float* minCorner)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= NUM_R1_REGIONS) return;

    int cPow = 1;
    for(int i = 0; i < C_DIM; ++i) cPow *= C_R1_LENGTH;
    int vPow = 1;
    for(int i = 0; i < V_DIM; ++i) vPow *= V_R1_LENGTH;

    int vRegion = tid % vPow;
    int aRegion = (tid / vPow) % cPow;
    int wRegion = tid / (vPow * cPow);

    int temp = wRegion;
    for(int i = W_DIM - 1; i >= 0; --i)
        {
            minCorner[tid * STATE_DIM + i] = W_MIN + (temp % W_R1_LENGTH) * W_R1_SIZE;
            temp /= W_R1_LENGTH;
        }

    temp = aRegion;
    for(int i = C_DIM - 1; i >= 0; --i)
        {
            minCorner[tid * STATE_DIM + W_DIM + i] = C_MIN + (temp % C_R1_LENGTH) * C_R1_SIZE;
            temp /= C_R1_LENGTH;
        }

    temp = vRegion;
    for(int i = V_DIM - 1; i >= 0; --i)
        {
            minCorner[tid * STATE_DIM + W_DIM + C_DIM + i] = V_MIN + (temp % V_R1_LENGTH) * V_R1_SIZE;
            temp /= V_R1_LENGTH;
        }
}

/***************************/
/* PROPAGATE FRONTIER KERNEL 1 */
/***************************/
// One Block Per Frontier Sample — CANDIDATE PRODUCER ONLY. No acceptance decision, no RNG draw:
// every collision-free sample is recorded with its cost and region, and the accept passes decide
// once the region statistics have converged.
__global__ void CountingStars_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minCorner,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, int* candDoor,
                                                   int* candCounts, uint* touchedR2Count,
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
    int x1SubVertex                  = getSubRegion(x1, x1Vertex, minCorner);

    // --- Update Graph statistics ---
    atomicAdd(&vertexCounter[x1Vertex], 1);
    if(valid)
        {
            atomicAdd(&validVertexCounter[x1Vertex], 1);

            // Cumulative cost from root
            float cost = s_x0Cost + edgeCost(s_x0, x1);

            // --- Region cost statistics. These are what the accept passes read once the launch has
            // finished; reading them HERE would see them mid-flight. ---
            if(minCostsR1[x1Vertex] > cost) atomicMinFloat(&minCostsR1[x1Vertex], cost);
            atomicAdd(&sumCostsR1[x1Vertex], cost);
            atomicAdd(&cntCostsR1[x1Vertex], 1);

            atomicAdd(&candCounts[x1Vertex], 1);

            // --- R2 MARKING, FOR THE COVERAGE METRIC ONLY. No door reads a sub-cell in v2;
            // ordinality replaced novelty. This survives so r2_coverage_pct stays comparable with
            // the KPAX-family baselines, and it stays in THIS form because the CAS's return value
            // is what makes the running total exact: exactly one thread in the whole launch can
            // turn a given cell from 0 to 1, so touchedR2Count gains exactly one per cell, ever.
            //
            // READ-THEN-CAS, not a bare CAS. The overwhelming majority of candidates land in cells
            // that were claimed iterations ago, and a plain load rejects those without touching the
            // atomic unit at all. The two are exactly equivalent: a cell only ever goes 0 -> 1, so a
            // load that sees 1 can never be a stale rejection of a cell that is still free. ---
            if(activeSubVertices[x1SubVertex] == 0 && atomicCAS(&activeSubVertices[x1SubVertex], 0, 1) == 0)
                atomicAdd(touchedR2Count, 1u);

            // --- Record the candidate. No admission decision, no RNG draw. The door is CLEARED
            // rather than left alone: these slots are reused every iteration, and a stale door from
            // an earlier batch would be read by Part A as an admission. ---
            candDoor[tid]              = CS_DOOR_NONE;
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
__global__ void CountingStars_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, int iterations,
                                                   float* minCorner,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, int* candDoor,
                                                   int* candCounts, uint* touchedR2Count,
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
    int x1SubVertex                  = getSubRegion(x1, x1Vertex, minCorner);

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

            atomicAdd(&candCounts[x1Vertex], 1);

            // --- R2 marking for the coverage metric only, read-then-CAS (see kernel 1). ---
            if(activeSubVertices[x1SubVertex] == 0 && atomicCAS(&activeSubVertices[x1SubVertex], 0, 1) == 0)
                atomicAdd(touchedR2Count, 1u);

            // --- Record the candidate (see kernel 1 for why the door is cleared here). ---
            candDoor[tid]              = CS_DOOR_NONE;
            unexploredSampleCosts[tid] = cost;
            frontierNextXR1s[tid]      = x1Vertex;
            frontierNext[tid]          = true;
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* ACCEPT PASS 1 - measure, do not decide */
/***************************/
// Runs after propagate has finished, so minCostsR1 is converged rather than mid-flight -- the one
// invariant CleanCost established that this planner keeps.
//
// It computes the two quantities the budget is spent against, and NOTHING else:
//
//   distance   (cost - minCostsR1[r]) / costScale, and 0 IS THE OPTIMAL MARK. costScale is
//              CleanCost's global scale -- (mean cost over valid samples) - (min over regions) --
//              which is what makes the distance scale-free rather than a raw cost difference.
//   ordinality regionNodeCount[r], the candidate's REGION's population. Per-region, not
//              per-candidate: "freshest" means "from the least-populated region", which is a single
//              read with no per-candidate counter behind it.
//
// THE OPTIMAL TEST IS `cost <= minCostsR1[r]`, NOT `distance == 0.0f`, and the difference matters
// when costScale collapses to 0 (an empty or single-cost tree): the division would be 0/0. The
// comparison is exact, is what v1's cost door was, and distance is DERIVED from it afterwards --
// so the flag written into candDistance is 0 for exactly the set the comparison selects.
__global__ void CountingStars_acceptPass1_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                                 float* minCostsR1, int* frontierNextXR1s, float* unexploredSampleCosts,
                                                 int* regionNodeCount, float costScale, bool costAccept,
                                                 float* candDistance, int* ordHistogram, uint* optimalCount)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierNextSize) return;

    // ONE indirection, and every array below is indexed by idx -- the unexplored-sample SLOT --
    // never by tid, the compacted position. Indexing by tid would read another candidate's data.
    int   idx  = activeFrontierNextIdxs[tid];
    float cost = unexploredSampleCosts[idx];
    int   xR1  = frontierNextXR1s[idx];
    float m    = minCostsR1[xR1];

    // THE EARLY RETURN IS GATED, and that gate is the whole reason this kernel takes the flag.
    // It SKIPS THE HISTOGRAM: with cost acceptance off, an ungated return would leave every region
    // best out of the freshness competition as well as out of the optimal door, so it would be
    // rejected every iteration for the crime of being cheap. Falling through instead lets it
    // compete on ordinality like any other candidate.
    if(costAccept && cost <= m)
        {
            // OPTIMAL. Uncapped, and it does not enter the histogram: the freshness door spends
            // what is LEFT of the budget after these, so counting them as fresh too would let one
            // node consume two doors' worth of it.
            candDistance[idx] = 0.0f;
            atomicAdd(optimalCount, 1u);
            return;
        }

    // Non-optimal, so the distance written here MUST NOT BE ZERO -- pass 2 reads 0 as the optimal
    // mark. Two ways it could be: a collapsed costScale (the ratio is 0/0 rather than infinite),
    // and an underflow when the spread is enormous next to the gap. Both fall back to the raw
    // difference, which cannot be 0 here: cost > m, and float subtraction of nearby values is exact.
    //
    // WITH costAccept FALSE a region best reaches this line and writes distance 0 anyway (cost == m
    // makes both the ratio and the fallback 0). That is harmless ONLY because pass 2 gates the
    // optimal branch on the FLAG rather than on distance == 0. Do not "simplify" pass 2 back to a
    // bare distance test.
    float d = (costScale > 0.0f) ? ((cost - m) / costScale) : (cost - m);
    candDistance[idx] = (d != 0.0f) ? d : (cost - m);

    int ord = regionNodeCount[xR1];
    if(ord < 0) ord = 0;
    if(ord >= CS_ORD_BUCKETS) ord = CS_ORD_BUCKETS - 1;
    atomicAdd(&ordHistogram[ord], 1);
}

/***************************/
/* ACCEPT PASS 2 - the ONLY admission decision */
/***************************/
// Admits in priority order against the cutoff the host solved from pass 1's histogram:
//
//   OPTIMAL   distance == 0                                      door = COST     (uncapped)
//   FRESHEST  ordinality <  cutoff                               door = EXPLORE
//             ordinality == cutoff, with probability pBoundary   door = EXPLORE
//
// THE BOUNDARY ROLL IS WHAT MAKES THE COUNT EXACT. The X-th freshest node almost never falls on a
// bucket edge, and admitting the whole boundary bucket would overshoot the budget by up to one
// bucket's width -- which, at a coarse ordinality where thousands of candidates share a value, is
// most of a frontier. The roll spends the fractional remainder and nothing more.
//
// ORDINALITY IS RE-READ HERE RATHER THAN CARRIED FROM PASS 1, and that is safe by construction:
// regionNodeCount is only written by Part A of updateFrontier, which runs after both passes. The
// clamp must match pass 1's exactly, or a top-bucket candidate would be compared against a cutoff
// derived from a histogram it was never counted in.
__global__ void CountingStars_acceptPass2_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                                 int* frontierNextXR1s, int* regionNodeCount,
                                                 float* candDistance, bool* frontierNext, int* candDoor,
                                                 bool* regionCovered, curandState* randomSeeds,
                                                 int ordCutoff, float pBoundary, bool costAccept,
                                                 unsigned long long* doorCounts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierNextSize) return;

    int idx = activeFrontierNextIdxs[tid];
    int xR1 = frontierNextXR1s[idx];

    // --- OPTIMAL: first claim on the budget, every iteration. That is a stronger optimality
    // guarantee than v1's region-best reactivation, which only restored a region's best AFTER it
    // had already been passed over. ---
    // GATED ON THE FLAG, NOT ON THE DISTANCE. With cost acceptance off a region best still carries
    // distance 0 out of pass 1 (see there), so a bare distance test would let the door back in.
    if(costAccept && candDistance[idx] == 0.0f)
        {
            candDoor[idx] = CS_DOOR_COST;
            // Part B's guarantee is deduplicated against this: a region whose best came back in
            // through the top door does not also spend a guarantee slot on the node it superseded.
            regionCovered[xR1] = true;
            atomicAdd(&doorCounts[CountingStars::CS_SLOT_COST], 1ULL);
            return;
        }

    // --- FRESHEST: from the least-populated regions, up to the cutoff. ---
    int ord = regionNodeCount[xR1];
    if(ord < 0) ord = 0;
    if(ord >= CS_ORD_BUCKETS) ord = CS_ORD_BUCKETS - 1;

    bool take = (ord < ordCutoff);
    if(!take && ord == ordCutoff && pBoundary > 0.0f)
        {
            curandState seed = randomSeeds[idx];
            take             = (curand_uniform(&seed) < pBoundary);
            randomSeeds[idx] = seed;
        }

    if(take)
        {
            candDoor[idx] = CS_DOOR_EXPLORE;
            atomicAdd(&doorCounts[CountingStars::CS_SLOT_EXPLORE], 1ULL);
            return;
        }

    // --- Rejected. Subtractive, like CleanCost's: propagate set the flag, admission leaves it, and
    // only rejection clears it. ---
    candDoor[idx]     = CS_DOOR_NONE;
    frontierNext[idx] = false;
}

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// Part A inserts this iteration's admitted candidates; Part B fills what the budget has left. The
// two run in one launch over disjoint index ranges -- Part A owns
// [treeSize, treeSize + frontierNextSize) and Part B owns [0, treeSize) -- so they never contend for
// a node.
//
// EVERY BRANCH THAT SETS frontier[i] = true MUST WRITE nodeBlocks[i]. A missed one leaves the node
// carrying whatever block count the previous occupant of its tree slot had, and it fails silently:
// the node simply expands by the wrong amount.
__global__ void
CountingStars_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               curandState* randomSeeds,
                               int* candDoor, int* nodeDoor, int* nodeBlocks,
                               int* regionNodeCount, bool* regionCovered,
                               float* minCostsR1, int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
                               float* minCost, float* unexploredSampleCosts, bool* goalSet,
                               int* iterations, int iteration,
                               float pReactivate, int maxBlocks, int otherBlocks, bool costAccept,
                               unsigned long long* doorCounts)
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
            frontierNext[x1UnexploredIdx] = false;

            float* x1  = &unexploredSamples[x1UnexploredIdx * SAMPLE_DIM];
            int x0Idx  = unexploredSamplesParentIdxs[x1UnexploredIdx];
            float cost = unexploredSampleCosts[x1UnexploredIdx];
            int xR1    = frontierNextXR1s[x1UnexploredIdx];
            int door   = candDoor[x1UnexploredIdx];

            // Transfer to tree
            treeSamplesParentIdxs[x1TreeIdx] = x0Idx;
            for(int i = 0; i < SAMPLE_DIM; i++)
                treeSamples[x1TreeIdx * SAMPLE_DIM + i] = x1[i];
            treeSampleCosts[x1TreeIdx] = cost;
            treeXR1s[x1TreeIdx]        = xR1;
            nodeDoor[x1TreeIdx]        = door;

            // Always add to frontier (it survived the gate)
            frontier[x1TreeIdx] = true;

            // --- THE REGION'S POPULATION, which is the ordinality every LATER iteration's freshness
            // cutoff is measured against. Cumulative over the run and never reset mid-run, so it is
            // "how many nodes this region has ever taken". The atomicAdd's return value is not
            // needed any more -- v1 used it to index the geometric fan-out ramp, which is gone with
            // the explore door that indexed it -- but the increment itself is load-bearing. ---
            atomicAdd(&regionNodeCount[xR1], 1);

            // --- FAN-OUT, decided here and read next iteration by propagateFrontier.
            //
            // OPTIMAL gets the full maxBlocks; everyone else shares what the design budget has left,
            // which the host has already divided into otherBlocks. The split is non-binding while
            // the frontier lands at or under B (otherBlocks is then exactly maxBlocks) and bites on
            // an overshoot, which is the case it exists for. ---
            int blocks = (door == CS_DOOR_COST) ? maxBlocks : otherBlocks;
            nodeBlocks[x1TreeIdx] = (blocks > 1) ? blocks : 1;

            // Update best-node index if this is the new region best. THE GUARANTEE'S TABLE: Part B
            // reads it to put a region's cheapest node back when no optimal admission covered the
            // region this iteration. One atomicExch on a branch that is already taken.
            if(cost <= minCostsR1[xR1])
                atomicExch(&bestNodeIdxPerR1[xR1], x1TreeIdx);

            // Goal criteria check - accumulate goal nodes into goalSet; the min-cost path is
            // reconstructed afterwards by getControlPathToGoal.
            //
            // NO ORDERING CONSTRAINT. This used to have to run last, because it cleared a repeat
            // count the fan-out write could otherwise resurrect -- leaving a node with count > 0 and
            // frontier == false, owning a slice of d_activeFrontierRepeatIdxs_ that no thread
            // writes, so propagate expanded stale tree indices into phantom-parented nodes at cost 0
            // that then won minCost. Counts are not written here at all now: clearing the frontier
            // bit is sufficient, because propagateFrontier zeroes every count and then writes only
            // the compacted frontier, which a goal node is by definition not in.
            if(distance(x1, s_xGoal) < GOAL_THRESH && cost <= *minCost)
                {
                    atomicMinFloat(minCost, cost);
                    goalSet[x1TreeIdx]  = true;
                    frontier[x1TreeIdx] = false;
                    iterations[x1TreeIdx] = iteration;
                }
        }

    // --- Part B: Re-activate existing tree nodes, filling what the budget has left ---
    else if(tid < frontierNextSize + treeSize)
        {
            int treeIdx = tid - frontierNextSize;
            if(goalSet[treeIdx]) return;
            if(frontier[treeIdx]) return;   // already in the frontier; nothing to draw for

            int xR1 = treeXR1s[treeIdx];

            // --- THE REGION-BEST GUARANTEE, DEDUPLICATED. Unconditional for an UNCOVERED region:
            // no roll, no budget, ahead of the draw.
            //
            // This is KinoPaxPlus's invariant and the reason a region's cheapest node keeps getting
            // expanded until it stops being cheapest. Without it a region's best sits in the frontier
            // only by luck, its subtree goes many iterations without improvement, and FINAL COST
            // STALLS.
            //
            // WHAT THE DEDUP BUYS OVER v1. v1 ran this for EVERY active region every iteration, so
            // F >= nActive unconditionally. Here a region whose best was just re-admitted through
            // the optimal door is already covered, so the guarantee only pays for the regions the
            // top door missed -- and the host has counted exactly those (see
            // CountingStars_UncoveredBest) before choosing pReactivate.
            //
            // IT IS STILL A FLOOR ON F AT THE UNCOVERED-REGION COUNT, and that is the honest limit
            // on what B can control: this arm is UNCONDITIONAL, so once nActive exceeds B the
            // guarantee alone overruns the budget and the draw contributes nothing. B binds only
            // while B > NUM_R1_REGIONS. Capping it here is the obvious next lever if the sweep says
            // B is not moving F -- KinoPaxPlus's precedent is hysteresis (un-prune a region best
            // only after ~5 idle iterations), which caps re-entry at nActive/5 without giving up
            // the invariant.
            //
            // bestNodeIdxPerR1 is written in Part A under `cost <= minCostsR1[xR1]` with an
            // atomicExch, so ties resolve arbitrarily -- exactly one node per region, which is what
            // this arm wants. A Part A write racing this read can only move the guarantee by one
            // node for one iteration, and the covered case is gated by regionCovered anyway.
            // GATED with the optimal door, and they must be gated TOGETHER. regionCovered is
            // written only by the optimal door, so disabling that door alone would leave every
            // active region reading as uncovered and fire this arm for all of them -- turning a
            // change meant to SHRINK the frontier into one that grows it to ~nActive.
            if(costAccept && !regionCovered[xR1] && treeIdx == bestNodeIdxPerR1[xR1])
                {
                    frontier[treeIdx]   = true;
                    nodeDoor[treeIdx]   = CS_DOOR_BEST;
                    nodeBlocks[treeIdx] = (otherBlocks > 1) ? otherBlocks : 1;
                    atomicAdd(&doorCounts[CountingStars::CS_SLOT_BEST], 1ULL);
                    return;
                }

            // --- THE DRAW. Uniform over what the guarantee did not take, at
            // p = (B - admitted - guaranteed) / treeSize, so the EXPECTED number added is exactly
            // the budget's remainder whatever the tree size.
            //
            // It exists for a different job from the guarantee. The guarantee keeps every region's
            // best expandable, which is an OPTIMALITY mechanism. This keeps a thin random sample of
            // ordinary nodes alive, which is a REACH mechanism -- without it the only nodes ever
            // re-expanded would be region bests, and the tree would only ever deepen along its own
            // cheapest paths.
            //
            // Uniform is the simplest thing that hits the count exactly. Top-K by cost, by recency,
            // or a mix all slot in HERE and nowhere else. ---
            if(pReactivate > 0.0f)
                {
                    curandState seed = randomSeeds[treeIdx];
                    bool take        = curand_uniform(&seed) < pReactivate;
                    randomSeeds[treeIdx] = seed;

                    if(take)
                        {
                            frontier[treeIdx]   = true;
                            nodeDoor[treeIdx]   = CS_DOOR_REACT;
                            nodeBlocks[treeIdx] = (otherBlocks > 1) ? otherBlocks : 1;
                            atomicAdd(&doorCounts[CountingStars::CS_SLOT_REACT], 1ULL);
                        }
                }
        }
}


void CountingStars::updateFrontier()
{
    // --- Find indices and size of the candidate list ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontierNext_[MAX_TREE_SIZE - 1]) ? ++h_frontierNextSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // Collision-free candidates the accept passes are about to judge. Captured here because the
    // post-gate re-scan below overwrites h_frontierNextSize_ with the survivors.
    //
    // Do NOT reconstruct it as frontierRepeatSize * 32 * nu: h_propAttempted_ is set by two
    // different formulas depending on which propagate path ran (repeatSize * 32 on kernel1,
    // repeatSize * propIterations on kernel2), so that product is a no-op round trip in one branch
    // and overstates by up to 32x in the other.
    h_candidatesPreGate_ = h_frontierNextSize_;

    // Per-iteration accumulators the accept passes fill. regionCovered MUST be cleared here rather
    // than in propagate: it is written by accept pass 2 and read by Part B, both inside this call.
    thrust::fill(d_doorCounts_.begin(), d_doorCounts_.end(), 0ULL);
    thrust::fill(d_ordHistogram_.begin(), d_ordHistogram_.end(), 0);
    thrust::fill(d_optimalCount_.begin(), d_optimalCount_.end(), 0u);
    thrust::fill(d_regionCovered_.begin(), d_regionCovered_.end(), false);

    // --- Collision-free fraction. Diagnostic only now that nothing consumes it, but it is the only
    // remaining window into propagation efficiency and it costs two reductions over NUM_R1_REGIONS.
    // REDUCED INTO 64-BIT: counterArray is int, and summed over every region across a full run it
    // reaches ~1e9 -- a 32-bit accumulator would silently overflow at larger MAX_ITER. ---
    long long totAll   = thrust::reduce(graph_.d_counterArray_.begin(), graph_.d_counterArray_.end(), (long long)0);
    long long validAll = thrust::reduce(graph_.d_validCounterArray_.begin(), graph_.d_validCounterArray_.end(), (long long)0);
    h_globalCollisionFrac_ = (totAll > 0) ? float(totAll - validAll) / float(totAll) : 0.0f;

    // --- CleanCost's GLOBAL cost scale: (mean cost over all valid samples) - (min over regions).
    // It is the denominator of a candidate's distance, which is what makes "distance 0" a
    // scale-free statement instead of one in raw cost units. Unreached regions contribute sum = 0,
    // cnt = 0, min = MAX_FLOAT, so all three reductions are correct with no masking. Three passes
    // over NUM_R1_REGIONS against the two existing MAX_TREE_SIZE scans -- negligible. Must run
    // after propagate (which fills the arrays) and before accept pass 1. ---
    float sumAll = thrust::reduce(d_sumCostsR1_.begin(), d_sumCostsR1_.end(), 0.0f);
    int   cntAll = thrust::reduce(d_cntCostsR1_.begin(), d_cntCostsR1_.end(), 0);
    float minAll = thrust::reduce(d_minCostsR1_.begin(), d_minCostsR1_.end(), MAX_FLOAT, thrust::minimum<float>());
    h_costScale_ = (cntAll > 0 && minAll < MAX_FLOAT) ? (sumAll / (float)cntAll - minAll) : 0.0f;

    h_optimalCount_ = 0;
    h_ordCutoff_    = 0;
    h_pBoundary_    = 0.0f;
    for(int i = 0; i < CS_ORD_BUCKETS; i++) h_ordHistogram_[i] = 0;

    // ================================================================================
    // THE ADMISSION DECISION, IN TWO PASSES. Guard the launches: iDivUp(0, block) is 0 blocks,
    // which is cudaErrorInvalidConfiguration.
    // ================================================================================
    if(h_frontierNextSize_ > 0)
        {
            // --- Pass 1: measure. Fills candDistance, the ordinality histogram and the optimal
            // count; stamps no door, because the cutoff is not known yet. ---
            CountingStars_acceptPass1_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
              d_activeFrontierIdxs_ptr_, h_frontierNextSize_,
              d_minCostsR1_ptr_, d_frontierNextXR1s_ptr_, d_unexploredSampleCosts_ptr_,
              d_regionNodeCount_ptr_, h_costScale_, h_costAccept_,
              d_candDistance_ptr_, d_ordHistogram_ptr_, d_optimalCount_ptr_);

            cudaMemcpy(&h_optimalCount_, d_optimalCount_ptr_, sizeof(uint), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ordHistogram_, d_ordHistogram_ptr_, CS_ORD_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);

            // --- SOLVE THE CUTOFF. This is the exclusive scan, done on 256 host ints because that
            // is cheaper than launching a kernel to scan them and copying the answer back.
            //
            // The optimal door has already spent optimalCount of the budget and is UNCAPPED, so
            // `remaining` can be zero -- at which point freshness admits nothing, which is the
            // correct answer and not a degenerate one.
            //
            // X is the number of freshest nodes to take. `cutoff` is the bucket the X-th of them
            // falls in, and `pBoundary` is the fraction of that bucket needed to reach exactly X.
            // Everything strictly below the cutoff is admitted whole.
            //
            // If the loop never breaks, the whole candidate pool is fresher than X demands: cutoff
            // lands at CS_ORD_BUCKETS, which no clamped ordinality can equal, so every non-optimal
            // candidate passes `ord < cutoff`. That is the intended saturation, not an overrun. ---
            float remaining = fmaxf(0.0f, float(h_goalFrontierSize_) - float(h_optimalCount_));
            float X         = h_exploreFrac_ * remaining;

            h_ordCutoff_ = CS_ORD_BUCKETS;
            h_pBoundary_ = 0.0f;
            float acc    = 0.0f;
            for(int k = 0; k < CS_ORD_BUCKETS; k++)
                {
                    float h = float(h_ordHistogram_[k]);
                    if(acc + h >= X)
                        {
                            h_ordCutoff_ = k;
                            h_pBoundary_ = (h > 0.0f) ? fminf(1.0f, fmaxf(0.0f, (X - acc) / h)) : 0.0f;
                            break;
                        }
                    acc += h;
                }

            // --- Pass 2: decide. The only door writer, and the only place frontierNext is cleared. ---
            CountingStars_acceptPass2_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
              d_activeFrontierIdxs_ptr_, h_frontierNextSize_,
              d_frontierNextXR1s_ptr_, d_regionNodeCount_ptr_,
              d_candDistance_ptr_, d_frontierNext_ptr_, d_candDoor_ptr_,
              d_regionCovered_ptr_, d_randomSeeds_ptr_,
              h_ordCutoff_, h_pBoundary_, h_costAccept_,
              d_doorCounts_ptr_);
        }

    // --- Re-scan after the accept passes. The trailing-element correction matters: a candidate
    // landing in the last slot is otherwise dropped from the count. ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontierNext_[MAX_TREE_SIZE - 1]) ? ++h_frontierNextSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Check tree capacity ---
    if(h_treeSize_ + h_frontierNextSize_ >= MAX_TREE_SIZE)
        {
            h_propIterations_ = 0;
            return;
        }

    // ================================================================================
    // WHAT THE BUDGET HAS LEFT, solved on the host so the kernel spends a known quantity rather
    // than discovering one. THIS IS THE WHOLE POINT OF THE DESIGN: the frontier is
    //
    //     admitted  +  guaranteed  +  E[draw]   =   max(B, admitted + guaranteed)
    //
    // by construction, not by feedback. When admitted + guaranteed already exceeds B the draw
    // contributes nothing and B becomes a soft floor -- which is exactly the case the header
    // documents for B <= NUM_R1_REGIONS.
    // ================================================================================
    // With cost acceptance off the guarantee arm never fires, so the reduction has nothing to
    // count -- skip it rather than sweeping NUM_R1_REGIONS to prove it is 0. regionCovered is
    // all-false in that case too (only the optimal door writes it), so the predicate would return
    // the full active-region count and be WRONG, not merely wasteful.
    int guaranteed = 0;
    if(h_costAccept_)
        {
            CountingStars_UncoveredBest uncovered{d_bestNodeIdxPerR1_ptr_, d_regionCovered_ptr_};
            guaranteed = thrust::transform_reduce(thrust::device,
                                                  thrust::counting_iterator<int>(0),
                                                  thrust::counting_iterator<int>(NUM_R1_REGIONS),
                                                  uncovered, 0, thrust::plus<int>());
        }
    h_guaranteedReact_ = (uint)guaranteed;

    float reactBudget = fmaxf(0.0f, float(h_goalFrontierSize_) - float(h_frontierNextSize_) - float(guaranteed));
    // Clamped at 1: a remainder above the tree size means "take everything", not an invalid p.
    float pReactivate = (h_treeSize_ > 0) ? fminf(1.0f, reactBudget / float(h_treeSize_)) : 0.0f;

    // --- THE FAN-OUT SPLIT. blockBudget = maxBlocks * B is the DESIGN budget; optimal nodes take
    // maxBlocks each off the top and everyone else divides the rest evenly.
    //
    // It is deliberately non-binding in the nominal case: frontierPlan is max(B, admitted +
    // guaranteed), so when the frontier lands at B the divisor is exactly B - optimalCount and
    // otherBlocks comes out at maxBlocks. It bites on an OVERSHOOT, where the optimal door keeps
    // its full boost and the overshoot is paid for by everyone else -- and it collapses to the
    // rep >= 1 floor when the optimal count alone exceeds B, which is the honest answer there.
    //
    // The buffer bound (blockCeiling, in propagateFrontier) is a SEPARATE constraint and both must
    // hold; this one does not replace it. ---
    float frontierPlan  = float(h_frontierNextSize_) + float(guaranteed) + reactBudget;
    float otherCount    = fmaxf(1.0f, frontierPlan - float(h_optimalCount_));
    long long blockBudget = (long long)h_maxBlocks_ * (long long)h_goalFrontierSize_;
    long long spare       = blockBudget - (long long)h_maxBlocks_ * (long long)h_optimalCount_;
    int otherBlocks = (spare > 0) ? (int)floorf(float(spare) / otherCount) : 1;
    if(otherBlocks < 1) otherBlocks = 1;
    // Never above maxBlocks. Unreachable while frontierPlan >= B (the algebra caps it at maxBlocks
    // exactly), so this is a guard on that invariant rather than a live clamp.
    if(otherBlocks > h_maxBlocks_) otherBlocks = h_maxBlocks_;

    // --- Update Frontier. Part A inserts and stamps blocks; Part B fills the remainder. ---
    CountingStars_updateFrontier_kernel<<<iDivUp(h_frontierNextSize_ + h_treeSize_, h_blockSize_), h_blockSize_>>>(
      d_frontier_ptr_, d_frontierNext_ptr_, d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_goalSample_ptr_, h_treeSize_,
      d_unexploredSamples_ptr_, d_treeSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, d_randomSeeds_ptr_,
      d_candDoor_ptr_, d_nodeDoor_ptr_, d_nodeBlocks_ptr_,
      d_regionNodeCount_ptr_, d_regionCovered_ptr_,
      d_minCostsR1_ptr_, d_treeXR1s_ptr_, d_frontierNextXR1s_ptr_, d_bestNodeIdxPerR1_ptr_,
      d_minCost_ptr_, d_unexploredSampleCosts_ptr_, d_goalSet_ptr_,
      d_iterations_ptr_, h_itr_,
      pReactivate, h_maxBlocks_, otherBlocks, h_costAccept_,
      d_doorCounts_ptr_);

    // --- Read back the door counts. One memcpy for the whole "what built this tree" answer. ---
    cudaMemcpy(h_doorCounts_, d_doorCounts_ptr_, CS_NUM_DOOR_SLOTS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    h_admittedExplore_ = (uint)h_doorCounts_[CS_SLOT_EXPLORE];
    h_admittedCost_    = (uint)h_doorCounts_[CS_SLOT_COST];
    h_reactivated_     = (uint)h_doorCounts_[CS_SLOT_REACT];
    h_reactivatedBest_ = (uint)h_doorCounts_[CS_SLOT_BEST];
    cudaMemcpy(&h_touchedR2_, d_touchedR2Count_ptr_, sizeof(uint), cudaMemcpyDeviceToHost);

    // What the doors actually committed, from the REALISED counts rather than the plan: admissions
    // plus the guaranteed and drawn reactivations that survived Part B's skips. Read against
    // goal_frontier_size -- this is the claim the whole design rests on, in one column.
    h_budgetUsed_ = h_frontierNextSize_ + h_reactivatedBest_ + h_reactivated_;

    // --- Sync goal state ---
    cudaMemcpy(&h_minCost_, d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);

    // --- Update Tree Size ---
    h_treeSize_ += h_frontierNextSize_;
}

/***************************/
/* GET CONTROL PATH TO GOAL */
/***************************/
void CountingStars::getControlPathToGoal()
{
    thrust::exclusive_scan(d_goalSet_.begin(), d_goalSet_.end(), d_goalSetScanIdx_.begin(), 0, thrust::plus<uint>());
    h_solSetSize_ = d_goalSetScanIdx_[MAX_TREE_SIZE - 1];
    (d_goalSet_[MAX_TREE_SIZE - 1]) ? ++h_solSetSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_goalSet_ptr_, d_goalSetScanIdx_ptr_, d_goalSetIdxs_ptr_);

    if(h_solSetSize_ == 0) return;

    CountingStars_getControlPathToGoal_kernel<<<iDivUp(h_solSetSize_, h_blockSize_), h_blockSize_>>>(
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
__global__ void CountingStars_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
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

void CountingStars::writeExecutionTimeToCSV(double time)
{
    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/ExecutionTime");
    filename.str("");
    filename << "Data/ExecutionTime/executionTime.csv";
    writeValueToCSV(time, filename.str());
}
