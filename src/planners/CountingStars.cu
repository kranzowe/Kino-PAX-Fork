// CountingStars -- KinoPAX*, with per-region COUNTS in place of a global acceptance probability.
//
// WHAT IT KEEPS FROM THE STAR LINE. Propagate makes no admission decision; one accept kernel runs
// after it, once the region statistics have converged. That ordering is load-bearing and must not be
// relaxed: minCostsR1 / sumCostsR1 / cntCostsR1 are updated by atomics from the very threads that
// would read them, so a decision taken inside propagate would see a partial mean and two identical
// candidates would draw different answers purely from scheduling.
//
// Propagate does now do COUNTING -- candidate counts, novel-cell claims. That is not a relaxation of
// the rule: counting with atomics is exact and order-independent, and the rule is about STATISTICS
// being mid-flight. Nothing reads minCostsR1 until the launch has finished.
//
// WHY THE PROBABILITY HAD TO GO. COMBO admitted with min(pMax, shape * pTargetAccept), where
// pTargetAccept was solved so the EXPECTED admission count hit a growth target. Two things follow,
// and together they make the rule unable to do the one job it was there for:
//
//   1. The shape is a normalised blend of sigmoids -- neutral 0.5, ceiling 1.0 -- so the very best
//      candidate can be at most 2x as likely to enter as an average one.
//   2. pTargetAccept divides by the MEASURED mean shape. Sharpen the shape, the mean falls, pTarget
//      rises to compensate, and the gain is handed straight back.
//
// So acceptance was a REALLOCATION mechanism at a fixed total. It could not concentrate, and nodes
// came out dense everywhere instead of sparse where it mattered.
//
// WHAT THE PLANNERS THAT WORK ACTUALLY DO.
//   KPAX admits EVERY candidate landing in a virgin R2 sub-region -- no roll, no cap. Its Syclop
//   roll is ~1% in a mature region, two orders of magnitude weaker; the novelty door does nearly all
//   of the work.
//   KinoPaxPlus divides the whole propagation budget over the frontier UNIFORMLY, bf =
//   MAX_TREE_SIZE/(F*32). It has no per-node fan-out weight at all. Its power is that parent-chain
//   pruning keeps F tiny: at F = 10 every node gets 40,000 propagations.
//
// THE THREE COUNTS. explore_count, cost_count and reactivation_count say how many nodes come in by
// each door. Probability survives only as the way a counted quota is filled INSIDE ONE REGION --
// never as a global normalised score. See the header for each.
//
//   COST      cost <= minCostsR1[r]. Quota 1 in v1, which is exactly what the atomicMin in propagate
//             already computes, so the door is free. Load-bearing for optimality convergence.
//   EXPLORE   won the atomicCAS on a virgin R2 cell, then won the region's quota with
//             p = exploreCount / novelCounts[r]. The CAS is a deliberate departure from KPAX, whose
//             non-exclusive claim would let one cell absorb a whole region's quota.
//   REACT     uniform over the tree at p = reactCount / treeSize, so the expected count is exactly
//             reactCount. This REPLACES COMBO's unconditional region-best reactivation, which pinned
//             F >= nActive and therefore pinned propagations-per-node near 32. F is the lever.
//
// FAN-OUT is decided at admission and stored per node; propagateFrontier only totals it and scales
// it to fit the buffer, which keeps it the single writer of activeFrontierRepeatCount and makes
// rep >= 1 structural. Reactivated nodes get 1; explore nodes get a GEOMETRIC decay in their
// region ordinal; cost nodes get maxBlocks/(novel this iteration + 1), so a quiet region optimises
// hard and a busy one does not double-spend.
//
// THE R2 MAPPING IS FIXED HERE AND ONLY HERE. Graph.cu's initializeRegions_kernel does not invert
// getRegion, so its min-corners are wrong and every R2 identity built on them is scrambled. This
// planner carries a corrected copy because the explore door IS the R2 novelty test; Graph.cu is left
// alone so the existing baselines stay comparable. See the header, and check_region_math.py.
//
// Opts into Graph's dynamic score floor (1/N_active rather than a fixed EPSILON); see Graph.cuh.
// Carries NO retroactive pruning.
#include "planners/CountingStars.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"
#include <thrust/transform_reduce.h>

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

    // CountingStars' OWN min-corner table -- see the header for why Graph.cu's is not usable.
    d_minCornerCS_            = thrust::device_vector<float>(NUM_R1_REGIONS * STATE_DIM);

    // Per-R1, reset every iteration. First per-iteration per-region arrays in this repo.
    d_novelCounts_            = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_candCounts_             = thrust::device_vector<int>(NUM_R1_REGIONS);
    // Per-R1, cumulative. atomicAdd's return value is a node's ordinal within its region.
    d_regionNodeCount_        = thrust::device_vector<int>(NUM_R1_REGIONS);
    // Per-node, tree-indexed, written once at admission.
    d_nodeBlocks_             = thrust::device_vector<int>(MAX_TREE_SIZE, 1);
    d_nodeDoor_               = thrust::device_vector<int>(MAX_TREE_SIZE, CS_DOOR_NONE);
    // Per-candidate, unexplored-sample-slot indexed.
    d_candNovel_              = thrust::device_vector<bool>(MAX_TREE_SIZE, false);
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
    d_minCornerCS_ptr_            = thrust::raw_pointer_cast(d_minCornerCS_.data());
    d_novelCounts_ptr_            = thrust::raw_pointer_cast(d_novelCounts_.data());
    d_candCounts_ptr_             = thrust::raw_pointer_cast(d_candCounts_.data());
    d_regionNodeCount_ptr_        = thrust::raw_pointer_cast(d_regionNodeCount_.data());
    d_nodeBlocks_ptr_             = thrust::raw_pointer_cast(d_nodeBlocks_.data());
    d_nodeDoor_ptr_               = thrust::raw_pointer_cast(d_nodeDoor_.data());
    d_candNovel_ptr_              = thrust::raw_pointer_cast(d_candNovel_.data());
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
    // THE THREE COUNTS. Each ramps linearly in u = treeSize / MAX_TREE_SIZE, from its *0 value at
    // an empty tree to its *1 value at a full one. Six numbers, and they are the entire tuning
    // surface -- see the header for why counts replaced the probability.
    // ================================================================================

    // Novelty admissions per R1 region per iteration. Eligibility is winning the atomic claim on a
    // virgin R2 sub-region; this is how many of those winners are kept. Rises SLOWLY -- reach is
    // worth most early, and the door self-limits anyway once a region's cells fill.
    h_exploreCount0_ = 5.0f;
    h_exploreCount1_ = 10.0f;

    // Cost admissions per R1 region per iteration. PINNED AT 1 IN v1, where it is exactly the
    // region-best rule the atomicMin on minCostsR1 already computes -- no new machinery at all.
    // Rises FASTER than explore because optimality matters more as the tree fills, but anything
    // above 1 needs a per-region top-K, which this repo has no primitive for: no sort, no CUB, no
    // per-region multi-slot storage. Until that exists, keep both endpoints at 1.
    h_costCount0_ = 1.0f;
    h_costCount1_ = 1.0f;

    // GLOBAL frontier cap -- the most important knob here, because it sets F and F sets
    // propagations-per-node. Seeded at 0: the frontier is then exactly this iteration's admissions,
    // which is the most KinoPaxPlus-like arm and the natural starting point for the sweep.
    h_reactCount0_ = 0.0f;
    h_reactCount1_ = 0.0f;

    // ---- Fan-out. Blocks a node gets are decided at admission; see the header for the rule. ----
    // 15 is KPAX's number, and rep is a plain COUNT OF BLOCKS with no alignment constraint --
    // repeatInd writes rep integer entries and kernel1 launches one 32-thread block per entry, so a
    // node at 15 gets 15 x 32 = 480 propagations.
    h_maxBlocks_   = 15;
    // 1 = halve at every ordinal (15, 7, 3, 1, 1 ...), the sparsest setting and the closest to
    // KPAX's realised one-shot burst. Larger values stretch the decay toward flat.
    h_fanHalfLife_ = 1;

    // ---- Derived per-iteration scalars. All recomputed before they are read; these are only the
    // values the CSV would show if a run somehow logged iteration 0. ----
    h_exploreCount_        = h_exploreCount0_;
    h_costCount_           = h_costCount0_;
    h_reactCount_          = h_reactCount0_;
    h_admittedExplore_     = 0;
    h_admittedCost_        = 0;
    h_reactivated_         = 0;
    h_blockCeiling_        = 0.0f;
    h_blockScale_          = 1.0f;
    h_globalCollisionFrac_ = 0.1f;
    h_propAttempted_       = 0;
    h_candidatesPreGate_   = 0;
    for(int i = 0; i < CS_NUM_DOOR_SLOTS; i++) h_doorCounts_[i] = 0ULL;

    h_activeBlockSize_ = 32;

    if(VERBOSE)
        {
            printf("/* Planner Type: CountingStars (Hybrid) */\n");
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
    // root still opens wide: d_nodeBlocks_ is filled with h_maxBlocks_ below, and the root's region
    // ordinal is 0, so the geometric ramp hands it the full boost on iteration 1.
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
    // Region node ordinals are CUMULATIVE over a run, so this is the one place they are cleared.
    // Carrying them across a reset would start every region's fan-out ramp already exhausted.
    thrust::fill(d_regionNodeCount_.begin(), d_regionNodeCount_.end(), 0);
    thrust::fill(d_novelCounts_.begin(), d_novelCounts_.end(), 0);
    thrust::fill(d_candCounts_.begin(), d_candCounts_.end(), 0);
    // maxBlocks, not 1: the root is admitted by no door, so nothing else would ever write its count.
    thrust::fill(d_nodeBlocks_.begin(), d_nodeBlocks_.end(), h_maxBlocks_);
    thrust::fill(d_nodeDoor_.begin(), d_nodeDoor_.end(), CS_DOOR_NONE);
    thrust::fill(d_candNovel_.begin(), d_candNovel_.end(), false);
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
    // value to 0 (the sibling STAR planners survive only by lucky layout). COMBO adds more
    // members still, so this assignment matters here at least as much.
    h_propIterations_ = 1;

    // Every derived scalar is reset, not just the ones the constructor happened to set. CleanCost
    // reset NONE of these, so a planner object reused across runs -- which every benchmark does --
    // carried the previous run's final values into iteration 1.
    thrust::fill(d_doorCounts_.begin(), d_doorCounts_.end(), 0ULL);
    for(int i = 0; i < CS_NUM_DOOR_SLOTS; i++) h_doorCounts_[i] = 0ULL;
    h_propAttempted_        = 0;
    h_candidatesPreGate_    = 0;
    h_frontierNextSize_     = 0;
    h_frontierRepeatSize_   = 0;
    h_globalCollisionFrac_  = 0.1f;
    h_exploreCount_         = h_exploreCount0_;
    h_costCount_            = h_costCount0_;
    h_reactCount_           = h_reactCount0_;
    h_admittedExplore_      = 0;
    h_admittedCost_         = 0;
    h_reactivated_          = 0;
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
            graph_.updateVertices();
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
    std::cout << "CountingStars execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return h_minCost_;
}

// Linear ramp between a count's empty-tree and full-tree endpoints.
//
// THIS FUNCTION IS THE GROWTH CONTROLLER'S SEAT. Everything about "how many nodes come in" routes
// through the three counts, and the counts route through here -- so replacing this fixed schedule
// with values derived from a per-iteration target is a change to one function and nothing else. The
// counts go fractional when that happens; the explore door already fills its quota with a
// probability, so it absorbs a fractional quota directly.
static inline float countingStarsRamp(float v0, float v1, float u)
{
    u = fminf(1.0f, fmaxf(0.0f, u));
    return v0 + (v1 - v0) * u;
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
    // frontier -- explore admissions, cost admissions and reactivations alike -- so the total is
    // known before a single block is launched. That makes the ceiling something to SOLVE against
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
            // scale == 1 means the ceiling did not bind. Below 1 says the buffer, not the fan-out
            // rule, is setting how hard nodes expand -- and rep_hi collapsing toward 1 across the
            // board means F itself has eaten the budget, which is a reactCount problem.
            float excess  = float(wantBlocks) - float(h_frontierSize_);
            float allowed = h_blockCeiling_ - float(h_frontierSize_);
            h_blockScale_ = (excess > 0.0f) ? fminf(1.0f, fmaxf(0.0f, allowed / excess)) : 1.0f;

            CountingStars_assignFanout_kernel<<<iDivUp(h_frontierSize_, h_blockSize_), h_blockSize_>>>(
              h_frontierSize_, d_activeFrontierIdxs_ptr_, d_nodeBlocks_ptr_, h_blockScale_,
              d_activeFrontierRepeatCount_ptr_);
        }

    // --- Build frontier repeat vector ---
    // Safety net: any position repeatInd does not write must not expose a stale index from an
    // earlier iteration/cycle. Seeding with 0 (the root) makes a missed slot degrade to a
    // redundant root expansion instead of fathering nodes from uninitialised tree slots. With a
    // consistent repeat count this fill is a no-op, since [0, h_frontierRepeatSize_) is fully written.
    // --- Per-iteration per-region counters, zeroed before propagate fills them. These are the only
    // NUM_R1_REGIONS arrays in the repo that are not cumulative, and getting the reset wrong is
    // silent: novelCounts would keep growing and the explore door's probability would shrink toward
    // zero over a run. ---
    thrust::fill(d_novelCounts_.begin(), d_novelCounts_.end(), 0);
    thrust::fill(d_candCounts_.begin(), d_candCounts_.end(), 0);

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

            CountingStars_propagateFrontier_kernel2<<<iDivUp(h_propIterations_ * h_frontierRepeatSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              h_propIterations_, d_minCornerCS_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
              d_frontierNextXR1s_ptr_, d_candNovel_ptr_, d_candDoor_ptr_,
              d_novelCounts_ptr_, d_candCounts_ptr_,
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
              d_frontierNextXR1s_ptr_, d_candNovel_ptr_, d_candDoor_ptr_,
              d_novelCounts_ptr_, d_candCounts_ptr_,
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
// every collision-free sample is marked and its cost / region / sub-region freshness recorded, and
// the accept kernel decides once the region statistics and vertex scores have converged.
__global__ void CountingStars_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minCorner,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, bool* candNovel, int* candDoor,
                                                   int* novelCounts, int* candCounts,
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

            // --- Region cost statistics. These are what the accept kernel reads once the launch
            // has finished; reading them HERE would see them mid-flight. ---
            if(minCostsR1[x1Vertex] > cost) atomicMinFloat(&minCostsR1[x1Vertex], cost);
            atomicAdd(&sumCostsR1[x1Vertex], cost);
            atomicAdd(&cntCostsR1[x1Vertex], 1);

            atomicAdd(&candCounts[x1Vertex], 1);

            // --- EXCLUSIVE R2 CLAIM. atomicCAS, and the RETURN VALUE IS THE POINT: exactly one
            // thread in the whole launch can turn a given cell from 0 to 1, and that thread is the
            // one candidate this cell will ever make novel.
            //
            // THIS IS A DELIBERATE DEPARTURE FROM KPAX, whose own comment calls its behaviour
            // intentional: it reads the cell and then atomicExch'es it, discarding the return, so
            // EVERY thread landing in a virgin cell reads itself as novel. That is fatal to a quota.
            // 50 candidates in ONE cell would all be eligible, and the region's explore slots would
            // go to 50 nodes in the same sub-region instead of 50 different ones. With the CAS,
            // novelCounts is the number of DISTINCT cells claimed -- which is the thing the quota
            // should be rationing.
            //
            // The marking happens either way, so r2_coverage_pct and the graph's coverage are
            // unaffected; the CAS just also says who did it. ---
            bool novel = (atomicCAS(&activeSubVertices[x1SubVertex], 0, 1) == 0);
            if(novel) atomicAdd(&novelCounts[x1Vertex], 1);

            // --- Record the candidate. No admission decision, no RNG draw. The door is CLEARED
            // rather than left alone: these slots are reused every iteration, and a stale door from
            // an earlier batch would be read by Part A as an admission. ---
            candNovel[tid]             = novel;
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
                                                   int* frontierNextXR1s, bool* candNovel, int* candDoor,
                                                   int* novelCounts, int* candCounts,
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

            // --- EXCLUSIVE R2 CLAIM, exactly one winner per cell (see kernel 1). ---
            bool novel = (atomicCAS(&activeSubVertices[x1SubVertex], 0, 1) == 0);
            if(novel) atomicAdd(&novelCounts[x1Vertex], 1);

            // --- Record the candidate (see kernel 1 for why the door is cleared here). ---
            candNovel[tid]             = novel;
            candDoor[tid]              = CS_DOOR_NONE;
            unexploredSampleCosts[tid] = cost;
            frontierNextXR1s[tid]      = x1Vertex;
            frontierNext[tid]          = true;
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* ACCEPT KERNEL - the ONLY admission decision */
/***************************/
// Runs after propagate has finished, so minCostsR1 is converged rather than mid-flight -- the one
// invariant CleanCost established that this planner keeps. Propagate did the counting; this decides.
//
// TWO DOORS, BOTH COUNTED. Neither consults anything global.
//
//   COST     cost <= minCostsR1[r]                               quota costCount, == 1 in v1
//   EXPLORE  candNovel and rand() < exploreCount / novelCounts[r]
//
// COST TAKES PRECEDENCE, and the else-if matters: a candidate that is both its region's best and a
// novel claimant is recorded as COST and does NOT consume an explore slot. Letting it take both
// would quietly shrink the novelty quota by however many region-bests happened to be novel.
//
// WHY THE EXPLORE DOOR IS STILL A PROBABILITY, when this planner exists to remove one. Because it is
// a different object. pTargetAccept was a GLOBAL scalar dividing a growth target across every
// candidate everywhere, so a node's chance depended on what was happening in regions it had never
// seen. This is per-region and count-calibrated: expected admissions in region r are exactly
// exploreCount, novelCounts[r] is that region's own denominator, and when novelCounts <= exploreCount
// the probability is >= 1 and every claimant is taken. Nothing normalises across regions.
//
// cost_count > 1 would need a per-region top-K, and this repo has no primitive for it: no sort, no
// CUB, no per-region multi-slot storage. The cheapest honest route would be a per-region cost
// threshold slewed toward costCount admissions -- one float per region, no sort -- but that is a
// tracked threshold and belongs in a later version, measured.
__global__ void CountingStars_accept_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                            float* minCostsR1, int* frontierNextXR1s, float* unexploredSampleCosts,
                                            bool* frontierNext, bool* candNovel, int* candDoor,
                                            curandState* randomSeeds,
                                            int* novelCounts, float exploreCount,
                                            unsigned long long* doorCounts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierNextSize) return;

    // ONE indirection, and every array below is indexed by idx -- the unexplored-sample SLOT --
    // never by tid, the compacted position. Indexing by tid would read another candidate's data.
    int   idx  = activeFrontierNextIdxs[tid];
    float cost = unexploredSampleCosts[idx];
    int   xR1  = frontierNextXR1s[idx];

    // --- COST door: the region's new best. LOAD-BEARING for optimality convergence, and free --
    // the atomicMin in propagate has already computed exactly this set. ---
    if(cost <= minCostsR1[xR1])
        {
            candDoor[idx] = CS_DOOR_COST;
            atomicAdd(&doorCounts[CountingStars::CS_SLOT_COST], 1ULL);
            return;
        }

    // --- EXPLORE door: won a virgin R2 cell, then won the region's quota. ---
    if(candNovel[idx])
        {
            int n = novelCounts[xR1];
            // n >= 1 whenever candNovel is set -- this thread's own claim incremented it -- so the
            // guard is belt and braces against a torn read rather than a real case.
            float p = (n > 0) ? (exploreCount / (float)n) : 0.0f;

            curandState seed = randomSeeds[idx];
            bool take        = (p >= 1.0f) || (curand_uniform(&seed) < p);
            randomSeeds[idx] = seed;

            if(take)
                {
                    candDoor[idx] = CS_DOOR_EXPLORE;
                    atomicAdd(&doorCounts[CountingStars::CS_SLOT_EXPLORE], 1ULL);
                    return;
                }
        }

    // --- Rejected. Subtractive, like CleanCost's: propagate set the flag, admission leaves it, and
    // only rejection clears it. ---
    candDoor[idx]     = CS_DOOR_NONE;
    frontierNext[idx] = false;
}

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// Part A inserts this iteration's admitted candidates; Part B draws reactivations. The two run in
// one launch over disjoint index ranges -- Part A owns [treeSize, treeSize + frontierNextSize) and
// Part B owns [0, treeSize) -- so they never contend for a node.
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
                               int* regionNodeCount, int* novelCounts,
                               float* minCostsR1, int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
                               float* minCost, float* unexploredSampleCosts, bool* goalSet,
                               int* iterations, int iteration,
                               float pReactivate, int maxBlocks, int fanHalfLife,
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

            // --- THE NODE'S ORDINAL IN ITS REGION. atomicAdd's RETURN VALUE, which is used nowhere
            // else in this planner and in only one other place in the repo (spatialHash's cell
            // insertion). Cumulative over the run, never reset mid-run, so it is "how many nodes
            // this region has ever taken" -- which is exactly the axis the fan-out ramp decays on. ---
            int ordinal = atomicAdd(&regionNodeCount[xR1], 1);

            // --- FAN-OUT, decided here and read next iteration by propagateFrontier. ---
            int blocks;
            if(door == CS_DOOR_EXPLORE)
                {
                    // GEOMETRIC decay: 15, 7, 3, 1, 1 ... at halfLife 1. A linear ramp
                    // max(maxBlocks - ordinal, 1) would spend 65 blocks on a region's first five
                    // nodes; this spends 27, with almost everything on the first two. KPAX's
                    // realised behaviour is the sharper of the two -- its validVertexCounter is
                    // cumulative and gains ~32 per frontier node per iteration, so a region crosses
                    // `< 10` almost at once and the x15 is a one-shot burst, not a ramp.
                    int shift = (fanHalfLife > 0) ? (ordinal / fanHalfLife) : ordinal;
                    blocks    = (shift < 31) ? (maxBlocks >> shift) : 0;
                }
            else
                {
                    // COST door. A region that is still producing novelty does not also need its
                    // best node expanded hard -- the explore admissions are already covering it. A
                    // QUIET region gets the full boost, which is what turns a stalled region into an
                    // optimisation target instead of dead weight.
                    int novel = novelCounts[xR1];
                    blocks    = maxBlocks / (novel + 1);
                }
            nodeBlocks[x1TreeIdx] = (blocks > 1) ? blocks : 1;

            // Update best-node index if this is the new region best.
            //
            // NOTHING READS THIS TODAY, and that is deliberate rather than an oversight. COMBO used
            // it for the unconditional region-best reactivation that pinned F >= nActive, which is
            // exactly what Part B replaced. It is kept maintained -- one atomicExch on a branch that
            // is already taken -- because it is the table a smarter reactivation needs: "top-K by
            // cost" is a scan of this array, not a sort of the tree. Dropping it now would mean
            // rebuilding it then.
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

    // --- Part B: Re-activate existing tree nodes ---
    else if(tid < frontierNextSize + treeSize)
        {
            int treeIdx = tid - frontierNextSize;
            if(goalSet[treeIdx]) return;
            if(frontier[treeIdx]) return;   // already in the frontier; nothing to draw for

            // --- UNIFORM over the whole tree, at p = reactCount / treeSize, so the EXPECTED number
            // added is exactly reactCount whatever the tree size.
            //
            // THIS REPLACED AN UNCONDITIONAL PER-REGION GUARANTEE, and that is the single most
            // consequential change in the planner. COMBO reactivated every region's best with no
            // roll and no budget, which pinned F >= nActive -- and F is what sets
            // propagations-per-node. KinoPaxPlus divides the whole budget over a frontier it prunes
            // down to a handful of nodes and gets ~40,000 propagations each; a frontier pinned at
            // ~32k regions gets ~32. No fan-out weighting closes a gap like that. Only a smaller F
            // does, and reactCount is the knob that sets it.
            //
            // The cost of losing the guarantee is that a region's best node is no longer certain to
            // be in the frontier every iteration. It is still admitted the moment it is created (the
            // COST door), and it is still drawn here like anything else -- the guarantee is now
            // statistical rather than absolute. Watch first-solution cost for the consequence.
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
                            // One block. A reactivated node is being revisited, not discovered --
                            // the boost belongs to nodes opening new ground.
                            nodeBlocks[treeIdx] = 1;
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

    // Collision-free candidates the accept kernel is about to judge. Captured here because the
    // post-gate re-scan below overwrites h_frontierNextSize_ with the survivors.
    //
    // Do NOT reconstruct it as frontierRepeatSize * 32 * nu: h_propAttempted_ is set by two
    // different formulas depending on which propagate path ran (repeatSize * 32 on kernel1,
    // repeatSize * propIterations on kernel2), so that product is a no-op round trip in one branch
    // and overstates by up to 32x in the other.
    h_candidatesPreGate_ = h_frontierNextSize_;

    thrust::fill(d_doorCounts_.begin(), d_doorCounts_.end(), 0ULL);

    // --- Collision-free fraction. Diagnostic only now that nothing consumes it, but it is the only
    // remaining window into propagation efficiency and it costs two reductions over NUM_R1_REGIONS.
    // REDUCED INTO 64-BIT: counterArray is int, and summed over every region across a full run it
    // reaches ~1e9 -- a 32-bit accumulator would silently overflow at larger MAX_ITER. ---
    long long totAll   = thrust::reduce(graph_.d_counterArray_.begin(), graph_.d_counterArray_.end(), (long long)0);
    long long validAll = thrust::reduce(graph_.d_validCounterArray_.begin(), graph_.d_validCounterArray_.end(), (long long)0);
    h_globalCollisionFrac_ = (totAll > 0) ? float(totAll - validAll) / float(totAll) : 0.0f;

    // ================================================================================
    // THE COUNTS. Three lerps, and this is the entire "how many nodes come in" decision.
    //
    // A growth controller replaces exactly this block: derive the counts from a per-iteration target
    // instead of a schedule, and nothing downstream changes. That is the architectural claim, and it
    // holds only while every door routes through a count -- the moment any door consults a global
    // normalised score again, it stops holding.
    // ================================================================================
    float u = fminf(1.0f, fmaxf(0.0f, float(h_treeSize_) / float(MAX_TREE_SIZE)));
    h_exploreCount_ = countingStarsRamp(h_exploreCount0_, h_exploreCount1_, u);
    h_costCount_    = countingStarsRamp(h_costCount0_, h_costCount1_, u);
    h_reactCount_   = countingStarsRamp(h_reactCount0_, h_reactCount1_, u);

    // --- THE admission decision. Two doors, both counted; see the kernel. ---
    // Guard the launch: iDivUp(0, block) is 0 blocks, which is cudaErrorInvalidConfiguration.
    if(h_frontierNextSize_ > 0)
        {
            CountingStars_accept_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
              d_activeFrontierIdxs_ptr_, h_frontierNextSize_,
              d_minCostsR1_ptr_, d_frontierNextXR1s_ptr_, d_unexploredSampleCosts_ptr_,
              d_frontierNext_ptr_, d_candNovel_ptr_, d_candDoor_ptr_,
              d_randomSeeds_ptr_,
              d_novelCounts_ptr_, h_exploreCount_,
              d_doorCounts_ptr_);
        }

    // --- Re-scan after the accept kernel. The trailing-element correction matters: a candidate
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

    // --- Reactivation probability. reactCount is a GLOBAL count, so the per-node probability is
    // reactCount / treeSize and the expected number added is reactCount whatever the tree size.
    // Clamped at 1: a reactCount above the tree size means "take everything", not an invalid p. ---
    float pReactivate = (h_treeSize_ > 0) ? fminf(1.0f, h_reactCount_ / float(h_treeSize_)) : 0.0f;

    // --- Update Frontier. Part A inserts and stamps blocks; Part B draws reactivations. ---
    CountingStars_updateFrontier_kernel<<<iDivUp(h_frontierNextSize_ + h_treeSize_, h_blockSize_), h_blockSize_>>>(
      d_frontier_ptr_, d_frontierNext_ptr_, d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_goalSample_ptr_, h_treeSize_,
      d_unexploredSamples_ptr_, d_treeSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, d_randomSeeds_ptr_,
      d_candDoor_ptr_, d_nodeDoor_ptr_, d_nodeBlocks_ptr_,
      d_regionNodeCount_ptr_, d_novelCounts_ptr_,
      d_minCostsR1_ptr_, d_treeXR1s_ptr_, d_frontierNextXR1s_ptr_, d_bestNodeIdxPerR1_ptr_,
      d_minCost_ptr_, d_unexploredSampleCosts_ptr_, d_goalSet_ptr_,
      d_iterations_ptr_, h_itr_,
      pReactivate, h_maxBlocks_, h_fanHalfLife_,
      d_doorCounts_ptr_);

    // --- Read back the door counts. One memcpy for the whole "what built this tree" answer. ---
    cudaMemcpy(h_doorCounts_, d_doorCounts_ptr_, CS_NUM_DOOR_SLOTS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    h_admittedExplore_ = (uint)h_doorCounts_[CS_SLOT_EXPLORE];
    h_admittedCost_    = (uint)h_doorCounts_[CS_SLOT_COST];
    h_reactivated_     = (uint)h_doorCounts_[CS_SLOT_REACT];

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
