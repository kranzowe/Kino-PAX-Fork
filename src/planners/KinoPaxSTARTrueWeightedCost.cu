// KinoPaxSTARTrueWeightedCost -- KinoPaxSTARcostprune with a WEIGHTED-SUM acceptance rule.
//
// costprune combines the two acceptance probabilities multiplicatively (costProb * syclop). That
// collapses to zero in exactly the cells that hold a narrow passage: Graph.cu:249 drives the
// Syclop score down quartically where the valid-sample fraction is low, and multiplying by
// costProb <= 1 destroys even the additive fAccept term that KPAX relies on to keep resampling
// there. This variant replaces the product with
//
//     P_combined = min(1, w*P_syclop + (1-w)*P_cost + P_floor)
//
// applied at BOTH acceptance points (the insertion gate and the reactivation branch), where
// P_syclop = vertexScores[xR1] + fAccept is the full KPAX rule and P_floor is fixed. One knob:
// w = 1 reproduces KPAX's acceptance, w = 0 is pure cost-greedy.
//
// P_cost is costProbExp (helper.cuh), not costKeepProb: exp(-k*(cost-m)/(mean-m)) is exactly 1 at
// the region min AND has a real gradient across the whole range, where min(1,(mean/cost)^k) is
// pinned at 1 for every cost at or below the mean. It carries no floor -- P_floor is added once,
// here, in weightedAccept().
//
// Carries NO retroactive pruning: see KinoPaxSTARTrueWeightedCost for the cost-guarded version.
// KinoPaxSTARTrueWeightedCost -- KinoPaxSTARWeightedCost plus COST-GUARDED retroactive pruning.
//
// Prunes a node that was admitted BECAUSE it was its region's cheapest, once it is no longer the
// cheapest. Nodes the Syclop exploration roll admitted are never touched -- that guard is the
// whole difference from the earlier KinoPaxSTARNoPruneAncestor, which applied the same test to
// every node in the tree and consequently froze the entire exploration population on the first
// pruning pass (Part B returns early on pruned[], and the dormancy branch only rehabilitates
// nodes that have become region-best, which an explorer never does).
//
// No goal-bias acceptance anywhere in this file -- that lives only in KinoPaxSTAR.
#include "planners/KinoPaxSTARTrueWeightedCost.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"

KinoPaxSTARTrueWeightedCost::KinoPaxSTARTrueWeightedCost()
{
    graph_ = Graph(W_SIZE);

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
    d_maxCostsR1_             = thrust::device_vector<float>(NUM_R1_REGIONS);
    d_sumCostsR1_             = thrust::device_vector<float>(NUM_R1_REGIONS);
    d_cntCostsR1_             = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_bestNodeIdxPerR1_       = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_treeXR1s_               = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_frontierNextXR1s_       = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_unexploredSampleCosts_  = thrust::device_vector<float>(MAX_TREE_SIZE);
    d_goalSet_                = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_goalSetIdxs_            = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_goalSetScanIdx_         = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_pruned_                 = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_frontierNextAdmitBest_  = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_treeAdmitBest_          = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_treeInactiveIterations_ = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_ancestorBad_            = thrust::device_vector<bool>(MAX_TREE_SIZE);
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
    d_maxCostsR1_ptr_             = thrust::raw_pointer_cast(d_maxCostsR1_.data());
    d_sumCostsR1_ptr_             = thrust::raw_pointer_cast(d_sumCostsR1_.data());
    d_cntCostsR1_ptr_             = thrust::raw_pointer_cast(d_cntCostsR1_.data());
    d_bestNodeIdxPerR1_ptr_       = thrust::raw_pointer_cast(d_bestNodeIdxPerR1_.data());
    d_treeXR1s_ptr_               = thrust::raw_pointer_cast(d_treeXR1s_.data());
    d_frontierNextXR1s_ptr_       = thrust::raw_pointer_cast(d_frontierNextXR1s_.data());
    d_unexploredSampleCosts_ptr_  = thrust::raw_pointer_cast(d_unexploredSampleCosts_.data());
    d_goalSet_ptr_                = thrust::raw_pointer_cast(d_goalSet_.data());
    d_goalSetIdxs_ptr_            = thrust::raw_pointer_cast(d_goalSetIdxs_.data());
    d_goalSetScanIdx_ptr_         = thrust::raw_pointer_cast(d_goalSetScanIdx_.data());
    d_pruned_ptr_                 = thrust::raw_pointer_cast(d_pruned_.data());
    d_frontierNextAdmitBest_ptr_  = thrust::raw_pointer_cast(d_frontierNextAdmitBest_.data());
    d_treeAdmitBest_ptr_          = thrust::raw_pointer_cast(d_treeAdmitBest_.data());
    d_treeInactiveIterations_ptr_ = thrust::raw_pointer_cast(d_treeInactiveIterations_.data());
    d_ancestorBad_ptr_            = thrust::raw_pointer_cast(d_ancestorBad_.data());
    d_iterations_ptr_             = thrust::raw_pointer_cast(d_iterations_.data());
    d_pathCosts_ptr_              = thrust::raw_pointer_cast(d_pathCosts_.data());
    d_controlPathsToGoal_ptr_     = thrust::raw_pointer_cast(d_controlPathsToGoal_.data());

    cudaMalloc(&d_minCost_ptr_, sizeof(float));

    // Spatial hash grid for fast collision detection
    d_spatialHashGrid_ = createSpatialHashGrid();

    // Host buffer for the best (min-cost) reconstructed trajectory
    h_controlPathsToGoal_ = new float[MAX_ITER * SAMPLE_DIM];

    // WeightedCost tunables. P_floor defaults to EPSILON, matching the additive floor already
    // baked into the Syclop score itself (Graph.cu: vertexScores = EPSILON + score/total).
    h_costWeight_   = 0.5f;
    h_costPruneExp_ = 1.0f;
    h_probFloor_    = EPSILON;
    // Only governs the propagate-time dual-track acceptance; w replaces it everywhere else.
    h_acceptCap_    = 0.1f;

    // Cost pruning: OFF by default, so the fresh class reproduces KinoPaxSTARWeightedCost exactly.
    h_ancestorPrune_     = 0;
    h_dormancyThreshold_ = 5;
    h_ancestorTol_       = 0.0f;

    h_activeBlockSize_ = 32;

    if(VERBOSE)
        {
            printf("/* Planner Type: KinoPaxSTARTrueWeightedCost (Hybrid) */\n");
            printf("/* Number of R1 Vertices: %d */\n", NUM_R1_REGIONS);
            printf("/* Number of R2 Vertices: %d */\n", NUM_R2_REGIONS);
            printf("/***************************/\n");
        }
}

KinoPaxSTARTrueWeightedCost::~KinoPaxSTARTrueWeightedCost()
{
    destroySpatialHashGrid(d_spatialHashGrid_);
    delete[] h_controlPathsToGoal_;
}

void KinoPaxSTARTrueWeightedCost::resetPlanner(float* h_initial, float* h_goal)
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
    thrust::fill(graph_.d_counterArray_.begin(), graph_.d_counterArray_.end(), 0);
    thrust::fill(graph_.d_validCounterArray_.begin(), graph_.d_validCounterArray_.end(), 0);

    // Tree state
    thrust::fill(d_treeSamples_.begin(), d_treeSamples_.end(), 0.0f);
    thrust::fill(d_treeSamplesParentIdxs_.begin(), d_treeSamplesParentIdxs_.end(), -1);
    thrust::fill(d_treeSampleCosts_.begin(), d_treeSampleCosts_.end(), 0.0f);
    thrust::fill(d_frontier_.begin(), d_frontier_.begin() + 1, true);

    // KinoPaxPlus optimization state
    thrust::fill(d_minCostsR1_.begin(), d_minCostsR1_.end(), MAX_FLOAT);
    thrust::fill(d_maxCostsR1_.begin(), d_maxCostsR1_.end(), 0.0f);
    thrust::fill(d_sumCostsR1_.begin(), d_sumCostsR1_.end(), 0.0f);
    thrust::fill(d_cntCostsR1_.begin(), d_cntCostsR1_.end(), 0);
    thrust::fill(d_bestNodeIdxPerR1_.begin(), d_bestNodeIdxPerR1_.end(), -1);
    thrust::fill(d_treeXR1s_.begin(), d_treeXR1s_.end(), 0);
    thrust::fill(d_frontierNextXR1s_.begin(), d_frontierNextXR1s_.end(), 0);
    thrust::fill(d_unexploredSampleCosts_.begin(), d_unexploredSampleCosts_.end(), 0.0f);
    thrust::fill(d_goalSet_.begin(), d_goalSet_.end(), false);
    thrust::fill(d_pruned_.begin(), d_pruned_.end(), false);
    thrust::fill(d_frontierNextAdmitBest_.begin(), d_frontierNextAdmitBest_.end(), false);
    thrust::fill(d_treeAdmitBest_.begin(), d_treeAdmitBest_.end(), false);
    thrust::fill(d_treeInactiveIterations_.begin(), d_treeInactiveIterations_.end(), 0);
    thrust::fill(d_ancestorBad_.begin(), d_ancestorBad_.end(), false);
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

    cudaMemcpy(d_treeSamples_ptr_, h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costToGoal_ptr_, &h_costToGoal_, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pathToGoal_ptr_, &h_pathToGoal_, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minCost_ptr_, &h_minCost_, sizeof(float), cudaMemcpyHostToDevice);

    initializeRandomSeeds(static_cast<unsigned int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
}

void KinoPaxSTARTrueWeightedCost::plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
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
    std::cout << "KinoPaxSTARTrueWeightedCost execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void KinoPaxSTARTrueWeightedCost::planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr)
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
    std::cout << "KinoPaxSTARTrueWeightedCost execution time: " << executionTime << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
}

float KinoPaxSTARTrueWeightedCost::planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
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
    std::cout << "KinoPaxSTARTrueWeightedCost execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return h_minCost_;
}

void KinoPaxSTARTrueWeightedCost::propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount)
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

            KinoPaxSTARTrueWeightedCost_propagateFrontier_kernel2<<<iDivUp(h_propIterations_ * h_frontierRepeatSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              graph_.d_vertexScoreArray_ptr_, d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              h_propIterations_, graph_.d_minValueInRegion_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_maxCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_, d_frontierNextXR1s_ptr_, d_frontierNextAdmitBest_ptr_, d_unexploredSampleCosts_ptr_, h_acceptCap_, h_spatialHashGrid_);
        }
    else
        {
            KinoPaxSTARTrueWeightedCost_propagateFrontier_kernel1<<<iDivUp(h_frontierRepeatSize_ * h_activeBlockSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              graph_.d_vertexScoreArray_ptr_, d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              graph_.d_minValueInRegion_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_maxCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_, d_frontierNextXR1s_ptr_, d_frontierNextAdmitBest_ptr_, d_unexploredSampleCosts_ptr_, h_acceptCap_, h_spatialHashGrid_);
        }
}

/***************************/
/* PROPAGATE FRONTIER KERNEL 1 */
/***************************/
// One Block Per Frontier Sample — dual acceptance: best-in-region OR vertex score
__global__ void KinoPaxSTARTrueWeightedCost_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, float* maxCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, bool* frontierNextAdmitBest,
                                                   float* unexploredSampleCosts, float acceptCap, SpatialHashGrid spatialHashGrid)
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

            // Track 1: Best-in-region (KinoPaxPlus)
            if(minCostsR1[x1Vertex] > cost) atomicMinFloat(&minCostsR1[x1Vertex], cost);
            atomicMaxFloat(&maxCostsR1[x1Vertex], cost);
            atomicAdd(&sumCostsR1[x1Vertex], cost);
            atomicAdd(&cntCostsR1[x1Vertex], 1);
            bool isBest = (cost <= minCostsR1[x1Vertex]);

            // Track 2: Exploration (KPAX vertex score OR new sub-region)
            bool acceptedByExploration = (curand_uniform(&randSeed) < fminf(vertexScores[x1Vertex], acceptCap))
                                       || (!activeSubVertices[x1SubVertex]);

            if(isBest || acceptedByExploration)
                {
                    unexploredSampleCosts[tid] = cost;
                    frontierNextXR1s[tid]      = x1Vertex;
                    frontierNext[tid]          = true;
                    // Admitted ONLY because it was the region minimum? A node the Syclop roll
                    // would have taken anyway is an explorer and must stay immune to cost pruning.
                    // Both flags are computed above regardless, so this touches no RNG state.
                    frontierNextAdmitBest[tid] = isBest && !acceptedByExploration;
                }

            if(activeSubVertices[x1SubVertex] == 0) atomicExch(&activeSubVertices[x1SubVertex], 1);
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* PROPAGATE FRONTIER KERNEL 2 */
/***************************/
// Iterations mode — dual acceptance: best-in-region OR vertex score
__global__ void KinoPaxSTARTrueWeightedCost_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, int iterations,
                                                   float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, float* maxCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, bool* frontierNextAdmitBest,
                                                   float* unexploredSampleCosts, float acceptCap, SpatialHashGrid spatialHashGrid)
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

            // Track 1: Best-in-region
            if(minCostsR1[x1Vertex] > cost) atomicMinFloat(&minCostsR1[x1Vertex], cost);
            atomicMaxFloat(&maxCostsR1[x1Vertex], cost);
            atomicAdd(&sumCostsR1[x1Vertex], cost);
            atomicAdd(&cntCostsR1[x1Vertex], 1);
            bool isBest = (cost <= minCostsR1[x1Vertex]);

            // Track 2: Exploration
            bool acceptedByExploration = (curand_uniform(&randSeed) < fminf(vertexScores[x1Vertex], acceptCap))
                                       || (activeSubVertices[x1SubVertex] == 0);

            if(isBest || acceptedByExploration)
                {
                    unexploredSampleCosts[tid] = cost;
                    frontierNextXR1s[tid]      = x1Vertex;
                    frontierNext[tid]          = true;
                    // Admitted ONLY because it was the region minimum? A node the Syclop roll
                    // would have taken anyway is an explorer and must stay immune to cost pruning.
                    // Both flags are computed above regardless, so this touches no RNG state.
                    frontierNextAdmitBest[tid] = isBest && !acceptedByExploration;
                }

            if(activeSubVertices[x1SubVertex] == 0) atomicExch(&activeSubVertices[x1SubVertex], 1);
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* COST-BASED PRUNING KERNEL */
/***************************/
// Min-cost (best-in-region) candidates are exempt and always kept. Non-best candidates are
// retained with the weighted-sum probability
//     min(1, w*(vertexScore + fAccept) + (1-w)*costProbExp(...) + P_floor)
__global__ void KinoPaxSTARTrueWeightedCost_costPrune_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                                  float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                  int* frontierNextXR1s, float* unexploredSampleCosts,
                                                  bool* frontierNext, curandState* randomSeeds,
                                                  float* vertexScores, float fAccept,
                                                  float costWeight, float costPruneExp, float probFloor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierNextSize) return;

    int idx    = activeFrontierNextIdxs[tid];
    float cost = unexploredSampleCosts[idx];
    int xR1    = frontierNextXR1s[idx];
    float m    = minCostsR1[xR1];

    // --- Min-cost candidates are exempt: always inserted (every region best stays in the frontier).
    // This early return is LOAD-BEARING under the weighted sum and must not be folded into the
    // formula: at cost == m, P_cost == 1 but P_combined == min(1, w*P_syclop + (1-w) + floor),
    // which is below 1 whenever P_syclop < 1 -- so a region best could be rejected. ---
    if(cost <= m) return;

    // --- Non-best candidates: weighted sum of the Syclop and cost probabilities. ---
    float pCost   = costProbExp(m, sumCostsR1[xR1], cntCostsR1[xR1], cost, costPruneExp);
    float pSyclop = vertexScores[xR1] + fAccept;
    float acceptanceProbability = weightedAccept(costWeight, pSyclop, pCost, probFloor);

    curandState seed = randomSeeds[idx];
    bool accept      = curand_uniform(&seed) < acceptanceProbability;
    randomSeeds[idx] = seed;

    if(!accept) frontierNext[idx] = false;
}

/***************************/
/* COST PRUNING KERNEL (guarded to cost-admitted nodes) */
/***************************/
// One thread per tree node, re-run every iteration.
//
// The FIRST line is the fix: a node the Syclop exploration roll admitted is never touched. The
// predecessor (KinoPaxSTARNoPruneAncestor) omitted this guard, and because Syclop-admitted nodes
// are non-minimum by construction, every one of them was tombstoned on the first pass -- and
// Part B returns early on pruned[], so they never reactivated. Exploration died instantly.
//
// After the guard the branches follow KinoPaxPlus's pruningTree_kernel:
//   A. dormancy / un-prune -- a pruned node that is (or becomes) its region's cheapest accrues
//      credit; past dormancyThreshold iterations of that it is resurrected. The only un-prune.
//   B. amnesty             -- once that credit is banked the node is immune while region-best.
//   C. the prune test.
// Falling through C does NOT clear pruned[]: pruning is instantaneous, rehabilitation takes
// dormancyThreshold+1 iterations. That asymmetry is KinoPaxPlus's, preserved.
//
// Mode 2's chain uses one parent lookup rather than an O(depth) walk. bad(a) is MONOTONE --
// minCostsR1 only decreases and a node's cost/region are written once at insertion -- so
// ancestorBad(i) = bad(i) || ancestorBad(parent(i)) is evaluated by a single sticky flag. A
// child's index always exceeds its parent's, so the recurrence is well-founded on index order.
// Within one launch a thread may read a parent flag before the parent updates it, which only
// ever DELAYS a prune; the flags are sticky and the kernel re-runs every iteration.
__global__ void KinoPaxSTARTrueWeightedCost_pruningTree_kernel(int treeSize, int* treeSamplesParentIdxs,
                                                  float* treeSampleCosts, float* minCostsR1, int* treeXR1s,
                                                  bool* admitBest, bool* pruned, bool* ancestorBad,
                                                  uint* inactiveIterations,
                                                  int ancestorPrune, int dormancyThreshold, float ancestorTol)
{
    int treeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(treeIdx >= treeSize) return;

    // --- THE GUARD: cost pruning only ever touches cost-admitted nodes. ---
    if(!admitBest[treeIdx]) return;

    float nodeCost = treeSampleCosts[treeIdx];
    int   nodeR1   = treeXR1s[treeIdx];
    bool  isBest   = (nodeCost <= minCostsR1[nodeR1]);

    // --- A: dormancy accumulation and un-prune ---
    if(pruned[treeIdx] && isBest)
        {
            inactiveIterations[treeIdx]++;
            if(inactiveIterations[treeIdx] > dormancyThreshold) pruned[treeIdx] = false;
            return;
        }

    // --- B: permanent amnesty while region-best ---
    if(inactiveIterations[treeIdx] > dormancyThreshold && isBest) return;

    // --- C: prune test ---
    bool selfBad = (nodeCost > minCostsR1[nodeR1] * (1.0f + ancestorTol));

    if(ancestorPrune == 1)
        {
            // Stale-best: admitted as the region minimum, no longer the region minimum.
            if(selfBad) pruned[treeIdx] = true;
            return;
        }

    // Mode 2: stale-best OR any ancestor stale, memoized.
    if(!ancestorBad[treeIdx])
        {
            int parent = treeSamplesParentIdxs[treeIdx];
            ancestorBad[treeIdx] = selfBad || (parent >= 0 && ancestorBad[parent]);
        }
    if(ancestorBad[treeIdx]) pruned[treeIdx] = true;
}

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// Hybrid: adds new frontier nodes to tree, re-activates best-per-region + probabilistic others
__global__ void
KinoPaxSTARTrueWeightedCost_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               uint* activeFrontierRepeatCount, int* validVertexCounter, curandState* randomSeeds,
                               float* vertexScores, float fAccept,
                               float acceptCap, float costWeight, float costPruneExp, float probFloor,
                               float* minCostsR1, float* maxCostsR1, float* sumCostsR1, int* cntCostsR1,
                               int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
                               bool* frontierNextAdmitBest, bool* treeAdmitBest,
                               float* minCost, float* unexploredSampleCosts, bool* goalSet, bool* pruned,
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
            treeAdmitBest[x1TreeIdx]   = frontierNextAdmitBest[x1UnexploredIdx];

            // Always add to frontier (passed dual filter + pruning)
            frontier[x1TreeIdx] = true;

            // Repeat count based on region exploration (KPAX logic)
            if(validVertexCounter[xR1] < 10)
                activeFrontierRepeatCount[x1TreeIdx] = 15;
            else
                activeFrontierRepeatCount[x1TreeIdx] = 1;

            // Update best-node index if this is the new region best
            if(cost <= minCostsR1[xR1])
                atomicExch(&bestNodeIdxPerR1[xR1], x1TreeIdx);

            // Goal criteria check — accumulate goal nodes into goalSet; the min-cost
            // path is reconstructed afterwards by getControlPathToGoal.
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

            // GUARANTEE: Best node per region is ALWAYS in the frontier.
            // Checked BEFORE pruned[] so a region best cannot lose its guaranteed slot to a
            // stale ancestor-pruning tombstone.
            if(treeIdx == bestNodeIdxPerR1[xR1])
                {
                    frontier[treeIdx]                  = true;
                    activeFrontierRepeatCount[treeIdx] = 1;
                    return;
                }

            if(pruned[treeIdx]) return;

            // REACTIVATION: weighted sum of the Syclop and cost probabilities -- the same rule
            // the insertion gate uses. At w = 1 this is min(1, vertexScore + fAccept + P_floor),
            // i.e. KPAX's rule, which is the control arm for the exploration-speed question.
            if(frontier[treeIdx] == 0)
                {
                    float pCost   = costProbExp(minCostsR1[xR1], sumCostsR1[xR1], cntCostsR1[xR1],
                                                treeSampleCosts[treeIdx], costPruneExp);
                    float pSyclop = vertexScores[xR1] + fAccept;
                    float reactivationProb = weightedAccept(costWeight, pSyclop, pCost, probFloor);

                    curandState seed = randomSeeds[treeIdx];
                    if(curand_uniform(&seed) < reactivationProb)
                        {
                            frontier[treeIdx]                  = true;
                            activeFrontierRepeatCount[treeIdx] = 1;
                        }
                    randomSeeds[treeIdx] = seed;
                }
        }
}

void KinoPaxSTARTrueWeightedCost::updateFrontier()
{
    // --- Cost pruning: retroactively tombstone cost-admitted nodes that went stale ---
    // Runs on the pre-insertion tree so it sees the minCostsR1 propagateFrontier just lowered.
    // Skipped entirely when off, so the control arm carries no overhead.
    if(h_ancestorPrune_ != 0 && h_treeSize_ > 0)
        {
            KinoPaxSTARTrueWeightedCost_pruningTree_kernel<<<iDivUp(h_treeSize_, h_blockSize_), h_blockSize_>>>(
              h_treeSize_, d_treeSamplesParentIdxs_ptr_, d_treeSampleCosts_ptr_, d_minCostsR1_ptr_,
              d_treeXR1s_ptr_, d_treeAdmitBest_ptr_, d_pruned_ptr_, d_ancestorBad_ptr_,
              d_treeInactiveIterations_ptr_,
              h_ancestorPrune_, h_dormancyThreshold_, h_ancestorTol_);
        }

    // --- Find indices and size of the next frontier ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Compute fAccept (KPAX re-activation boost) ---
    // Moved AHEAD of the admission gate: the weighted-sum rule needs P_syclop = vertexScore +
    // fAccept at both acceptance points, and the gate is the first of them. Consequence: treeAddSize
    // is based on the PRE-gate frontierNextSize rather than the post-gate one, so fAccept comes out
    // marginally smaller than in KinoPaxSTARcostprune. Using one value for the whole iteration is
    // also more consistent than computing it between the two uses.
    float treeAddSize = 1 - (float(h_treeSize_ + h_frontierNextSize_) / (MAX_TREE_SIZE));
    h_fAccept_        = (h_itr_ * EPSILON) * pow(treeAddSize, 5);

    // --- Admission gate: min-cost candidates exempt; non-best kept via the weighted sum ---
    KinoPaxSTARTrueWeightedCost_costPrune_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
      d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
      d_frontierNextXR1s_ptr_,
      d_unexploredSampleCosts_ptr_, d_frontierNext_ptr_, d_randomSeeds_ptr_,
      graph_.d_vertexScoreArray_ptr_, h_fAccept_,
      h_costWeight_, h_costPruneExp_, h_probFloor_);

    // --- Re-scan after pruning ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Check tree capacity ---
    if(h_treeSize_ + h_frontierNextSize_ >= MAX_TREE_SIZE)
        {
            h_propIterations_ = 0;
            return;
        }

    // --- Update Frontier (fAccept already computed above, before the gate) ---
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    KinoPaxSTARTrueWeightedCost_updateFrontier_kernel<<<iDivUp(h_frontierNextSize_ + h_treeSize_, h_blockSize_), h_blockSize_>>>(
      d_frontier_ptr_, d_frontierNext_ptr_, d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_goalSample_ptr_, h_treeSize_,
      d_unexploredSamples_ptr_, d_treeSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, d_activeFrontierRepeatCount_ptr_, graph_.d_validCounterArray_ptr_, d_randomSeeds_ptr_,
      graph_.d_vertexScoreArray_ptr_, h_fAccept_,
      h_acceptCap_, h_costWeight_, h_costPruneExp_, h_probFloor_,
      d_minCostsR1_ptr_, d_maxCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
      d_treeXR1s_ptr_, d_frontierNextXR1s_ptr_, d_bestNodeIdxPerR1_ptr_,
      d_frontierNextAdmitBest_ptr_, d_treeAdmitBest_ptr_,
      d_minCost_ptr_, d_unexploredSampleCosts_ptr_, d_goalSet_ptr_, d_pruned_ptr_,
      d_iterations_ptr_, h_itr_);

    // --- Sync goal state ---
    cudaMemcpy(&h_minCost_, d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);

    // --- Update Tree Size ---
    h_treeSize_ += h_frontierNextSize_;
}

/***************************/
/* GET CONTROL PATH TO GOAL */
/***************************/
void KinoPaxSTARTrueWeightedCost::getControlPathToGoal()
{
    thrust::exclusive_scan(d_goalSet_.begin(), d_goalSet_.end(), d_goalSetScanIdx_.begin(), 0, thrust::plus<uint>());
    h_solSetSize_ = d_goalSetScanIdx_[MAX_TREE_SIZE - 1];
    (d_goalSet_[MAX_TREE_SIZE - 1]) ? ++h_solSetSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_goalSet_ptr_, d_goalSetScanIdx_ptr_, d_goalSetIdxs_ptr_);

    if(h_solSetSize_ == 0) return;

    KinoPaxSTARTrueWeightedCost_getControlPathToGoal_kernel<<<iDivUp(h_solSetSize_, h_blockSize_), h_blockSize_>>>(
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
__global__ void KinoPaxSTARTrueWeightedCost_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
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

void KinoPaxSTARTrueWeightedCost::writeExecutionTimeToCSV(double time)
{
    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/ExecutionTime");
    filename.str("");
    filename << "Data/ExecutionTime/executionTime.csv";
    writeValueToCSV(time, filename.str());
}
