#include "planners/KinoPaxSTAR.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"

KinoPaxSTAR::KinoPaxSTAR()
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
    d_bestNodeIdxPerR1_       = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_treeXR1s_               = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_frontierNextXR1s_       = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_unexploredSampleCosts_  = thrust::device_vector<float>(MAX_TREE_SIZE);
    d_goalSet_                = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_goalSetIdxs_            = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_goalSetScanIdx_         = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_pruned_                 = thrust::device_vector<bool>(MAX_TREE_SIZE);
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
    d_bestNodeIdxPerR1_ptr_       = thrust::raw_pointer_cast(d_bestNodeIdxPerR1_.data());
    d_treeXR1s_ptr_               = thrust::raw_pointer_cast(d_treeXR1s_.data());
    d_frontierNextXR1s_ptr_       = thrust::raw_pointer_cast(d_frontierNextXR1s_.data());
    d_unexploredSampleCosts_ptr_  = thrust::raw_pointer_cast(d_unexploredSampleCosts_.data());
    d_goalSet_ptr_                = thrust::raw_pointer_cast(d_goalSet_.data());
    d_goalSetIdxs_ptr_            = thrust::raw_pointer_cast(d_goalSetIdxs_.data());
    d_goalSetScanIdx_ptr_         = thrust::raw_pointer_cast(d_goalSetScanIdx_.data());
    d_pruned_ptr_                 = thrust::raw_pointer_cast(d_pruned_.data());
    d_iterations_ptr_             = thrust::raw_pointer_cast(d_iterations_.data());
    d_pathCosts_ptr_              = thrust::raw_pointer_cast(d_pathCosts_.data());
    d_controlPathsToGoal_ptr_     = thrust::raw_pointer_cast(d_controlPathsToGoal_.data());

    cudaMalloc(&d_minCost_ptr_, sizeof(float));

    // Spatial hash grid for fast collision detection
    d_spatialHashGrid_ = createSpatialHashGrid();

    // Host buffer for the best (min-cost) reconstructed trajectory
    h_controlPathsToGoal_ = new float[MAX_ITER * SAMPLE_DIM];

    // PruneKPAX goal-progress gate tunables. maxRegression auto-scales to the workspace
    // (PruneKPAX's Model-3 default of 10 == 0.1*(100-0)); tune per environment as needed.
    h_maxRegression_   = 0.1f * (W_MAX - W_MIN);
    h_explorationBias_ = 0.3f;
    h_goalBias_        = 0.7f;

    h_activeBlockSize_ = 32;

    if(VERBOSE)
        {
            printf("/* Planner Type: KinoPaxSTAR (Hybrid) */\n");
            printf("/* Number of R1 Vertices: %d */\n", NUM_R1_REGIONS);
            printf("/* Number of R2 Vertices: %d */\n", NUM_R2_REGIONS);
            printf("/***************************/\n");
        }
}

KinoPaxSTAR::~KinoPaxSTAR()
{
    destroySpatialHashGrid(d_spatialHashGrid_);
    delete[] h_controlPathsToGoal_;
}

void KinoPaxSTAR::resetPlanner(float* h_initial, float* h_goal)
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
    thrust::fill(d_bestNodeIdxPerR1_.begin(), d_bestNodeIdxPerR1_.end(), -1);
    thrust::fill(d_treeXR1s_.begin(), d_treeXR1s_.end(), 0);
    thrust::fill(d_frontierNextXR1s_.begin(), d_frontierNextXR1s_.end(), 0);
    thrust::fill(d_unexploredSampleCosts_.begin(), d_unexploredSampleCosts_.end(), 0.0f);
    thrust::fill(d_goalSet_.begin(), d_goalSet_.end(), false);
    thrust::fill(d_pruned_.begin(), d_pruned_.end(), false);
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
    // h_propIterations_ == 0, and propagateFrontier only assigns it on the
    // tree-nearly-full path. Without this init it holds indeterminate garbage
    // (matches the fix already in KinoPaxSTARNoPruneNoSpatialHash::resetPlanner).
    h_propIterations_ = 1;

    cudaMemcpy(d_treeSamples_ptr_, h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costToGoal_ptr_, &h_costToGoal_, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pathToGoal_ptr_, &h_pathToGoal_, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minCost_ptr_, &h_minCost_, sizeof(float), cudaMemcpyHostToDevice);

    initializeRandomSeeds(static_cast<unsigned int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
}

void KinoPaxSTAR::plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
{
    cudaEvent_t start, stop, firstSol;
    float milliseconds = 0;
    float firstSolMs   = -1.0f;
    bool  foundFirst   = false;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&firstSol);
    cudaEventRecord(start);

    resetPlanner(h_initial, h_goal);

    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            graph_.updateVertices();
            updateFrontier();

            // Time to first solution: record the instant minCost first becomes finite (a goal node
            // was reached). Recorded on the stream; its elapsed time is read after the sync below.
            if(!foundFirst && h_minCost_ < 0.5f * MAX_FLOAT)
                {
                    cudaEventRecord(firstSol);
                    foundFirst = true;
                }

            // Run to MAX_ITER / tree-full, continuing to improve minCost (no first-solution break).
            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    if(foundFirst) cudaEventElapsedTime(&firstSolMs, start, firstSol);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "KinoPaxSTAR execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_
              << ". Time to first solution: " << (foundFirst ? firstSolMs : -1.0f) << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(firstSol);
}

void KinoPaxSTAR::planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr)
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
    std::cout << "KinoPaxSTAR execution time: " << executionTime << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
}

float KinoPaxSTAR::planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
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
    std::cout << "KinoPaxSTAR execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return h_minCost_;
}

void KinoPaxSTAR::propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount)
{
    // --- Build spatial hash grid for fast collision detection ---
    // MODEL 4 (2D CW) bypasses the spatial hash: its build/query index obs[2]/bb[2], which is out
    // of bounds for W_DIM=2. With only ~10 defenders, the W_DIM-generic brute-force path is used.
#if MODEL != 4
    updateSpatialHashGrid(d_spatialHashGrid_, d_obstacles_ptr, h_obstaclesCount);
#endif
    cudaMemcpy(&h_spatialHashGrid_, d_spatialHashGrid_, sizeof(SpatialHashGrid), cudaMemcpyDeviceToHost);

    // --- Find indices and size of frontier ---
    thrust::exclusive_scan(d_frontier_.begin(), d_frontier_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontier_[MAX_TREE_SIZE - 1]) ? ++h_frontierSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontier_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Build frontier repeat vector ---
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

            KinoPaxSTAR_propagateFrontier_kernel2<<<iDivUp(h_propIterations_ * h_frontierRepeatSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              graph_.d_vertexScoreArray_ptr_, d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              h_propIterations_, graph_.d_minValueInRegion_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_frontierNextXR1s_ptr_, d_unexploredSampleCosts_ptr_, h_spatialHashGrid_);
        }
    else
        {
            KinoPaxSTAR_propagateFrontier_kernel1<<<iDivUp(h_frontierRepeatSize_ * h_activeBlockSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              graph_.d_vertexScoreArray_ptr_, d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              graph_.d_minValueInRegion_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_frontierNextXR1s_ptr_, d_unexploredSampleCosts_ptr_, h_spatialHashGrid_);
        }
}

/***************************/
/* PROPAGATE FRONTIER KERNEL 1 */
/***************************/
// One Block Per Frontier Sample — dual acceptance: best-in-region OR vertex score
__global__ void KinoPaxSTAR_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, int* frontierNextXR1s,
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
#if MODEL == 4
    bool valid                       = propagateAndCheck(s_x0, x1, &randSeed, obstacles, obstaclesCount);
#else
    bool valid                       = propagateAndCheckSpatialHash(s_x0, x1, &randSeed, spatialHashGrid, obstacles, obstaclesCount);
#endif
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
            bool isBest = (cost <= minCostsR1[x1Vertex]);

            // Track 2: Exploration (KPAX vertex score OR new sub-region)
            bool acceptedByExploration = (curand_uniform(&randSeed) < vertexScores[x1Vertex])
                                       || (!activeSubVertices[x1SubVertex]);

            if(isBest || acceptedByExploration)
                {
                    unexploredSampleCosts[tid] = cost;
                    frontierNextXR1s[tid]      = x1Vertex;
                    frontierNext[tid]          = true;
                }

            if(activeSubVertices[x1SubVertex] == 0) atomicExch(&activeSubVertices[x1SubVertex], 1);
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* PROPAGATE FRONTIER KERNEL 2 */
/***************************/
// Iterations mode — dual acceptance: best-in-region OR vertex score
__global__ void KinoPaxSTAR_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, int iterations,
                                                   float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, int* frontierNextXR1s,
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
#if MODEL == 4
    bool valid                       = propagateAndCheck(x0, x1, &randSeed, obstacles, obstaclesCount);
#else
    bool valid                       = propagateAndCheckSpatialHash(x0, x1, &randSeed, spatialHashGrid, obstacles, obstaclesCount);
#endif
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
            bool isBest = (cost <= minCostsR1[x1Vertex]);

            // Track 2: Exploration
            bool acceptedByExploration = (curand_uniform(&randSeed) < vertexScores[x1Vertex])
                                       || (activeSubVertices[x1SubVertex] == 0);

            if(isBest || acceptedByExploration)
                {
                    unexploredSampleCosts[tid] = cost;
                    frontierNextXR1s[tid]      = x1Vertex;
                    frontierNext[tid]          = true;
                }

            if(activeSubVertices[x1SubVertex] == 0) atomicExch(&activeSubVertices[x1SubVertex], 1);
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* GOAL-PROGRESS PRUNING KERNEL */
/***************************/
// Min-cost (best-in-region) candidates are exempt and always kept. Non-best candidates
// pass PruneKPAX's greedy-toward-goal probabilistic gate before insertion.
__global__ void KinoPaxSTAR_goalProgressPrune_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                                  float* minCostsR1, int* frontierNextXR1s, float* unexploredSampleCosts,
                                                  float* unexploredSamples, int* unexploredSamplesParentIdxs,
                                                  float* treeSamples, float* xGoal, bool* frontierNext,
                                                  curandState* randomSeeds, float maxRegression, float explorationBias,
                                                  float goalBias)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierNextSize) return;

    int idx    = activeFrontierNextIdxs[tid];
    float cost = unexploredSampleCosts[idx];
    int xR1    = frontierNextXR1s[idx];

    // --- Min-cost candidates are exempt: always inserted (every region best stays in the frontier). ---
    if(cost <= minCostsR1[xR1]) return;

    // --- Non-best candidates: greedy-toward-goal probabilistic gate (xGoal read from global). ---
    float* x1              = &unexploredSamples[idx * SAMPLE_DIM];
    int x0Idx              = unexploredSamplesParentIdxs[idx];
    float distToGoal       = distance(x1, xGoal);
    float parentDistToGoal = distance(&treeSamples[x0Idx * SAMPLE_DIM], xGoal);
    float progressToGoal   = parentDistToGoal - distToGoal;  // positive = moved toward goal

    float normalizedProgress    = (progressToGoal + maxRegression) / (2.0f * maxRegression);
    normalizedProgress          = fminf(fmaxf(normalizedProgress, 0.0f), 1.0f);
    float acceptanceProbability = fminf(explorationBias + goalBias * normalizedProgress, 1.0f);

    curandState seed = randomSeeds[idx];
    bool accept      = curand_uniform(&seed) < acceptanceProbability;
    randomSeeds[idx] = seed;

    if(!accept) frontierNext[idx] = false;
}

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// Hybrid: adds new frontier nodes to tree, re-activates best-per-region + probabilistic others
__global__ void
KinoPaxSTAR_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               uint* activeFrontierRepeatCount, int* validVertexCounter, curandState* randomSeeds,
                               float* vertexScores, float fAccept,
                               float maxRegression, float explorationBias, float goalBias,
                               float* minCostsR1, int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
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

            // Record birth iteration for EVERY inserted node (not only goal nodes), so
            // tree-growth tooling can filter/animate the tree by iteration.
            iterations[x1TreeIdx] = iteration;

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
                    goalSet[x1TreeIdx]  = true;
                    frontier[x1TreeIdx] = false;
                }
        }

    // --- Part B: Re-activate existing tree nodes ---
    else if(tid < frontierNextSize + treeSize)
        {
            int treeIdx = tid - frontierNextSize;
            if(goalSet[treeIdx] || pruned[treeIdx]) return;

            int xR1 = treeXR1s[treeIdx];

            // GUARANTEE: Best node per region is ALWAYS in the frontier
            if(treeIdx == bestNodeIdxPerR1[xR1])
                {
                    frontier[treeIdx]                  = true;
                    activeFrontierRepeatCount[treeIdx] = 1;
                    return;
                }

            // REACTIVATION: Non-best nodes re-enter via PruneKPAX goal-progress
            // probability of acceptance, blended with the Syclop exploration score.
            if(frontier[treeIdx] == 0)
                {
                    float* xNode            = &treeSamples[treeIdx * SAMPLE_DIM];
                    int    parentIdx        = treeSamplesParentIdxs[treeIdx];
                    float  distToGoal       = distance(xNode, s_xGoal);
                    float  parentDistToGoal = (parentIdx >= 0)
                                                ? distance(&treeSamples[parentIdx * SAMPLE_DIM], s_xGoal)
                                                : distToGoal;  // root: no progress
                    float  progressToGoal   = parentDistToGoal - distToGoal;

                    float normalizedProgress    = (progressToGoal + maxRegression) / (2.0f * maxRegression);
                    normalizedProgress          = fminf(fmaxf(normalizedProgress, 0.0f), 1.0f);
                    float acceptanceProbability = fminf(explorationBias + goalBias * normalizedProgress, 1.0f);

                    // Blend goal-progress PoA with the Syclop exploration weighting
                    float reactivationProb = acceptanceProbability * (vertexScores[xR1] + fAccept);

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

void KinoPaxSTAR::updateFrontier()
{
    // --- Find indices and size of the next frontier ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Goal-progress admission gate: min-cost candidates exempt; non-best pass the greedy-goal roll ---
    KinoPaxSTAR_goalProgressPrune_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
      d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_minCostsR1_ptr_, d_frontierNextXR1s_ptr_,
      d_unexploredSampleCosts_ptr_, d_unexploredSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_,
      d_treeSamples_ptr_, d_goalSample_ptr_, d_frontierNext_ptr_, d_randomSeeds_ptr_,
      h_maxRegression_, h_explorationBias_, h_goalBias_);

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

    // --- Compute fAccept (KPAX re-activation boost) ---
    float treeAddSize = 1 - (float(h_treeSize_ + h_frontierNextSize_) / (MAX_TREE_SIZE));
    h_fAccept_        = (h_itr_ * EPSILON) * pow(treeAddSize, 5);

    // --- Update Frontier ---
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    KinoPaxSTAR_updateFrontier_kernel<<<iDivUp(h_frontierNextSize_ + h_treeSize_, h_blockSize_), h_blockSize_>>>(
      d_frontier_ptr_, d_frontierNext_ptr_, d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_goalSample_ptr_, h_treeSize_,
      d_unexploredSamples_ptr_, d_treeSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, d_activeFrontierRepeatCount_ptr_, graph_.d_validCounterArray_ptr_, d_randomSeeds_ptr_,
      graph_.d_vertexScoreArray_ptr_, h_fAccept_,
      h_maxRegression_, h_explorationBias_, h_goalBias_,
      d_minCostsR1_ptr_, d_treeXR1s_ptr_, d_frontierNextXR1s_ptr_, d_bestNodeIdxPerR1_ptr_,
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
void KinoPaxSTAR::getControlPathToGoal()
{
    thrust::exclusive_scan(d_goalSet_.begin(), d_goalSet_.end(), d_goalSetScanIdx_.begin(), 0, thrust::plus<uint>());
    h_solSetSize_ = d_goalSetScanIdx_[MAX_TREE_SIZE - 1];
    (d_goalSet_[MAX_TREE_SIZE - 1]) ? ++h_solSetSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_goalSet_ptr_, d_goalSetScanIdx_ptr_, d_goalSetIdxs_ptr_);

    if(h_solSetSize_ == 0) return;

    KinoPaxSTAR_getControlPathToGoal_kernel<<<iDivUp(h_solSetSize_, h_blockSize_), h_blockSize_>>>(
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
__global__ void KinoPaxSTAR_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
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

void KinoPaxSTAR::writeExecutionTimeToCSV(double time)
{
    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/ExecutionTime");
    filename.str("");
    filename << "Data/ExecutionTime/executionTime.csv";
    writeValueToCSV(time, filename.str());
}
