#include "planners/PruneKPAX.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"

PruneKPAX::PruneKPAX()
{
    graph_ = Graph(W_SIZE);

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

    h_activeBlockSize_ = 32;

    // Initialize tunable pruning parameters with default values
    h_maxRegression_ = 10.0f;       // Default: allow 10 units of regression
    h_explorationBias_ = 0.3f;      // Default: 30% base exploration probability
    h_goalBias_ = 0.7f;             // Default: 70% goal-directed bias multiplier

    // Create spatial hash grid for fast collision detection
    d_spatialHashGrid_ = createSpatialHashGrid();

    if(VERBOSE)
        {
            printf("/* Planner Type: KPAX */\n");
            printf("/* Number of R1 Vertices: %d */\n", NUM_R1_REGIONS);
            printf("/* Number of R2 Vertices: %d */\n", NUM_R2_REGIONS);
            printf("/* Spatial Hash Grid: %dx%dx%d cells */\n", GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z);
            printf("/***************************/\n");
        }
}

PruneKPAX::~PruneKPAX()
{
    destroySpatialHashGrid(d_spatialHashGrid_);
}

void PruneKPAX::plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, bool saveTree)
{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- INITIALIZE PruneKPAX ---
    thrust::fill(d_frontier_.begin(), d_frontier_.end(), false);
    thrust::fill(d_frontierNext_.begin(), d_frontierNext_.end(), false);
    thrust::fill(d_activeFrontierIdxs_.begin(), d_activeFrontierIdxs_.end(), 0);
    thrust::fill(d_unexploredSamples_.begin(), d_unexploredSamples_.end(), 0.0f);
    thrust::fill(d_unexploredSamplesParentIdxs_.begin(), d_unexploredSamplesParentIdxs_.end(), -1);
    thrust::fill(d_frontierScanIdx_.begin(), d_frontierScanIdx_.end(), 0);
    thrust::fill(d_frontierRepeatScanIdx_.begin(), d_frontierRepeatScanIdx_.end(), 0);
    thrust::fill(d_goalSample_.begin(), d_goalSample_.end(), 0.0f);
    thrust::fill(graph_.d_activeSubVertices_.begin(), graph_.d_activeSubVertices_.end(), false);
    thrust::fill(graph_.d_vertexScoreArray_.begin(), graph_.d_vertexScoreArray_.end(), 0.0f);
    thrust::fill(graph_.d_counterArray_.begin(), graph_.d_counterArray_.end(), 0);
    thrust::fill(graph_.d_validCounterArray_.begin(), graph_.d_validCounterArray_.end(), 0);
    thrust::fill(d_treeSamples_.begin(), d_treeSamples_.end(), 0.0f);
    thrust::fill(d_treeSamplesParentIdxs_.begin(), d_treeSamplesParentIdxs_.end(), -1);
    thrust::fill(d_treeSampleCosts_.begin(), d_treeSampleCosts_.end(), 0.0f);
    thrust::fill(d_frontier_.begin(), d_frontier_.begin() + 1, true);
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.begin() + 1, 5);  // TODO make this not hard coded to 5.

    h_treeSize_     = 1;
    h_itr_          = 0;
    h_costToGoal_   = 0;
    h_pathToGoal_   = 0;
    h_frontierSize_ = 0;

    cudaMemcpy(d_treeSamples_ptr_, h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costToGoal_ptr_, &h_costToGoal_, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pathToGoal_ptr_, &h_pathToGoal_, sizeof(int), cudaMemcpyHostToDevice);

    initializeRandomSeeds(static_cast<unsigned int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
    // --- END INITIALIZATION ---

    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            graph_.updateVertices();
            updateFrontier();
            if(h_pathToGoal_ != 0)
                {
                    cudaMemcpy(h_controlPathToGoal_, d_controlPathToGoal_ptr_, h_itr_ * SAMPLE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
                    break;
                }
        }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "PruneKPAX execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_ << ". Tree Size: " << h_treeSize_
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void PruneKPAX::planDebug(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- INITIALIZE KPAX ---
    thrust::fill(d_frontier_.begin(), d_frontier_.end(), false);
    thrust::fill(d_frontierNext_.begin(), d_frontierNext_.end(), false);
    thrust::fill(d_activeFrontierIdxs_.begin(), d_activeFrontierIdxs_.end(), 0);
    thrust::fill(d_unexploredSamples_.begin(), d_unexploredSamples_.end(), 0.0f);
    thrust::fill(d_unexploredSamplesParentIdxs_.begin(), d_unexploredSamplesParentIdxs_.end(), -1);
    thrust::fill(d_frontierScanIdx_.begin(), d_frontierScanIdx_.end(), 0);
    thrust::fill(d_frontierRepeatScanIdx_.begin(), d_frontierRepeatScanIdx_.end(), 0);
    thrust::fill(d_goalSample_.begin(), d_goalSample_.end(), 0.0f);
    thrust::fill(graph_.d_activeSubVertices_.begin(), graph_.d_activeSubVertices_.end(), false);
    thrust::fill(graph_.d_vertexScoreArray_.begin(), graph_.d_vertexScoreArray_.end(), 0.0f);
    thrust::fill(graph_.d_counterArray_.begin(), graph_.d_counterArray_.end(), 0);
    thrust::fill(graph_.d_validCounterArray_.begin(), graph_.d_validCounterArray_.end(), 0);
    thrust::fill(d_treeSamples_.begin(), d_treeSamples_.end(), 0.0f);
    thrust::fill(d_treeSamplesParentIdxs_.begin(), d_treeSamplesParentIdxs_.end(), -1);
    thrust::fill(d_treeSampleCosts_.begin(), d_treeSampleCosts_.end(), 0.0f);
    thrust::fill(d_frontier_.begin(), d_frontier_.begin() + 1, true);
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.begin() + 1, 5);

    h_treeSize_     = 1;
    h_itr_          = 0;
    h_costToGoal_   = 0;
    h_pathToGoal_   = 0;
    h_frontierSize_ = 0;

    cudaMemcpy(d_treeSamples_ptr_, h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costToGoal_ptr_, &h_costToGoal_, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pathToGoal_ptr_, &h_pathToGoal_, sizeof(int), cudaMemcpyHostToDevice);

    initializeRandomSeeds(static_cast<unsigned int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
    // --- END INITIALIZATION ---

    // Create timing events for profiling
    cudaEvent_t propagateStart, propagateStop;
    cudaEvent_t updateVerticesStart, updateVerticesStop;
    cudaEvent_t updateFrontierStart, updateFrontierStop;
    
    cudaEventCreate(&propagateStart);
    cudaEventCreate(&propagateStop);
    cudaEventCreate(&updateVerticesStart);
    cudaEventCreate(&updateVerticesStop);
    cudaEventCreate(&updateFrontierStart);
    cudaEventCreate(&updateFrontierStop);
    
    float totalPropagateTime = 0.0f;
    float totalUpdateVerticesTime = 0.0f;
    float totalUpdateFrontierTime = 0.0f;
    float kernelTime = 0.0f;

    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            
            // Time propagateFrontier
            cudaEventRecord(propagateStart);
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            cudaEventRecord(propagateStop);
            cudaEventSynchronize(propagateStop);
            cudaEventElapsedTime(&kernelTime, propagateStart, propagateStop);
            totalPropagateTime += kernelTime;
            
            // Time updateVertices
            cudaEventRecord(updateVerticesStart);
            graph_.updateVertices();
            cudaEventRecord(updateVerticesStop);
            cudaEventSynchronize(updateVerticesStop);
            cudaEventElapsedTime(&kernelTime, updateVerticesStart, updateVerticesStop);
            totalUpdateVerticesTime += kernelTime;
            
            // Time updateFrontier
            cudaEventRecord(updateFrontierStart);
            updateFrontier();
            cudaEventRecord(updateFrontierStop);
            cudaEventSynchronize(updateFrontierStop);
            cudaEventElapsedTime(&kernelTime, updateFrontierStart, updateFrontierStop);
            totalUpdateFrontierTime += kernelTime;
            
            if(h_pathToGoal_ != 0)
                {
                    cudaMemcpy(h_controlPathToGoal_, d_controlPathToGoal_ptr_, h_itr_ * SAMPLE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
                    break;
                }
        }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Print timing breakdown
    std::cout << "\n=== PruneKPAX TIMING BREAKDOWN ===" << std::endl;
    std::cout << "Total execution time: " << milliseconds / 1000.0 << " seconds" << std::endl;
    std::cout << "Iterations: " << h_itr_ << std::endl;
    std::cout << "Tree Size: " << h_treeSize_ << std::endl;
    std::cout << "\nKernel Times (cumulative):" << std::endl;
    std::cout << "  propagateFrontier:  " << totalPropagateTime / 1000.0 << " s (" 
              << (totalPropagateTime / milliseconds) * 100.0 << "%)" << std::endl;
    std::cout << "  updateVertices:     " << totalUpdateVerticesTime / 1000.0 << " s (" 
              << (totalUpdateVerticesTime / milliseconds) * 100.0 << "%)" << std::endl;
    std::cout << "  updateFrontier:     " << totalUpdateFrontierTime / 1000.0 << " s (" 
              << (totalUpdateFrontierTime / milliseconds) * 100.0 << "%)" << std::endl;
    std::cout << "============================\n" << std::endl;
    
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    
    // Clean up timing events
    cudaEventDestroy(propagateStart);
    cudaEventDestroy(propagateStop);
    cudaEventDestroy(updateVerticesStart);
    cudaEventDestroy(updateVerticesStop);
    cudaEventDestroy(updateFrontierStart);
    cudaEventDestroy(updateFrontierStop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void PruneKPAX::planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr)
{
    double t_kgmtStart = std::clock();

    // --- PruneKPAX INITIALIZATION ---
    thrust::fill(d_frontier_.begin(), d_frontier_.end(), false);
    thrust::fill(d_frontierNext_.begin(), d_frontierNext_.end(), false);
    thrust::fill(d_activeFrontierIdxs_.begin(), d_activeFrontierIdxs_.end(), 0);
    thrust::fill(d_unexploredSamples_.begin(), d_unexploredSamples_.end(), 0.0f);
    thrust::fill(d_unexploredSamplesParentIdxs_.begin(), d_unexploredSamplesParentIdxs_.end(), -1);
    thrust::fill(d_frontierScanIdx_.begin(), d_frontierScanIdx_.end(), 0);
    thrust::fill(d_frontierRepeatScanIdx_.begin(), d_frontierRepeatScanIdx_.end(), 0);
    thrust::fill(d_goalSample_.begin(), d_goalSample_.end(), 0.0f);
    thrust::fill(graph_.d_activeSubVertices_.begin(), graph_.d_activeSubVertices_.end(), false);
    thrust::fill(graph_.d_vertexScoreArray_.begin(), graph_.d_vertexScoreArray_.end(), 0.0f);
    thrust::fill(graph_.d_counterArray_.begin(), graph_.d_counterArray_.end(), 0);
    thrust::fill(graph_.d_validCounterArray_.begin(), graph_.d_validCounterArray_.end(), 0);
    thrust::fill(d_treeSamples_.begin(), d_treeSamples_.end(), 0.0f);
    thrust::fill(d_treeSamplesParentIdxs_.begin(), d_treeSamplesParentIdxs_.end(), -1);
    thrust::fill(d_treeSampleCosts_.begin(), d_treeSampleCosts_.end(), 0.0f);
    thrust::fill(d_frontier_.begin(), d_frontier_.begin() + 1, true);
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.begin() + 1, 5);

    h_treeSize_     = 1;
    h_itr_          = 0;
    h_costToGoal_   = 0;
    h_pathToGoal_   = 0;
    h_frontierSize_ = 0;

    cudaMemcpy(d_treeSamples_ptr_, h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costToGoal_ptr_, &h_costToGoal_, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pathToGoal_ptr_, &h_pathToGoal_, sizeof(int), cudaMemcpyHostToDevice);

    initializeRandomSeeds(static_cast<unsigned int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
    // --- END INITIALIZATION ---

    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            printf("Iteration: %d, Tree Size: %d, Frontier Size: %d\n", h_itr_, h_treeSize_, h_frontierSize_);
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            graph_.updateVertices();
            updateFrontier();
            writeDeviceVectorsToCSV(benchItr);
            if(h_pathToGoal_ != 0)
                {
                    printf("Goal Reached\n");
                    cudaMemcpy(h_controlPathToGoal_, d_controlPathToGoal_ptr_, h_itr_ * SAMPLE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
                    break;
                }
        }

    double executionTime = (std::clock() - t_kgmtStart) / (double)CLOCKS_PER_SEC;
    std::cout << "PruneKPAX execution time: " << executionTime << " seconds. Iterations: " << h_itr_ << ". Tree Size: " << h_treeSize_
              << std::endl;
}

void PruneKPAX::propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount)
{
    // --- Build spatial hash grid for fast collision detection ---
    updateSpatialHashGrid(d_spatialHashGrid_, d_obstacles_ptr, h_obstaclesCount);
    cudaMemcpy(&h_spatialHashGrid_, d_spatialHashGrid_, sizeof(SpatialHashGrid), cudaMemcpyDeviceToHost);

    // --- Find indices and size of frontier. ---
    thrust::exclusive_scan(d_frontier_.begin(), d_frontier_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontier_[MAX_TREE_SIZE - 1]) ? ++h_frontierSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontier_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Build frontier repeat vector. ---
    thrust::exclusive_scan(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), d_frontierRepeatScanIdx_.begin(), 0,
                           thrust::plus<uint>());
    repeatInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_activeFrontierIdxs_ptr_, d_activeFrontierRepeatCount_ptr_,
                                             d_frontierRepeatScanIdx_ptr_, d_activeFrontierRepeatIdxs_ptr_);
    h_frontierRepeatSize_ = d_frontierRepeatScanIdx_[MAX_TREE_SIZE - 1];
    (d_activeFrontierRepeatCount_[MAX_TREE_SIZE - 1]) ? h_frontierRepeatSize_ += d_activeFrontierRepeatCount_[MAX_TREE_SIZE - 1] : 0;

    if(h_frontierRepeatSize_ * h_activeBlockSize_ > (MAX_TREE_SIZE - h_treeSize_))
        {
            h_propIterations_ = std::min(int(float(MAX_TREE_SIZE - h_treeSize_) / float(h_frontierRepeatSize_)), int(h_activeBlockSize_));

            if(h_propIterations_ == 0)
                {
                    // TODO: update this. possibly update tree size in here
                    h_propIterations_   = 1;
                    h_frontierNextSize_ = MAX_TREE_SIZE - h_treeSize_;
                    thrust::fill(d_frontierNext_.begin(), d_frontierNext_.end(), false);
                }

            // --- Propagate Frontier. iterations new samples per frontier sample---
            prune_propagateFrontier_kernel2<<<iDivUp(h_propIterations_ * h_frontierRepeatSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              graph_.d_vertexScoreArray_ptr_, d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              h_propIterations_, graph_.d_minValueInRegion_ptr_, h_spatialHashGrid_);
        }
    else
        {
            // --- Propagate Frontier. Block Size new samples per frontier sample. ---
            prune_propagateFrontier_kernel1<<<iDivUp(h_frontierRepeatSize_ * h_activeBlockSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              graph_.d_vertexScoreArray_ptr_, d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              graph_.d_minValueInRegion_ptr_, h_spatialHashGrid_);
        }
}

/***************************/
/* PROPAGATE FRONTIER KERNEL  1*/
/***************************/
// --- Propagates current frontier. Builds new frontier. ---
// --- One Block Per Frontier Sample ---
__global__ void prune_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples, float* unexploredSamples,
                                          uint frontierSize, curandState* randomSeeds, int* unexploredSamplesParentIdxs, float* obstacles,
                                          int obstaclesCount, int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                          int* vertexCounter, int* validVertexCounter, float* minValueInRegion, SpatialHashGrid spatialHashGrid)
{
    if(blockIdx.x >= frontierSize) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= MAX_TREE_SIZE) return;

    // --- Load Frontier Sample Idx into shared memory.  ---
    __shared__ int s_x0Idx;
    if(threadIdx.x == 0)
        {
            s_x0Idx           = activeFrontierIdxs[blockIdx.x];
            frontier[s_x0Idx] = false;
        }
    __syncthreads();

    // --- Load Frontier Sample into shared memory. ---
    __shared__ float s_x0[SAMPLE_DIM];
    if(threadIdx.x < SAMPLE_DIM) s_x0[threadIdx.x] = treeSamples[s_x0Idx * SAMPLE_DIM + threadIdx.x];
    __syncthreads();

    // --- Propagate Sample and add it to unexplored sample set. ---
    float* x1                        = &unexploredSamples[tid * SAMPLE_DIM];
    unexploredSamplesParentIdxs[tid] = s_x0Idx;
    curandState randSeed             = randomSeeds[tid];
    bool valid                       = propagateAndCheckSpatialHash(s_x0, x1, &randSeed, spatialHashGrid, obstacles, obstaclesCount);
    int x1Vertex                     = getRegion(x1);
    int x1SubVertex                  = getSubRegion(x1, x1Vertex, minValueInRegion);

    // --- Update Graph sample count and populate next Frontier ---
    atomicAdd(&vertexCounter[x1Vertex], 1);
    if(valid)
        {
            atomicAdd(&validVertexCounter[x1Vertex], 1);
            if(curand_uniform(&randSeed) < vertexScores[x1Vertex] || !activeSubVertices[x1SubVertex]) frontierNext[tid] = true;
            if(activeSubVertices[x1SubVertex] == 0) atomicExch(&activeSubVertices[x1SubVertex], 1);
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* FRONTIER PROPAGATION KERNEL 2 */
/***************************/
// --- Iterations new samples per frontier sample---
__global__ void prune_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples, float* unexploredSamples,
                                          uint frontierSize, curandState* randomSeeds, int* unexploredSamplesParentIdxs, float* obstacles,
                                          int obstaclesCount, int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                          int* vertexCounter, int* validVertexCounter, int iterations, float* minValueInRegion, SpatialHashGrid spatialHashGrid)
{
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    frontier[tid] = false;
    if(tid >= frontierSize * iterations) return;
    if(tid >= MAX_TREE_SIZE) return;

    int activeFrontierIdx = tid / iterations;
    int x0Idx             = activeFrontierIdxs[activeFrontierIdx];

    // --- Load Frontier Sample into memory. ---
    float* x0 = &treeSamples[x0Idx * SAMPLE_DIM];

    // --- Propagate Sample and add it to unexplored sample set. ---
    float* x1                        = &unexploredSamples[tid * SAMPLE_DIM];
    unexploredSamplesParentIdxs[tid] = x0Idx;
    curandState randSeed             = randomSeeds[tid];
    bool valid                       = propagateAndCheckSpatialHash(x0, x1, &randSeed, spatialHashGrid, obstacles, obstaclesCount);
    int x1Vertex                     = getRegion(x1);
    int x1SubVertex                  = getSubRegion(x1, x1Vertex, minValueInRegion);

    // --- Update Graph sample count and populate next Frontier ---
    atomicAdd(&vertexCounter[x1Vertex], 1);
    if(valid)
        {
            atomicAdd(&validVertexCounter[x1Vertex], 1);
            if(curand_uniform(&randSeed) < vertexScores[x1Vertex] || activeSubVertices[x1SubVertex] == 0) frontierNext[tid] = true;
            if(activeSubVertices[x1SubVertex] == 0) atomicExch(&activeSubVertices[x1SubVertex], 1);
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// --- Adds previous frontier to the tree and builds new frontier. ---
__global__ void
prune_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize, float* xGoal, int treeSize,
                      float* unexploredSamples, float* treeSamples, int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs,
                      float* treeSampleCosts, int* pathToGoal, uint* activeFrontierRepeatCount, int* validVertexCounter,
                      curandState* randomSeeds, float* vertexScores, float* controlPathToGoal, float fAccept,
                      float goalBias, float explorationBias, float maxRegression)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_xGoal[SAMPLE_DIM];
    if(threadIdx.x < SAMPLE_DIM) s_xGoal[threadIdx.x] = xGoal[threadIdx.x];
    __syncthreads();

    // --- Add next frontier to frontier ---
    if(tid < frontierNextSize)
        {
            int x1TreeIdx       = treeSize + tid;
            int x1UnexploredIdx = activeFrontierNextIdxs[tid];
            frontierNext[activeFrontierNextIdxs[tid]] = false;
            float* x1            = &unexploredSamples[x1UnexploredIdx * SAMPLE_DIM];
            int x0Idx            = unexploredSamplesParentIdxs[x1UnexploredIdx];
            
            // --- Compute distances for goal-biased pruning ---
            float distToGoal = distance(x1, s_xGoal);
            float parentDistToGoal = distance(&treeSamples[x0Idx * SAMPLE_DIM], s_xGoal);
            float progressToGoal = parentDistToGoal - distToGoal;  // Positive = moving toward goal
            
            // --- Goal-biased acceptance probability ---
            // If we made progress toward goal, higher acceptance probability
            // If we moved away from goal, lower acceptance probability
    
            float normalizedProgress = (progressToGoal + maxRegression) / (2.0f * maxRegression);  // Maps to [0, 1]
            normalizedProgress = fminf(fmaxf(normalizedProgress, 0.0f), 1.0f);  // Clamp to [0, 1]

            float acceptanceProbability = explorationBias + goalBias * normalizedProgress;
            acceptanceProbability = fminf(acceptanceProbability, 1.0f);  // Cap at 1.0
            
            // --- Probabilistic pruning ---
            curandState seed = randomSeeds[x1TreeIdx];
            bool acceptNode = curand_uniform(&seed) < acceptanceProbability;
            randomSeeds[x1TreeIdx] = seed;
            
            if(!acceptNode)
            {
                // Node rejected, don't add to tree or frontier
                return;
            }
            
            // --- Node accepted, add to tree ---
            treeSamplesParentIdxs[x1TreeIdx] = x0Idx;
            for(int i = 0; i < SAMPLE_DIM; i++)
                treeSamples[x1TreeIdx * SAMPLE_DIM + i] = x1[i];
            treeSampleCosts[x1TreeIdx] = distToGoal;

            // --- Update Frontier ---
            frontier[x1TreeIdx] = true;

            int xVertex = getRegion(x1);
            if(validVertexCounter[xVertex] < 10)
                activeFrontierRepeatCount[x1TreeIdx] = 15;
            else
                activeFrontierRepeatCount[x1TreeIdx] = 1;

            // --- Goal Criteria Check ---
            if(distToGoal < GOAL_THRESH)
                {
                    pathToGoal[0] = x1TreeIdx;
                    int i         = 0;
                    for(int j = 0; j < SAMPLE_DIM; j++)
                        controlPathToGoal[i * SAMPLE_DIM + j] = x1[j];
                    i++;
                    while(x0Idx != -1)
                        {
                            for(int j = 0; j < SAMPLE_DIM; j++)
                                controlPathToGoal[i * SAMPLE_DIM + j] = treeSamples[x0Idx * SAMPLE_DIM + j];
                            x0Idx = treeSamplesParentIdxs[x0Idx];
                            i++;
                        }
                }
        }

    // --- Add inactive tree samples back to frontier ---
    else if(tid < frontierNextSize + treeSize)
        {
            int treeIdx       = tid - frontierNextSize;
            float* treeSample = &treeSamples[treeIdx * SAMPLE_DIM];
            int xVertex       = getRegion(treeSample);

            curandState seed = randomSeeds[treeIdx];
            if(frontier[treeIdx] == 0 && curand_uniform(&seed) <= vertexScores[xVertex] + fAccept)
                {
                    frontier[treeIdx]                  = true;
                    activeFrontierRepeatCount[treeIdx] = 1;
                }
        }
}

void PruneKPAX::updateFrontier()

    {
    // --- Find indices and size of the next frontier ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    float treeAddSize = 1 - (float(h_treeSize_ + h_frontierNextSize_) / (MAX_TREE_SIZE));
    h_fAccept_        = (h_itr_ * EPSILON) * pow(treeAddSize, 5);

    // --- Update Frontier ---
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    prune_updateFrontier_kernel<<<iDivUp(h_frontierNextSize_ + h_treeSize_, h_blockSize_), h_blockSize_>>>(
      d_frontier_ptr_, d_frontierNext_ptr_, d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_goalSample_ptr_, h_treeSize_,
      d_unexploredSamples_ptr_, d_treeSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, d_pathToGoal_ptr_, d_activeFrontierRepeatCount_ptr_, graph_.d_validCounterArray_ptr_, d_randomSeeds_ptr_,
      graph_.d_vertexScoreArray_ptr_, d_controlPathToGoal_ptr_, h_fAccept_,
      h_goalBias_, h_explorationBias_, h_maxRegression_); 

    // --- Check for goal criteria ---
    cudaMemcpy(&h_pathToGoal_, d_pathToGoal_ptr_, sizeof(int), cudaMemcpyDeviceToHost);

    // --- Update Tree Size ---
    h_treeSize_ += h_frontierNextSize_;
}

void PruneKPAX::writeDeviceVectorsToCSV(int itr)
{
    std::ostringstream filename;
    bool append = h_itr_ != 0;

    // Create necessary directories
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/Samples/Samples" + std::to_string(itr));
    std::filesystem::create_directories("Data/Parents/Parents" + std::to_string(itr));
    std::filesystem::create_directories("Data/TotalCountPerVertex/TotalCountPerVertex" + std::to_string(itr));
    std::filesystem::create_directories("Data/ValidCountPerVertex/ValidCountPerVertex" + std::to_string(itr));
    std::filesystem::create_directories("Data/Frontier/Frontier" + std::to_string(itr));
    std::filesystem::create_directories("Data/FrontierRepeatCount/FrontierRepeatCount" + std::to_string(itr));
    std::filesystem::create_directories("Data/VertexScores/VertexScores" + std::to_string(itr));
    std::filesystem::create_directories("Data/FrontierSize/FrontierSize" + std::to_string(itr));
    std::filesystem::create_directories("Data/TreeSize/TreeSize" + std::to_string(itr));
    std::filesystem::create_directories("Data/ExpandedNodes/ExpandedNodes" + std::to_string(itr));
    std::filesystem::create_directories("Data/ControlPathToGoal/ControlPathToGoal" + std::to_string(itr));

    // Write Samples
    filename.str("");
    filename << "Data/Samples/Samples" << itr << "/samples" << h_itr_ << ".csv";
    copyAndWriteVectorToCSV(d_treeSamples_, filename.str(), MAX_TREE_SIZE, SAMPLE_DIM, append);

    // Write Parents
    filename.str("");
    filename << "Data/Parents/Parents" << itr << "/parents" << h_itr_ << ".csv";
    copyAndWriteVectorToCSV(d_treeSamplesParentIdxs_, filename.str(), MAX_TREE_SIZE, 1, append);

    // Write Total Count Per Vertex
    filename.str("");
    filename << "Data/TotalCountPerVertex/TotalCountPerVertex" << itr << "/totalCountPerVertex.csv";
    copyAndWriteVectorToCSV(graph_.d_counterArray_, filename.str(), 1, NUM_R1_REGIONS, append);

    // Write Valid Count Per Vertex
    filename.str("");
    filename << "Data/ValidCountPerVertex/ValidCountPerVertex" << itr << "/validCountPerVertex.csv";
    copyAndWriteVectorToCSV(graph_.d_validCounterArray_, filename.str(), 1, NUM_R1_REGIONS, append);

    // Write Frontier
    filename.str("");
    filename << "Data/Frontier/Frontier" << itr << "/frontier.csv";
    copyAndWriteVectorToCSV(d_frontier_, filename.str(), 1, MAX_TREE_SIZE, append);

    // Write Frontier Repeat
    filename.str("");
    filename << "Data/FrontierRepeatCount/FrontierRepeatCount" << itr << "/frontierRepeatCount.csv";
    copyAndWriteVectorToCSV(d_activeFrontierRepeatCount_, filename.str(), 1, MAX_TREE_SIZE, append);

    // Write Vertex Scores
    filename.str("");
    filename << "Data/VertexScores/VertexScores" << itr << "/vertexScores.csv";
    copyAndWriteVectorToCSV(graph_.d_vertexScoreArray_, filename.str(), 1, NUM_R1_REGIONS, append);

    // Write Frontier Size
    filename.str("");
    filename << "Data/FrontierSize/FrontierSize" << itr << "/frontierSize.csv";
    writeValueToCSV(h_frontierSize_, filename.str());

    // Write Tree Size
    filename.str("");
    filename << "Data/TreeSize/TreeSize" << itr << "/treeSize.csv";
    writeValueToCSV(h_treeSize_, filename.str());

    // Expanded Nodes
    filename.str("");
    filename << "Data/ExpandedNodes/ExpandedNodes" << itr << "/expandedNodes.csv";
    if(h_frontierRepeatSize_ * h_activeBlockSize_ > (MAX_TREE_SIZE - h_treeSize_))
        {
            writeValueToCSV(h_propIterations_ * h_frontierRepeatSize_, filename.str());
        }
    else
        {
            writeValueToCSV(h_frontierRepeatSize_ * h_activeBlockSize_, filename.str());
        }

    // Write Control Path to Goal
    filename.str("");
    filename << "Data/ControlPathToGoal/ControlPathToGoal" << itr << "/controlPathToGoal.csv";
    if(h_pathToGoal_ != 0)
        {
            copyAndWriteVectorToCSV(d_controlPathToGoal_, filename.str(), h_itr_, SAMPLE_DIM, append);
        }
}

void PruneKPAX::writeExecutionTimeToCSV(double time)
{
    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/ExecutionTime");
    filename.str("");
    filename << "Data/ExecutionTime/executionTime.csv";
    writeValueToCSV(time, filename.str());
}
