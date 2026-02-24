#include "planners/OKPAX.cuh"
#include "config/config.h"

OKPAX::OKPAX()
{
    graph_ = OKPAXRegions(W_SIZE);

    d_frontier_                    = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_frontierNext_                = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_activeFrontierIdxs_          = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_goalSetIdxs_                 = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_unexploredSamples_           = thrust::device_vector<float>(MAX_TREE_SIZE * SAMPLE_DIM);
    d_unexploredSamplesParentIdxs_ = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_frontierScanIdx_             = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_goalSetScanIdx_              = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_goalSample_                  = thrust::device_vector<float>(SAMPLE_DIM);
    d_goalSet_                     = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_treeXR1s_                    = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_frontierNextXR1s_            = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_pruned_                      = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_treeInactiveIterations_      = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_unexploredSampleCosts_       = thrust::device_vector<float>(MAX_TREE_SIZE);
    d_pathCosts_                   = thrust::device_vector<float>(MAX_TREE_SIZE * 3);
    d_iterations_                  = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_controlPathsToGoal_          = thrust::device_vector<float>(MAX_ITER * SAMPLE_DIM);

    d_frontier_ptr_                    = thrust::raw_pointer_cast(d_frontier_.data());
    d_frontierNext_ptr_                = thrust::raw_pointer_cast(d_frontierNext_.data());
    d_activeFrontierIdxs_ptr_          = thrust::raw_pointer_cast(d_activeFrontierIdxs_.data());
    d_goalSetIdxs_ptr_                 = thrust::raw_pointer_cast(d_goalSetIdxs_.data());
    d_unexploredSamples_ptr_           = thrust::raw_pointer_cast(d_unexploredSamples_.data());
    d_unexploredSamplesParentIdxs_ptr_ = thrust::raw_pointer_cast(d_unexploredSamplesParentIdxs_.data());
    d_frontierScanIdx_ptr_             = thrust::raw_pointer_cast(d_frontierScanIdx_.data());
    d_goalSetScanIdx_ptr_              = thrust::raw_pointer_cast(d_goalSetScanIdx_.data());
    d_goalSample_ptr_                  = thrust::raw_pointer_cast(d_goalSample_.data());
    d_goalSet_ptr_                     = thrust::raw_pointer_cast(d_goalSet_.data());
    d_treeXR1s_ptr_                    = thrust::raw_pointer_cast(d_treeXR1s_.data());
    d_frontierNextXR1s_ptr_            = thrust::raw_pointer_cast(d_frontierNextXR1s_.data());
    d_pruned_ptr_                      = thrust::raw_pointer_cast(d_pruned_.data());
    d_treeInactiveIterations_ptr_      = thrust::raw_pointer_cast(d_treeInactiveIterations_.data());
    d_unexploredSampleCosts_ptr_       = thrust::raw_pointer_cast(d_unexploredSampleCosts_.data());
    d_pathCosts_ptr_                   = thrust::raw_pointer_cast(d_pathCosts_.data());
    d_iterations_ptr_                  = thrust::raw_pointer_cast(d_iterations_.data());
    d_controlPathsToGoal_ptr_          = thrust::raw_pointer_cast(d_controlPathsToGoal_.data());

    cudaMalloc(&d_minCost_ptr_, sizeof(float));

    h_activeBlockSize_ = 32;
    h_controlPathsToGoal_ = new float[MAX_ITER * SAMPLE_DIM];

    if(VERBOSE)
        {
            printf("/* Planner Type: OKPAX */\n");
            printf("/* Number of R1 Vertices: %d */\n", NUM_R1_REGIONS);
            printf("/* Number of R2 Vertices: %d */\n", NUM_R2_REGIONS);
            printf("/***************************/\n");
        }
}

void OKPAX::resetPlanner(float* h_initial, float* h_goal)
{
    // --- Resetting Device Vectors: ---
    thrust::fill(d_frontier_.begin(), d_frontier_.end(), false);
    thrust::fill(d_frontier_.begin(), d_frontier_.begin() + 1, true);
    thrust::fill(d_frontierNext_.begin(), d_frontierNext_.end(), false);
    thrust::fill(d_activeFrontierIdxs_.begin(), d_activeFrontierIdxs_.end(), 0);
    thrust::fill(d_goalSetIdxs_.begin(), d_goalSetIdxs_.end(), 0);
    thrust::fill(d_unexploredSamples_.begin(), d_unexploredSamples_.end(), 0.0f);
    thrust::fill(d_unexploredSamplesParentIdxs_.begin(), d_unexploredSamplesParentIdxs_.end(), -1);
    thrust::fill(d_frontierScanIdx_.begin(), d_frontierScanIdx_.end(), 0);
    thrust::fill(d_goalSetScanIdx_.begin(), d_goalSetScanIdx_.end(), 0);
    thrust::fill(d_goalSample_.begin(), d_goalSample_.end(), 0.0f);
    thrust::fill(graph_.d_minCostsR1_.begin(), graph_.d_minCostsR1_.end(), MAX_FLOAT);
    thrust::fill(d_treeSamples_.begin(), d_treeSamples_.end(), 0.0f);
    thrust::fill(d_treeSamplesParentIdxs_.begin(), d_treeSamplesParentIdxs_.end(), -1);
    thrust::fill(d_treeSampleCosts_.begin(), d_treeSampleCosts_.end(), 0.0f);
    thrust::fill(d_unexploredSampleCosts_.begin(), d_unexploredSampleCosts_.end(), 0.0f);
    thrust::fill(d_goalSet_.begin(), d_goalSet_.end(), false);
    thrust::fill(d_pathCosts_.begin(), d_pathCosts_.end(), 0.0f);
    thrust::fill(d_iterations_.begin(), d_iterations_.end(), 0);
    thrust::fill(d_pruned_.begin(), d_pruned_.end(), false);
    thrust::fill(d_treeInactiveIterations_.begin(), d_treeInactiveIterations_.end(), 0);
    thrust::fill(d_controlPathsToGoal_.begin(), d_controlPathsToGoal_.end(), 0.0f);

    h_treeSize_               = 1;
    h_itr_                    = 0;
    h_costToGoal_             = 0;
    h_pathToGoal_             = 0;
    h_frontierSize_           = 0;
    h_solSetSize_             = 0;
    h_minCost_                = MAX_FLOAT;
    h_propIterations_         = 1;
    h_addedNodes_             = 0;
    h_previousExpansionCount_ = 0;

    cudaMemcpy(d_treeSamples_ptr_, h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costToGoal_ptr_, &h_costToGoal_, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pathToGoal_ptr_, &h_pathToGoal_, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minCost_ptr_, &h_minCost_, sizeof(float), cudaMemcpyHostToDevice);

    initializeRandomSeeds(static_cast<unsigned int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
}

void OKPAX::plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- INITIALIZE Planner ---
    resetPlanner(h_initial, h_goal);

    // --- PLANNING ---
    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            updateFrontier();
            if(h_pathToGoal_ != 0)
                {
                    cudaMemcpy(h_controlPathsToGoal_, d_controlPathsToGoal_ptr_, h_itr_ * SAMPLE_DIM * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    break;
                }
        }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "OKPAX execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_ << ". Tree Size: " << h_treeSize_
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

float OKPAX::planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- INITIALIZE OKPAX ---
    resetPlanner(h_initial, h_goal);

    // --- PLANNING ---
    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            if(h_propIterations_ == 0) break;
            updateFrontier();
            if(h_pathToGoal_ != 0)
                {
                    cudaMemcpy(h_controlPathsToGoal_, d_controlPathsToGoal_ptr_, h_itr_ * SAMPLE_DIM * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    break;
                }
        }
    getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "OKPAX execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_ << ". Tree Size: " << h_treeSize_
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return h_minCost_;
}

void OKPAX::planBenchmark(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr)
{
    std::vector<float> iterationTimes;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- INITIALIZE OKPAX ---
    resetPlanner(h_initial, h_goal);

    // --- PLANNING ---
    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            if(h_propIterations_ == 0) break;
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            updateFrontier();

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            iterationTimes.push_back(milliseconds);
        }
    getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "OKPAX execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_ << ". Tree Size: " << h_treeSize_
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cuda timing has stopped by here but remove if you do not want the data.
    writeSolutionsToCSV(benchItr);
    writeSolutionCostsToCSV(benchItr);
    writeIterationTimeToCSV(iterationTimes, benchItr);
    // Until here.
}

void OKPAX::planPathsCollect(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr)
{
    std::vector<float> iterationTimes;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- INITIALIZE OKPAX ---
    resetPlanner(h_initial, h_goal);

    // --- PLANNING ---
    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            printf("Iteration: %d, Tree Size: %d, Frontier Size: %d\n", h_itr_, h_treeSize_, h_frontierSize_);
            if(h_propIterations_ == 0) break;
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            updateFrontier();
            getControlPathToGoalPathsCollect();

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            iterationTimes.push_back(milliseconds);
        }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "OKPAX execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_ << ". Tree Size: " << h_treeSize_
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cuda timing has stopped by here but remove if you do not want the data.
    writeSolutionsToCSV(benchItr);
    writeSolutionCostsToCSV(benchItr);
    writeIterationTimeToCSV(iterationTimes, benchItr);
    // Until here.
}

void OKPAX::planDataCollect(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr)
{
    // --- OKPAX INITIALIZATION ---
    resetPlanner(h_initial, h_goal);

    // --- PLANNING ---
    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            printf("Iteration: %d, Tree Size: %d, Frontier Size: %d\n", h_itr_, h_treeSize_, h_frontierSize_);
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            updateFrontier();
            writeDeviceVectorsToCSV(h_itr_);
            if(h_pathToGoal_ != 0)
                {
                    printf("Goal Reached\n");
                    cudaMemcpy(h_controlPathsToGoal_, d_controlPathsToGoal_ptr_, h_itr_ * SAMPLE_DIM * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    break;
                }
        }
    getControlPathToGoal();
    writeSolutionsToCSV();
    writeSolutionCostsToCSV();
    printf("h_solSetSize_: %d\n", h_solSetSize_);
    writeDeviceVectorsToCSV(h_itr_);
}

void OKPAX::propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount)
{
    // --- Find indices and size of frontier. ---
    thrust::exclusive_scan(d_frontier_.begin(), d_frontier_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontier_[MAX_TREE_SIZE - 1]) ? ++h_frontierSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontier_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    if(h_frontierSize_ * h_activeBlockSize_ > (MAX_TREE_SIZE))
        {
            h_propIterations_ = std::min(int(float(MAX_TREE_SIZE) / float(h_frontierSize_)), int(h_activeBlockSize_));
            if(h_propIterations_ == 0)
                {
                    printf("Tree Full\n");
                    return;
                }
            // --- Propagate Frontier. iterations times each.---
            OKPAX_propagateFrontier_kernel2<<<iDivUp(h_propIterations_ * h_frontierSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierSize_, d_randomSeeds_ptr_,
              d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, d_frontierNext_ptr_, h_propIterations_,
              d_treeSampleCosts_ptr_, graph_.d_minCostsR1_ptr_, d_frontierNextXR1s_ptr_, d_unexploredSampleCosts_ptr_);
        }
    else
        {
            int bf = std::floor(float(MAX_TREE_SIZE) / (float(h_frontierSize_) * float(h_activeBlockSize_)));
            // --- Propagate Frontier. bf*BlockSize times each. ---
            OKPAX_propagateFrontier_kernel1V2<<<iDivUp(bf * h_frontierSize_ * h_activeBlockSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierSize_, d_randomSeeds_ptr_,
              d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, d_frontierNext_ptr_, d_treeSampleCosts_ptr_,
              graph_.d_minCostsR1_ptr_, d_frontierNextXR1s_ptr_, d_unexploredSampleCosts_ptr_, bf);
        }
}

/***************************/
/* PROPAGATE FRONTIER KERNEL  1*/
/***************************/
// --- Propagates current frontier. Builds new frontier. ---
// --- One Block Per Frontier Sample ---
__global__ void OKPAX_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples, float* unexploredSamples,
                                                uint frontierSize, curandState* randomSeeds, int* unexploredSamplesParentIdxs,
                                                float* obstacles, int obstaclesCount, bool* frontierNext, float* treeSampleCosts,
                                                float* minCostsR1, int* frontierNextXR1s, float* unexploredSampleCosts)
{
    // if(blockIdx.x >= frontierSize) return;
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    frontier[tid] = false;
    if(tid >= MAX_TREE_SIZE) return;

    // --- Load Frontier Sample Idx into shared memory.  ---
    __shared__ int s_x0Idx;
    __shared__ float s_x0Cost;
    if(threadIdx.x == 0)
        {
            s_x0Idx           = activeFrontierIdxs[blockIdx.x];
            s_x0Cost          = treeSampleCosts[s_x0Idx];
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
    bool valid                       = propagateAndCheck(s_x0, x1, &randSeed, obstacles, obstaclesCount);

    // --- Update Graph sample count and populate next Frontier ---
    if(valid)
        {
            int x1R1   = OKPAX_getRegion(x1);
            float cost = s_x0Cost + distance(s_x0, x1);  // TODO: Currently just distance.
            if(minCostsR1[x1R1] > cost) atomicMinFloat(&minCostsR1[x1R1], cost);
            if(cost <= minCostsR1[x1R1])
                {
                    unexploredSampleCosts[tid] = cost;
                    frontierNextXR1s[tid]      = x1R1;
                    frontierNext[tid]          = true;
                }
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* PROPAGATE FRONTIER KERNEL  1*/
/***************************/
// --- Propagates current frontier. Builds new frontier. ---
// --- One Block Per Frontier Sample ---
__global__ void OKPAX_propagateFrontier_kernel1V2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples, float* unexploredSamples,
                                                  uint frontierSize, curandState* randomSeeds, int* unexploredSamplesParentIdxs,
                                                  float* obstacles, int obstaclesCount, bool* frontierNext, float* treeSampleCosts,
                                                  float* minCostsR1, int* frontierNextXR1s, float* unexploredSampleCosts, int bf)
{
    // if(blockIdx.x >= frontierSize) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= MAX_TREE_SIZE) return;

    // --- Load Frontier Sample Idx into shared memory.  ---
    __shared__ int s_x0Idx;
    __shared__ float s_x0Cost;
    if(threadIdx.x == 0)
        {
            s_x0Idx           = activeFrontierIdxs[blockIdx.x / bf];
            s_x0Cost          = treeSampleCosts[s_x0Idx];
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
    bool valid                       = propagateAndCheck(s_x0, x1, &randSeed, obstacles, obstaclesCount);

    // --- Update Graph sample count and populate next Frontier ---
    if(valid)
        {
            int x1R1   = OKPAX_getRegion(x1);
            float cost = s_x0Cost + distance(s_x0, x1);  // TODO: Currently just distance.
            if(minCostsR1[x1R1] > cost) atomicMinFloat(&minCostsR1[x1R1], cost);
            if(cost <= minCostsR1[x1R1])
                {
                    unexploredSampleCosts[tid] = cost;
                    frontierNextXR1s[tid]      = x1R1;
                    frontierNext[tid]          = true;
                }
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* FRONTIER PROPAGATION KERNEL 2 */
/***************************/
// --- Iterations new samples per frontier sample---
__global__ void
OKPAX_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples, float* unexploredSamples, uint frontierSize,
                                curandState* randomSeeds, int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                bool* frontierNext, int iterations, float* treeSampleCosts, float* minCostsR1, int* frontierNextXR1s,
                                float* unexploredSampleCosts)
{
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    frontier[tid] = false;
    if(tid >= frontierSize * iterations) return;
    if(tid >= MAX_TREE_SIZE) return;

    int activeFrontierIdx = tid / iterations;
    int x0Idx             = activeFrontierIdxs[activeFrontierIdx];
    float x0Cost          = treeSampleCosts[x0Idx];

    // --- Load Frontier Sample into memory. ---
    float* x0 = &treeSamples[x0Idx * SAMPLE_DIM];

    // --- Propagate Sample and add it to unexplored sample set. ---
    float* x1                        = &unexploredSamples[tid * SAMPLE_DIM];
    unexploredSamplesParentIdxs[tid] = x0Idx;
    curandState randSeed             = randomSeeds[tid];
    bool valid                       = propagateAndCheck(x0, x1, &randSeed, obstacles, obstaclesCount);

    // --- Update Graph sample count and populate next Frontier ---
    if(valid)
        {
            int x1R1   = OKPAX_getRegion(x1);
            float cost = x0Cost + distance(x0, x1);  // TODO: Currently just distance.
            if(minCostsR1[x1R1] > cost) atomicMinFloat(&minCostsR1[x1R1], cost);
            if(cost <= minCostsR1[x1R1])
                {
                    unexploredSampleCosts[tid] = cost;
                    frontierNextXR1s[tid]      = x1R1;
                    frontierNext[tid]          = true;
                }
        }

    randomSeeds[tid] = randSeed;
}

void OKPAX::updateFrontier()
{
    // --- Pruning Tree ---
    OKPAX_pruningTree_kernel<<<iDivUp(h_treeSize_, h_blockSize_), h_blockSize_>>>(
      h_treeSize_, d_treeSamplesParentIdxs_ptr_, d_treeSampleCosts_ptr_, d_goalSet_ptr_, graph_.d_minCostsR1_ptr_, d_treeXR1s_ptr_,
      d_pruned_ptr_, d_treeInactiveIterations_ptr_);

    // --- Find indices and size of the next frontier ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Pruning Frontier ---
    OKPAX_pruningFrontier_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
      d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, graph_.d_minCostsR1_ptr_, d_frontierNextXR1s_ptr_, d_treeXR1s_ptr_, d_frontierNext_ptr_,
      d_unexploredSampleCosts_ptr_, d_pruned_ptr_, d_treeInactiveIterations_ptr_);

    // --- Finding updated new samples with pruned tree ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Update Tree Size ---
    if(h_treeSize_ + h_frontierNextSize_ >= MAX_TREE_SIZE)
        {
            h_propIterations_ = 0;
            return;
        }

    // --- Update Frontier ---
    OKPAX_updateFrontier_kernel<<<iDivUp(h_frontierNextSize_ + h_treeSize_, h_blockSize_), h_blockSize_>>>(
      d_frontier_ptr_, d_frontierNext_ptr_, d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_goalSample_ptr_, h_treeSize_,
      d_unexploredSamples_ptr_, d_treeSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, d_randomSeeds_ptr_, d_controlPathsToGoal_ptr_, d_goalSet_ptr_, d_iterations_ptr_, h_itr_,
      graph_.d_minCostsR1_ptr_, d_treeXR1s_ptr_, d_frontierNextXR1s_ptr_, d_minCost_ptr_, d_unexploredSampleCosts_ptr_, d_pruned_ptr_);

    // --- Check for goal criteria ---
    cudaMemcpy(&h_pathToGoal_, d_pathToGoal_ptr_, sizeof(int), cudaMemcpyDeviceToHost);

    h_treeSize_ += h_frontierNextSize_;
}

/***************************/
/* TREE PRUNING Kernel */
/***************************/
__global__ void OKPAX_pruningTree_kernel(int treeSize, int* treeSamplesParentIdxs, float* treeSampleCosts, bool* goalSet, float* minCostsR1,
                                         int* treeXR1s, bool* pruned, uint* inactiveIterations)
{
    int treeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Checking Existing Samples: ---
    if(treeIdx < treeSize)
        {
            int x0Idx      = treeIdx;
            float nodeCost = treeSampleCosts[treeIdx];
            int nodeR1     = treeXR1s[treeIdx];

            // If a node is lowest cost in region, and the cost has not been improved in the region in K iterations, move it to active.
            if(pruned[treeIdx] && nodeCost <= minCostsR1[nodeR1])
                {
                    inactiveIterations[treeIdx]++;
                    if(inactiveIterations[treeIdx] > 5)
                        {
                            pruned[treeIdx] = false;
                        }
                    return;
                }
            // keep the node in active as long as it is the best in region.
            if(inactiveIterations[treeIdx] > 5 && nodeCost <= minCostsR1[nodeR1])
                {
                    return;
                }
            // pruning criteria.
            while(x0Idx != -1)
                {
                    int x0R1 = treeXR1s[x0Idx];
                    if(treeSampleCosts[x0Idx] > minCostsR1[x0R1])
                        {
                            pruned[treeIdx] = true;
                            return;
                        }
                    x0Idx = treeSamplesParentIdxs[x0Idx];
                }
        }
}

/***************************/
/* Frontier PRUNING Kernel */
/***************************/
__global__ void
OKPAX_pruningFrontier_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize, int* unexploredSamplesParentIdxs,
                             int* treeSamplesParentIdxs, float* treeSampleCosts, float* minCostsR1, int* frontierNextXR1s, int* treeXR1s,
                             bool* frontierNext, float* unexploredSampleCosts, bool* pruned, uint* inactiveIterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Checking New Samples: ---
    if(tid < frontierNextSize)
        {
            int treeIdx = activeFrontierNextIdxs[tid];
            float cost  = unexploredSampleCosts[treeIdx];
            int xR1     = frontierNextXR1s[treeIdx];
            if(cost > minCostsR1[xR1])
                {
                    frontierNext[treeIdx] = false;
                    return;
                }
        }
}

__global__ void OKPAX_pruning_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize, int treeSize, int* unexploredSamplesParentIdxs,
                                     int* treeSamplesParentIdxs, float* treeSampleCosts, bool* goalSet, float* minCostsR1, int* treeXR1s,
                                     int* frontierNextXR1s, bool* frontierNext, float* unexploredSampleCosts, bool* pruned)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Checking New Samples: ---
    if(tid < frontierNextSize)
        {
            int treeIdx = activeFrontierNextIdxs[tid];
            int x0Idx   = unexploredSamplesParentIdxs[treeIdx];

            while(x0Idx != -1)
                {
                    int x0R1 = treeXR1s[x0Idx];
                    if(minCostsR1[x0R1] / treeSampleCosts[x0Idx] < 0.99)
                        {
                            frontierNext[treeIdx] = false;
                            pruned[treeIdx]       = true;
                            return;
                        }
                    x0Idx = treeSamplesParentIdxs[x0Idx];
                }
        }

    // --- Checking Existing Samples: ---
    else if(tid < frontierNextSize + treeSize && !goalSet[tid])
        {
            int treeIdx = tid - frontierNextSize;
            int x0Idx   = treeSamplesParentIdxs[treeIdx];

            while(x0Idx != -1)
                {
                    int x0R1 = treeXR1s[x0Idx];
                    if(minCostsR1[x0R1] / treeSampleCosts[x0Idx] < 0.99)
                        {
                            pruned[treeIdx] = true;
                            return;
                        }
                    x0Idx = treeSamplesParentIdxs[x0Idx];
                }
        }
}

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// --- Adds previous frontier to the tree and builds new frontier. ---
__global__ void
OKPAX_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize, float* xGoal,
                            int treeSize, float* unexploredSamples, float* treeSamples, int* unexploredSamplesParentIdxs,
                            int* treeSamplesParentIdxs, float* treeSampleCosts, curandState* randomSeeds, float* controlPathToGoal,
                            bool* goalSet, int* iterations, int iteration, float* minCostsR1, int* treeXR1s, int* frontierNextXR1s,
                            float* minCost, float* unexploredSampleCosts, bool* pruned)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_xGoal[SAMPLE_DIM];
    if(threadIdx.x < SAMPLE_DIM) s_xGoal[threadIdx.x] = xGoal[threadIdx.x];
    __syncthreads();

    // --- Add next frontier to frontier ---
    if(tid < frontierNextSize)
        {
            // --- Update Tree ---
            int x1TreeIdx                             = treeSize + tid;               // --- Index of new tree sample ---
            int x1UnexploredIdx                       = activeFrontierNextIdxs[tid];  // --- Index of sample in unexplored sample set ---
            frontierNext[activeFrontierNextIdxs[tid]] = false;
            float* x1                        = &unexploredSamples[x1UnexploredIdx * SAMPLE_DIM];  // --- sample from unexplored set ---
            int x0Idx                        = unexploredSamplesParentIdxs[x1UnexploredIdx];      // --- parent of the unexplored sample ---
            treeSamplesParentIdxs[x1TreeIdx] = x0Idx;  // --- Transfer parent of unexplored sample to tree ---
            for(int i = 0; i < SAMPLE_DIM; i++)
                treeSamples[x1TreeIdx * SAMPLE_DIM + i] = x1[i];  // --- Transfer unexplored sample to tree ---
            float cost                 = unexploredSampleCosts[x1UnexploredIdx];
            treeSampleCosts[x1TreeIdx] = cost;

            // --- Update Frontier ---
            int xR1             = frontierNextXR1s[x1UnexploredIdx];
            treeXR1s[x1TreeIdx] = xR1;
            if(cost <= minCostsR1[xR1])
                {
                    frontier[x1TreeIdx] = true;
                }

            // --- Goal Criteria Check ---
            if(distance(x1, s_xGoal) < GOAL_THRESH && cost <= *minCost)
                {
                    atomicMinFloat(minCost, cost);
                    goalSet[x1TreeIdx]    = true;
                    frontier[x1TreeIdx]   = false;
                    iterations[x1TreeIdx] = iteration;  // TODO: Remove this. Only for creating cost/iteration plot.
                }
        }

    // --- Add inactive tree samples back to frontier. ---
    else if(tid < frontierNextSize + treeSize)
        {
            int treeIdx = tid - frontierNextSize;
            if(!goalSet[treeIdx] && !pruned[treeIdx])
                {
                    frontier[treeIdx] = true;
                }
        }
}

void OKPAX::getControlPathToGoal()
{
    thrust::exclusive_scan(d_goalSet_.begin(), d_goalSet_.end(), d_goalSetScanIdx_.begin(), 0, thrust::plus<uint>());
    h_solSetSize_ = d_goalSetScanIdx_[MAX_TREE_SIZE - 1];
    (d_goalSet_[MAX_TREE_SIZE - 1]) ? ++h_solSetSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_goalSet_ptr_, d_goalSetScanIdx_ptr_, d_goalSetIdxs_ptr_);

    OKPAX_getControlPathToGoal_kernel<<<iDivUp(h_solSetSize_, h_blockSize_), h_blockSize_>>>(
      d_controlPathsToGoal_ptr_, d_treeSamples_ptr_, d_treeSamplesParentIdxs_ptr_, d_goalSetIdxs_ptr_, h_solSetSize_, d_pathCosts_ptr_,
      d_treeSampleCosts_ptr_, d_iterations_ptr_, d_minCost_ptr_);

    cudaMemcpy(&h_minCost_, d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Cost to Goal: %f\n", h_minCost_);
}

__global__ void
OKPAX_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples, int* treeSamplesParentIdxs, uint* goalSetIdxs,
                                  int goalSetSize, float* pathCosts, float* treeSampleCosts, int* iterations, float* minCost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= MAX_TREE_SIZE || tid >= goalSetSize) return;

    int goalIdx = goalSetIdxs[tid];

    int x0Idx  = goalIdx;
    float cost = treeSampleCosts[goalIdx];

    // TODO: Remove this: only for creating cost/iteration plot. Keep if you would like information on when paths are found.
    int pathCostsIdx            = 3 * tid;
    pathCosts[pathCostsIdx]     = goalIdx;
    pathCosts[pathCostsIdx + 1] = cost;
    pathCosts[pathCostsIdx + 2] = iterations[goalIdx];
    // Until here.

    if(cost != *minCost) return;
    int i = 0;  // --- Iteration counter ---
    while(x0Idx != -1)
        {
            for(int j = 0; j < SAMPLE_DIM; j++)
                {
                    controlPathsToGoal[SAMPLE_DIM * i + j] = treeSamples[x0Idx * SAMPLE_DIM + j];
                }
            i++;
            x0Idx = treeSamplesParentIdxs[x0Idx];
        }
}

void OKPAX::getControlPathToGoalPathsCollect()
{
    thrust::exclusive_scan(d_goalSet_.begin(), d_goalSet_.end(), d_goalSetScanIdx_.begin(), 0, thrust::plus<uint>());
    h_solSetSize_ = d_goalSetScanIdx_[MAX_TREE_SIZE - 1];
    (d_goalSet_[MAX_TREE_SIZE - 1]) ? ++h_solSetSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_goalSet_ptr_, d_goalSetScanIdx_ptr_, d_goalSetIdxs_ptr_);

    if(h_solSetSize_ == 0) return;
    OKPAX_getControlPathToGoalPathsCollect_kernel<<<iDivUp(h_solSetSize_, h_blockSize_), h_blockSize_>>>(
      d_controlPathsToGoal_ptr_, d_treeSamples_ptr_, d_treeSamplesParentIdxs_ptr_, d_goalSetIdxs_ptr_, h_solSetSize_, d_pathCosts_ptr_,
      d_treeSampleCosts_ptr_, d_iterations_ptr_, d_minCost_ptr_, h_itr_, h_solSetSize_);

    cudaMemcpy(&h_minCost_, d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Cost to Goal: %f\n", h_minCost_);
}

__global__ void OKPAX_getControlPathToGoalPathsCollect_kernel(float* controlPathsToGoal, float* treeSamples, int* treeSamplesParentIdxs,
                                                              uint* goalSetIdxs, int goalSetSize, float* pathCosts, float* treeSampleCosts,
                                                              int* iterations, float* minCost, int itr, int numSols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= MAX_TREE_SIZE || tid >= goalSetSize) return;

    int goalIdx = goalSetIdxs[tid];

    int x0Idx  = goalIdx;
    float cost = treeSampleCosts[goalIdx];
    // TODO: Remove this: only for creating cost/iteration plot.
    int pathCostsIdx            = 3 * tid;
    pathCosts[pathCostsIdx]     = goalIdx;
    pathCosts[pathCostsIdx + 1] = cost;
    pathCosts[pathCostsIdx + 2] = iterations[goalIdx];
    if(cost != *minCost) return;
    int i = 0;  // --- Iteration counter ---
    while(x0Idx != -1)
        {
            for(int j = 0; j < SAMPLE_DIM; j++)
                {
                    controlPathsToGoal[numSols * itr * SAMPLE_DIM + (SAMPLE_DIM * i + j)] = treeSamples[x0Idx * SAMPLE_DIM + j];
                }
            i++;
            x0Idx = treeSamplesParentIdxs[x0Idx];
        }
}

// Placeholder implementations for CSV writing methods
void OKPAX::writeDeviceVectorsToCSV(int itr)
{
    // TODO: Implement if needed
}

void OKPAX::writeSolutionsToCSV(int benchItr)
{
    // TODO: Implement if needed
}

void OKPAX::writeSolutionCostsToCSV(int benchItr)
{
    // TODO: Implement if needed
}

void OKPAX::writeIterationTimeToCSV(std::vector<float>& iterationTimes, int benchItr)
{
    // TODO: Implement if needed
}

void OKPAX::writeExecutionTimeToCSV(double time)
{
    // TODO: Implement if needed
}

/***************************/
/* OKPAX GET REGION DEVICE FUNCTION */
/***************************/
// --- Returns the R1 region index for a given coordinate ---
// --- This is implemented in OKPAXRegions.cu but declared here for kernel usage ---
__device__ int OKPAX_getRegion(float* coord)
{
    // --- Workspace ---
    int wRegion = 0;
    int factor  = 1;
    int index;
    for(int i = W_DIM - 1; i >= 0; --i)
        {
            index = (int)(W_R1_LENGTH * (coord[i] - W_MIN) / (W_MAX - W_MIN));
            if(index >= W_R1_LENGTH) index = W_R1_LENGTH - 1;
            if(index < 0) index = 0;

            wRegion += factor * index;
            factor *= W_R1_LENGTH;
        }

    if(V_DIM == 1 && C_DIM == 1)
        {
            return wRegion;
        }

    // --- Attitude ---
    int aRegion = 0;
    if(C_R1_LENGTH > 1)
        {
            factor = 1;
            for(int i = C_DIM - 1; i >= 0; --i)
                {
                    index = (int)(C_R1_LENGTH * (coord[i + W_DIM] - C_MIN) / (C_MAX - C_MIN));
                    if(index >= C_R1_LENGTH) index = C_R1_LENGTH - 1;
                    if(index < 0) index = 0;

                    aRegion += factor * index;
                    factor *= C_R1_LENGTH;
                }
        }

    // --- Velocity ---
    int vRegion = 0;
    if(V_R1_LENGTH > 1)
        {
            factor = 1;
            for(int i = V_DIM - 1; i >= 0; --i)
                {
                    index = (int)(V_R1_LENGTH * (coord[i + W_DIM + C_DIM] - V_MIN) / (V_MAX - V_MIN));
                    if(index >= V_R1_LENGTH) index = V_R1_LENGTH - 1;
                    if(index < 0) index = 0;

                    vRegion += factor * index;
                    factor *= V_R1_LENGTH;
                }
        }

    return wRegion * pow(C_R1_LENGTH, C_DIM) * pow(V_R1_LENGTH, V_DIM) + aRegion * pow(V_R1_LENGTH, V_DIM) + vRegion;
}
