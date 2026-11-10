// ChainReaction -- see include/planners/ChainReaction.cuh for what it keeps and drops from CountingStars.
#include "planners/ChainReaction.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <sstream>

ChainReaction::ChainReaction()
{
    d_frontier_                    = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_frontierNext_                = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_activeFrontierIdxs_          = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_unexploredSamples_           = thrust::device_vector<float>(MAX_TREE_SIZE * SAMPLE_DIM);
    d_unexploredSamplesParentIdxs_ = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_unexploredSampleCosts_       = thrust::device_vector<float>(MAX_TREE_SIZE);
    d_frontierScanIdx_             = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_goalSample_                  = thrust::device_vector<float>(SAMPLE_DIM);
    d_goalSet_                     = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_goalSetIdxs_                 = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_goalSetScanIdx_              = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_iterations_                  = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_controlPathsToGoal_          = thrust::device_vector<float>(MAX_ITER * SAMPLE_DIM);

    d_frontier_ptr_                    = thrust::raw_pointer_cast(d_frontier_.data());
    d_frontierNext_ptr_                = thrust::raw_pointer_cast(d_frontierNext_.data());
    d_activeFrontierIdxs_ptr_          = thrust::raw_pointer_cast(d_activeFrontierIdxs_.data());
    d_unexploredSamples_ptr_           = thrust::raw_pointer_cast(d_unexploredSamples_.data());
    d_unexploredSamplesParentIdxs_ptr_ = thrust::raw_pointer_cast(d_unexploredSamplesParentIdxs_.data());
    d_unexploredSampleCosts_ptr_       = thrust::raw_pointer_cast(d_unexploredSampleCosts_.data());
    d_frontierScanIdx_ptr_             = thrust::raw_pointer_cast(d_frontierScanIdx_.data());
    d_goalSample_ptr_                  = thrust::raw_pointer_cast(d_goalSample_.data());
    d_goalSet_ptr_                     = thrust::raw_pointer_cast(d_goalSet_.data());
    d_goalSetIdxs_ptr_                 = thrust::raw_pointer_cast(d_goalSetIdxs_.data());
    d_goalSetScanIdx_ptr_              = thrust::raw_pointer_cast(d_goalSetScanIdx_.data());
    d_iterations_ptr_                  = thrust::raw_pointer_cast(d_iterations_.data());
    d_controlPathsToGoal_ptr_          = thrust::raw_pointer_cast(d_controlPathsToGoal_.data());

    cudaMalloc(&d_minCost_ptr_, sizeof(float));

    d_spatialHashGrid_ = createSpatialHashGrid();

    h_controlPathsToGoal_ = new float[MAX_ITER * SAMPLE_DIM];

    h_budget_           = 5000;
    h_frontierSize_     = 0;
    h_frontierNextSize_ = 0;
    h_propAttempted_    = 0;
    h_propIterations_   = 1;
    h_solSetSize_       = 0;
    h_minCost_          = MAX_FLOAT;

    if(VERBOSE)
        {
            printf("/* Planner Type: ChainReaction (budget B, accept = collision-free) */\n");
            printf("/***************************/\n");
        }
}

ChainReaction::~ChainReaction()
{
    destroySpatialHashGrid(d_spatialHashGrid_);
    cudaFree(d_minCost_ptr_);
    delete[] h_controlPathsToGoal_;
}

void ChainReaction::resetPlanner(float* h_initial, float* h_goal)
{
    thrust::fill(d_frontier_.begin(), d_frontier_.end(), false);
    thrust::fill(d_frontierNext_.begin(), d_frontierNext_.end(), false);
    thrust::fill(d_activeFrontierIdxs_.begin(), d_activeFrontierIdxs_.end(), 0);
    thrust::fill(d_unexploredSamples_.begin(), d_unexploredSamples_.end(), 0.0f);
    thrust::fill(d_unexploredSamplesParentIdxs_.begin(), d_unexploredSamplesParentIdxs_.end(), -1);
    thrust::fill(d_unexploredSampleCosts_.begin(), d_unexploredSampleCosts_.end(), 0.0f);
    thrust::fill(d_frontierScanIdx_.begin(), d_frontierScanIdx_.end(), 0);
    thrust::fill(d_goalSample_.begin(), d_goalSample_.end(), 0.0f);
    thrust::fill(d_goalSet_.begin(), d_goalSet_.end(), false);
    thrust::fill(d_iterations_.begin(), d_iterations_.end(), 0);
    thrust::fill(d_controlPathsToGoal_.begin(), d_controlPathsToGoal_.end(), 0.0f);

    thrust::fill(d_treeSamples_.begin(), d_treeSamples_.end(), 0.0f);
    thrust::fill(d_treeSamplesParentIdxs_.begin(), d_treeSamplesParentIdxs_.end(), -1);
    thrust::fill(d_treeSampleCosts_.begin(), d_treeSampleCosts_.end(), 0.0f);
    thrust::fill(d_frontier_.begin(), d_frontier_.begin() + 1, true);

    h_treeSize_         = 1;
    h_itr_              = 0;
    h_costToGoal_       = 0;
    h_pathToGoal_       = 0;
    h_frontierSize_     = 0;
    h_frontierNextSize_ = 0;
    h_propAttempted_    = 0;
    h_solSetSize_       = 0;
    h_minCost_          = MAX_FLOAT;
    // Nonzero before iteration 1: the benchmark loops break on h_propIterations_ == 0.
    h_propIterations_   = 1;
    if(h_budget_ < 1) h_budget_ = 1;

    cudaMemcpy(d_treeSamples_ptr_, h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costToGoal_ptr_, &h_costToGoal_, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pathToGoal_ptr_, &h_pathToGoal_, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minCost_ptr_, &h_minCost_, sizeof(float), cudaMemcpyHostToDevice);

    initializeRandomSeeds(static_cast<unsigned int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
}

void ChainReaction::plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
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

            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "ChainReaction execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ChainReaction::planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr)
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
    std::cout << "ChainReaction execution time: " << executionTime << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
}

float ChainReaction::planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
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
            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "ChainReaction execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return h_minCost_;
}

void ChainReaction::propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount)
{
    updateSpatialHashGrid(d_spatialHashGrid_, d_obstacles_ptr, h_obstaclesCount);
    cudaMemcpy(&h_spatialHashGrid_, d_spatialHashGrid_, sizeof(SpatialHashGrid), cudaMemcpyDeviceToHost);

    thrust::exclusive_scan(d_frontier_.begin(), d_frontier_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontier_[MAX_TREE_SIZE - 1]) ? ++h_frontierSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontier_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // Every frontier node is expanded this iteration and leaves the frontier; the kernel reads the
    // compacted list above, so the bits can be cleared now. Bits at or beyond h_treeSize_ are already false.
    thrust::fill(d_frontier_.begin(), d_frontier_.begin() + h_treeSize_, false);

    // One slot of margin below capacity so accepting every candidate can never reach MAX_TREE_SIZE.
    uint remaining = (h_treeSize_ + 1 < (uint)MAX_TREE_SIZE) ? (uint)MAX_TREE_SIZE - h_treeSize_ - 1 : 0;
    uint candidates = std::min((uint)h_budget_, remaining);

    if(h_frontierSize_ == 0 || candidates == 0)
        {
            h_propIterations_ = 0;
            h_propAttempted_  = 0;
            return;
        }

    h_propAttempted_  = candidates;
    h_propIterations_ = (candidates + h_frontierSize_ - 1) / h_frontierSize_;

    ChainReaction_propagateFrontier_kernel<<<iDivUp(candidates, h_blockSize_), h_blockSize_>>>(
      d_activeFrontierIdxs_ptr_, h_frontierSize_, candidates, d_treeSamples_ptr_, d_treeSampleCosts_ptr_,
      d_unexploredSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_unexploredSampleCosts_ptr_,
      d_frontierNext_ptr_, d_randomSeeds_ptr_, d_obstacles_ptr, h_obstaclesCount, h_spatialHashGrid_);
}

__global__ void ChainReaction_propagateFrontier_kernel(uint* activeFrontierIdxs, uint frontierSize, uint numCandidates,
                                                       float* treeSamples, float* treeSampleCosts,
                                                       float* unexploredSamples, int* unexploredSamplesParentIdxs,
                                                       float* unexploredSampleCosts, bool* frontierNext,
                                                       curandState* randomSeeds, float* obstacles, int obstaclesCount,
                                                       SpatialHashGrid spatialHashGrid)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= numCandidates) return;

    int x0Idx    = activeFrontierIdxs[tid % frontierSize];
    float x0Cost = treeSampleCosts[x0Idx];
    float* x0    = &treeSamples[x0Idx * SAMPLE_DIM];

    float* x1                        = &unexploredSamples[tid * SAMPLE_DIM];
    unexploredSamplesParentIdxs[tid] = x0Idx;
    curandState randSeed             = randomSeeds[tid];
    bool valid                       = propagateAndCheckSpatialHash(x0, x1, &randSeed, spatialHashGrid, obstacles, obstaclesCount);

    if(valid)
        {
            unexploredSampleCosts[tid] = x0Cost + edgeCost(x0, x1);
            frontierNext[tid]          = true;
        }

    randomSeeds[tid] = randSeed;
}

void ChainReaction::updateFrontier()
{
    if(h_propIterations_ == 0)
        {
            h_frontierNextSize_ = 0;
            return;
        }

    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontierNext_[MAX_TREE_SIZE - 1]) ? ++h_frontierNextSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // Unreachable while propagateFrontier caps the candidate count below capacity; kept as CountingStars does.
    if(h_treeSize_ + h_frontierNextSize_ >= MAX_TREE_SIZE)
        {
            h_propIterations_ = 0;
            return;
        }

    // No collision-free candidate: nothing becomes frontier, and nothing can ever revive the run.
    if(h_frontierNextSize_ == 0)
        {
            h_propIterations_ = 0;
            return;
        }

    ChainReaction_updateFrontier_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
      d_frontier_ptr_, d_frontierNext_ptr_, d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_goalSample_ptr_, h_treeSize_,
      d_unexploredSamples_ptr_, d_treeSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, d_unexploredSampleCosts_ptr_, d_minCost_ptr_, d_goalSet_ptr_, d_iterations_ptr_, h_itr_);

    cudaMemcpy(&h_minCost_, d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);

    h_treeSize_ += h_frontierNextSize_;
}

__global__ void ChainReaction_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs,
                                                    uint frontierNextSize, float* xGoal, int treeSize,
                                                    float* unexploredSamples, float* treeSamples,
                                                    int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs,
                                                    float* treeSampleCosts, float* unexploredSampleCosts,
                                                    float* minCost, bool* goalSet, int* iterations, int iteration)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_xGoal[SAMPLE_DIM];
    if(threadIdx.x < SAMPLE_DIM) s_xGoal[threadIdx.x] = xGoal[threadIdx.x];
    __syncthreads();

    if(tid >= frontierNextSize) return;

    int x1TreeIdx       = treeSize + tid;
    int x1UnexploredIdx = activeFrontierNextIdxs[tid];
    frontierNext[x1UnexploredIdx] = false;

    float* x1  = &unexploredSamples[x1UnexploredIdx * SAMPLE_DIM];
    int x0Idx  = unexploredSamplesParentIdxs[x1UnexploredIdx];
    float cost = unexploredSampleCosts[x1UnexploredIdx];

    treeSamplesParentIdxs[x1TreeIdx] = x0Idx;
    for(int i = 0; i < SAMPLE_DIM; i++)
        treeSamples[x1TreeIdx * SAMPLE_DIM + i] = x1[i];
    treeSampleCosts[x1TreeIdx] = cost;

    frontier[x1TreeIdx] = true;

    // Goal nodes are collected into goalSet (the min-cost path is rebuilt by getControlPathToGoal)
    // and are not expanded further.
    if(distance(x1, s_xGoal) < GOAL_THRESH && cost <= *minCost)
        {
            atomicMinFloat(minCost, cost);
            goalSet[x1TreeIdx]    = true;
            frontier[x1TreeIdx]   = false;
            iterations[x1TreeIdx] = iteration;
        }
}

void ChainReaction::getControlPathToGoal()
{
    thrust::exclusive_scan(d_goalSet_.begin(), d_goalSet_.end(), d_goalSetScanIdx_.begin(), 0, thrust::plus<uint>());
    h_solSetSize_ = d_goalSetScanIdx_[MAX_TREE_SIZE - 1];
    (d_goalSet_[MAX_TREE_SIZE - 1]) ? ++h_solSetSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_goalSet_ptr_, d_goalSetScanIdx_ptr_, d_goalSetIdxs_ptr_);

    if(h_solSetSize_ == 0) return;

    ChainReaction_getControlPathToGoal_kernel<<<iDivUp(h_solSetSize_, h_blockSize_), h_blockSize_>>>(
      d_controlPathsToGoal_ptr_, d_treeSamples_ptr_, d_treeSamplesParentIdxs_ptr_, d_goalSetIdxs_ptr_, h_solSetSize_,
      d_treeSampleCosts_ptr_, d_minCost_ptr_);

    cudaMemcpy(&h_minCost_, d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_controlPathsToGoal_, d_controlPathsToGoal_ptr_, MAX_ITER * SAMPLE_DIM * sizeof(float),
               cudaMemcpyDeviceToHost);
    printf("Cost to Goal: %f\n", h_minCost_);
}

__global__ void ChainReaction_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
                                                          int* treeSamplesParentIdxs, uint* goalSetIdxs, int goalSetSize,
                                                          float* treeSampleCosts, float* minCost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= MAX_TREE_SIZE || tid >= goalSetSize) return;

    int goalIdx = goalSetIdxs[tid];
    int x0Idx   = goalIdx;
    float cost  = treeSampleCosts[goalIdx];

    if(cost != *minCost) return;
    int i = 0;
    while(x0Idx != -1 && i < MAX_ITER)
        {
            for(int j = 0; j < SAMPLE_DIM; j++)
                controlPathsToGoal[SAMPLE_DIM * i + j] = treeSamples[x0Idx * SAMPLE_DIM + j];
            i++;
            x0Idx = treeSamplesParentIdxs[x0Idx];
        }
}

void ChainReaction::writeExecutionTimeToCSV(double time)
{
    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/ExecutionTime");
    filename.str("");
    filename << "Data/ExecutionTime/executionTime.csv";
    writeValueToCSV(time, filename.str());
}
