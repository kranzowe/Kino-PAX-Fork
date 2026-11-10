// FissionReaction -- see include/planners/FissionReaction.cuh. Differs from ChainReaction only in how many
// candidates an iteration launches (|frontier| * k) and which frontier node each candidate expands (t / k).
#include "planners/FissionReaction.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <algorithm>

FissionReaction::FissionReaction()
{
    h_branching_ = 2;
    h_truncated_ = false;

    if(VERBOSE)
        {
            printf("/* Planner Type: FissionReaction (fixed branching factor, accept = collision-free) */\n");
            printf("/***************************/\n");
        }
}

void FissionReaction::propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount)
{
    updateSpatialHashGrid(d_spatialHashGrid_, d_obstacles_ptr, h_obstaclesCount);
    cudaMemcpy(&h_spatialHashGrid_, d_spatialHashGrid_, sizeof(SpatialHashGrid), cudaMemcpyDeviceToHost);

    thrust::exclusive_scan(d_frontier_.begin(), d_frontier_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontier_[MAX_TREE_SIZE - 1]) ? ++h_frontierSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontier_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // Every frontier node is expanded this iteration and leaves the frontier (see ChainReaction).
    thrust::fill(d_frontier_.begin(), d_frontier_.begin() + h_treeSize_, false);

    uint k = (uint)std::max(1, h_branching_);
    // One slot of margin below capacity so accepting every candidate can never reach MAX_TREE_SIZE.
    uint remaining = (h_treeSize_ + 1 < (uint)MAX_TREE_SIZE) ? (uint)MAX_TREE_SIZE - h_treeSize_ - 1 : 0;
    unsigned long long wanted = (unsigned long long)h_frontierSize_ * k;
    uint candidates = (uint)std::min<unsigned long long>(wanted, (unsigned long long)remaining);

    h_truncated_ = (candidates < wanted);

    if(h_frontierSize_ == 0 || candidates == 0)
        {
            h_propIterations_ = 0;
            h_propAttempted_  = 0;
            return;
        }

    h_propAttempted_  = candidates;
    h_propIterations_ = k;

    FissionReaction_propagateFrontier_kernel<<<iDivUp(candidates, h_blockSize_), h_blockSize_>>>(
      d_activeFrontierIdxs_ptr_, k, candidates, d_treeSamples_ptr_, d_treeSampleCosts_ptr_,
      d_unexploredSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_unexploredSampleCosts_ptr_,
      d_frontierNext_ptr_, d_randomSeeds_ptr_, d_obstacles_ptr, h_obstaclesCount, h_spatialHashGrid_);
}

__global__ void FissionReaction_propagateFrontier_kernel(uint* activeFrontierIdxs, uint branching, uint numCandidates,
                                                         float* treeSamples, float* treeSampleCosts,
                                                         float* unexploredSamples, int* unexploredSamplesParentIdxs,
                                                         float* unexploredSampleCosts, bool* frontierNext,
                                                         curandState* randomSeeds, float* obstacles, int obstaclesCount,
                                                         SpatialHashGrid spatialHashGrid)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= numCandidates) return;

    int x0Idx    = activeFrontierIdxs[tid / branching];
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

void FissionReaction::updateFrontier()
{
    ChainReaction::updateFrontier();
    // The capacity-truncated iteration is the last one: its accepted nodes are inserted above, then the run ends.
    if(h_truncated_) h_propIterations_ = 0;
}
