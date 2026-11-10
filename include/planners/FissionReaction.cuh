#pragma once
#include "planners/ChainReaction.cuh"

// FissionReaction: ChainReaction without a budget. Every frontier node gets exactly h_branching_
// children, so an iteration launches |frontier| * k candidates; every collision-free candidate is
// accepted and becomes the next frontier. The frontier therefore grows ~(collision-free fraction * k)
// per iteration, i.e. exponentially, and the tree fills in a handful of iterations for larger k.
//
// Everything else is inherited from ChainReaction: construction, resetPlanner, Part A (tree insertion
// and goal check), getControlPathToGoal, the plan*() loops. Only the candidate-count rule and the
// thread -> frontier-node mapping (t / k instead of ChainReaction's t mod F) differ.
//
// Capacity: if |frontier| * k does not fit in the remaining tree capacity, that iteration expands a
// prefix of the frontier (index order, each node with exactly k children) and the run then ends.
class FissionReaction : public ChainReaction
{
public:
    FissionReaction();

    void propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount) override;
    void updateFrontier() override;

    // k: children per frontier node. Set by the caller before resetPlanner(); not reset. h_budget_ is unused.
    int h_branching_;
    // True when this iteration's candidates were cut to the remaining capacity (the last iteration).
    bool h_truncated_;
};

/**************************** DEVICE FUNCTIONS ****************************/

// One thread per candidate; thread t propagates from frontier node t / branching.
__global__ void FissionReaction_propagateFrontier_kernel(uint* activeFrontierIdxs, uint branching, uint numCandidates,
                                                         float* treeSamples, float* treeSampleCosts,
                                                         float* unexploredSamples, int* unexploredSamplesParentIdxs,
                                                         float* unexploredSampleCosts, bool* frontierNext,
                                                         curandState* randomSeeds, float* obstacles, int obstaclesCount,
                                                         SpatialHashGrid spatialHashGrid);
