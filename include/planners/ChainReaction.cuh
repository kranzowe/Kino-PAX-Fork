#pragma once
#include "planners/Planner.cuh"
#include "collisionCheck/spatialHash.cuh"
#include <cmath>

// ChainReaction: CountingStars with everything but the chain removed.
//
// One iteration:  frontier F  ->  B_eff candidate propagations, spread round-robin over F (thread t
// expands frontier node t mod F, so every node gets floor(B_eff/F) or ceil(B_eff/F) children)  ->
// every collision-free candidate is accepted  ->  the accepted nodes are the next frontier.
// A node leaves the frontier when expanded and never comes back.
//
// Taken from CountingStars unchanged: the collision-checked propagation + edgeCost, the frontier
// compaction (exclusive_scan + findInd), Part A's tree insertion and goal check, and
// getControlPathToGoal. Removed: Graph / per-region cost statistics, the accept passes and their
// histograms, the door bitmask, reactivation (Part B), the fan-out arrays, and the B ramp. The only
// edits to what remains are the thread->node mapping (t mod F, instead of kernel2's t / iterations)
// and B_eff = min(B, remaining tree capacity - 1), which replaces kernel2's near-full-tree logic.
// The planner is region-free, so the R1 discretization has no effect on it.
class ChainReaction : public Planner
{
public:
    ChainReaction();
    ~ChainReaction();

    void plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount) override;
    float planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount);
    void planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr);
    void resetPlanner(float* h_initial, float* h_goal);
    // Virtual so FissionReaction can replace the candidate-count rule and still be driven by plan*().
    virtual void propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount);
    virtual void updateFrontier();
    void getControlPathToGoal();
    void writeExecutionTimeToCSV(double time);

    // B: candidate propagations per iteration. Set by the caller before resetPlanner(); not reset.
    int h_budget_;

    uint h_frontierSize_, h_frontierNextSize_, h_solSetSize_;
    // Candidates actually launched this iteration (== min(B, remaining capacity - 1)).
    uint h_propAttempted_;
    // Largest branching factor this iteration, ceil(candidates / frontier). 0 means the run is over
    // (empty frontier or full tree); resetPlanner leaves it at 1, since the benchmark loops break on 0.
    uint h_propIterations_;
    float h_minCost_;
    float* h_controlPathsToGoal_;

    thrust::device_vector<bool> d_frontier_, d_frontierNext_;
    thrust::device_vector<uint> d_activeFrontierIdxs_, d_frontierScanIdx_;
    thrust::device_vector<int> d_unexploredSamplesParentIdxs_;
    thrust::device_vector<float> d_unexploredSamples_, d_unexploredSampleCosts_, d_goalSample_;
    thrust::device_vector<bool> d_goalSet_;
    thrust::device_vector<uint> d_goalSetIdxs_, d_goalSetScanIdx_;
    thrust::device_vector<int> d_iterations_;
    thrust::device_vector<float> d_controlPathsToGoal_;

    float *d_unexploredSamples_ptr_, *d_unexploredSampleCosts_ptr_, *d_goalSample_ptr_, *d_minCost_ptr_;
    float* d_controlPathsToGoal_ptr_;
    bool *d_frontier_ptr_, *d_frontierNext_ptr_, *d_goalSet_ptr_;
    uint *d_activeFrontierIdxs_ptr_, *d_frontierScanIdx_ptr_, *d_goalSetIdxs_ptr_, *d_goalSetScanIdx_ptr_;
    int *d_unexploredSamplesParentIdxs_ptr_, *d_iterations_ptr_;

    SpatialHashGrid* d_spatialHashGrid_;
    SpatialHashGrid h_spatialHashGrid_;
};

/**************************** DEVICE FUNCTIONS ****************************/

// One thread per candidate. Thread t propagates from frontier node t mod frontierSize and records
// the result iff it is collision-free; no other admission test exists.
__global__ void ChainReaction_propagateFrontier_kernel(uint* activeFrontierIdxs, uint frontierSize, uint numCandidates,
                                                       float* treeSamples, float* treeSampleCosts,
                                                       float* unexploredSamples, int* unexploredSamplesParentIdxs,
                                                       float* unexploredSampleCosts, bool* frontierNext,
                                                       curandState* randomSeeds, float* obstacles, int obstaclesCount,
                                                       SpatialHashGrid spatialHashGrid);

// Inserts every collision-free candidate into the tree (they are the next frontier) and marks goal nodes.
__global__ void ChainReaction_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs,
                                                    uint frontierNextSize, float* xGoal, int treeSize,
                                                    float* unexploredSamples, float* treeSamples,
                                                    int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs,
                                                    float* treeSampleCosts, float* unexploredSampleCosts,
                                                    float* minCost, bool* goalSet, int* iterations, int iteration);

__global__ void ChainReaction_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
                                                          int* treeSamplesParentIdxs, uint* goalSetIdxs, int goalSetSize,
                                                          float* treeSampleCosts, float* minCost);
