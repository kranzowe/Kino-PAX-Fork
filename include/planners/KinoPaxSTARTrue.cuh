#pragma once
#include "planners/Planner.cuh"
#include "graphs/Graph.cuh"
#include "collisionCheck/spatialHash.cuh"

class KinoPaxSTARTrue : public Planner
{
public:
    /**************************** CONSTRUCTORS ****************************/
    KinoPaxSTARTrue();
    ~KinoPaxSTARTrue();

    /****************************    METHODS    ****************************/
    void plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount) override;
    float planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount);
    void planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr);
    void resetPlanner(float* h_initial, float* h_goal);
    void propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount);
    void updateFrontier();
    void getControlPathToGoal();
    void writeExecutionTimeToCSV(double time);

    /****************************    FIELDS    ****************************/
    // --- host fields ---
    Graph graph_;
    uint h_frontierSize_, h_frontierNextSize_, h_activeBlockSize_, h_frontierRepeatSize_, h_propIterations_, h_solSetSize_;
    float h_fAccept_;

// ---- Cost pruning, GUARDED to cost-admitted nodes ----
// h_ancestorPrune_     0 = off (reproduces stock KinoPaxSTARNoGoalBias exactly)
//                      1 = stale-best: a node admitted because it was its region's minimum and
//                          is no longer the minimum
//                      2 = 1, plus the memoized ancestor chain over the same population
// h_dormancyThreshold_ iterations a pruned-but-region-best node survives before un-pruning (5)
// h_ancestorTol_       slack: cost > minCostsR1[r] * (1 + tol). 0 = KinoPaxPlus's strict test.
//
// THE GUARD IS THE POINT. The predecessor applied this test to every tree node, and
// Syclop-admitted nodes are non-minimum BY CONSTRUCTION (they were admitted despite failing
// isBest), so every one of them was tombstoned on the first pass and -- since Part B returns
// early on pruned[] -- never reactivated. That froze the entire exploration population.
// KinoPaxPlus gets away with the same rule because its pruningFrontier_kernel hard-rejects
// cost > minCostsR1 at insertion, so its tree is almost entirely min-cost nodes to begin with.
    int   h_ancestorPrune_;
    int   h_dormancyThreshold_;
    float h_ancestorTol_;
    float h_minCost_;
    float* h_controlPathsToGoal_;

    // --- device fields (KPAX exploration) ---
    thrust::device_vector<bool> d_frontier_, d_frontierNext_;
    thrust::device_vector<uint> d_activeFrontierIdxs_, d_frontierScanIdx_, d_activeFrontierRepeatCount_,
      d_frontierRepeatScanIdx_, d_activeFrontierRepeatIdxs_;
    thrust::device_vector<int> d_unexploredSamplesParentIdxs_;
    thrust::device_vector<float> d_unexploredSamples_, d_goalSample_;

    // --- device fields (KinoPaxPlus optimization) ---
    thrust::device_vector<float> d_minCostsR1_;
    thrust::device_vector<float> d_unexploredSampleCosts_;
    thrust::device_vector<int> d_bestNodeIdxPerR1_;
    thrust::device_vector<int> d_treeXR1s_, d_frontierNextXR1s_;
    thrust::device_vector<bool> d_goalSet_, d_pruned_;
    // Admission reason: true iff the node was admitted ONLY because it was its region's minimum.
    // Written per unexplored sample in propagate, copied to the tree in the update kernel.
    thrust::device_vector<bool> d_frontierNextAdmitBest_, d_treeAdmitBest_;
    thrust::device_vector<uint> d_treeInactiveIterations_;
    thrust::device_vector<bool> d_ancestorBad_;
    thrust::device_vector<uint> d_goalSetIdxs_, d_goalSetScanIdx_;
    thrust::device_vector<int> d_iterations_;
    thrust::device_vector<float> d_pathCosts_, d_controlPathsToGoal_;

    // --- raw pointers ---
    float *d_unexploredSamples_ptr_, *d_goalSample_ptr_, *d_unexploredSampleCosts_ptr_;
    float *d_minCostsR1_ptr_, *d_minCost_ptr_;
    float *d_pathCosts_ptr_, *d_controlPathsToGoal_ptr_;
    bool *d_frontier_ptr_, *d_frontierNext_ptr_, *d_goalSet_ptr_, *d_pruned_ptr_;
    bool *d_frontierNextAdmitBest_ptr_, *d_treeAdmitBest_ptr_, *d_ancestorBad_ptr_;
    uint *d_treeInactiveIterations_ptr_;
    uint *d_activeFrontierIdxs_ptr_, *d_frontierScanIdx_ptr_, *d_activeFrontierRepeatCount_ptr_,
      *d_frontierRepeatScanIdx_ptr_, *d_activeFrontierRepeatIdxs_ptr_;
    uint *d_goalSetIdxs_ptr_, *d_goalSetScanIdx_ptr_;
    int *d_unexploredSamplesParentIdxs_ptr_, *d_treeXR1s_ptr_, *d_frontierNextXR1s_ptr_;
    int *d_bestNodeIdxPerR1_ptr_, *d_iterations_ptr_;

    // --- Spatial hash grid for collision detection ---
    SpatialHashGrid* d_spatialHashGrid_;
    SpatialHashGrid h_spatialHashGrid_;
};

/**************************** DEVICE FUNCTIONS ****************************/

/***************************/
/* PROPAGATE FRONTIER KERNEL 1 */
/***************************/
__global__ void KinoPaxSTARTrue_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, int* frontierNextXR1s,
                                                   bool* frontierNextAdmitBest,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid);

/***************************/
/* PROPAGATE FRONTIER KERNEL 2 */
/***************************/
__global__ void KinoPaxSTARTrue_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, int iterations,
                                                   float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, int* frontierNextXR1s,
                                                   bool* frontierNextAdmitBest,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid);

/***************************/
/* COST PRUNING KERNEL (guarded to cost-admitted nodes) */
/***************************/
__global__ void KinoPaxSTARTrue_pruningTree_kernel(int treeSize, int* treeSamplesParentIdxs,
                                                  float* treeSampleCosts, float* minCostsR1, int* treeXR1s,
                                                  bool* admitBest, bool* pruned, bool* ancestorBad,
                                                  uint* inactiveIterations,
                                                  int ancestorPrune, int dormancyThreshold, float ancestorTol);

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
__global__ void
KinoPaxSTARTrue_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               uint* activeFrontierRepeatCount, int* validVertexCounter, curandState* randomSeeds,
                               float* vertexScores, float fAccept,
                               float* minCostsR1, int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
                               bool* frontierNextAdmitBest, bool* treeAdmitBest,
                               float* minCost, float* unexploredSampleCosts, bool* goalSet, bool* pruned,
                               int* iterations, int iteration);

/***************************/
/* GET CONTROL PATH TO GOAL KERNEL */
/***************************/
__global__ void KinoPaxSTARTrue_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
                                                     int* treeSamplesParentIdxs, uint* goalSetIdxs, int goalSetSize,
                                                     float* pathCosts, float* treeSampleCosts, int* iterations,
                                                     float* minCost);
