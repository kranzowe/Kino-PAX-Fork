#pragma once
#include "planners/Planner.cuh"
#include "graphs/KinoPaxPlusRegions.cuh"
#include "collisionCheck/spatialHash.cuh"

class KinoPaxPlusSpatialHash : public Planner
{
public:
    /**************************** CONSTRUCTORS ****************************/
    KinoPaxPlusSpatialHash();
    ~KinoPaxPlusSpatialHash();

    /****************************    METHODS    ****************************/
    void plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount) override;
    float planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount);
    void planBenchmark(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr);
    void planPathsCollect(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr);
    void planDataCollect(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr);
    void propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount);
    void updateFrontier();
    void resetPlanner(float* h_initial, float* h_goal);
    void getControlPathToGoal();
    void getControlPathToGoalPathsCollect();
    void writeDeviceVectorsToCSV(int itr);
    void writeSolutionsToCSV(int benchItr = 0);
    void writeSolutionCostsToCSV(int benchItr = 0);
    void writeIterationTimeToCSV(std::vector<float>& iterationTimes, int benchItr);
    void writeExecutionTimeToCSV(double time);

    /****************************    FIELDS    ****************************/
    // --- host fields ---
    KinoPaxPlusRegions graph_;
    uint h_frontierSize_, h_frontierNextSize_, h_activeBlockSize_, h_propIterations_, h_solSetSize_;
    float h_minCost_;
    int h_previousExpansionCount_;
    int h_addedNodes_;
    float* h_controlPathsToGoal_;

    // --- device fields ---
    thrust::device_vector<bool> d_frontier_, d_frontierNext_, d_goalSet_, d_pruned_;
    thrust::device_vector<uint> d_activeFrontierIdxs_, d_goalSetIdxs_, d_frontierScanIdx_, d_goalSetScanIdx_;
    thrust::device_vector<uint> d_treeInactiveIterations_;
    thrust::device_vector<int> d_unexploredSamplesParentIdxs_, d_treeXR1s_, d_frontierNextXR1s_;
    thrust::device_vector<int> d_iterations_;
    thrust::device_vector<float> d_unexploredSamples_, d_goalSample_, d_unexploredSampleCosts_;
    thrust::device_vector<float> d_pathCosts_, d_controlPathsToGoal_;

    float *d_unexploredSamples_ptr_, *d_goalSample_ptr_, *d_unexploredSampleCosts_ptr_;
    float *d_pathCosts_ptr_, *d_controlPathsToGoal_ptr_, *d_minCost_ptr_;
    bool *d_frontier_ptr_, *d_frontierNext_ptr_, *d_goalSet_ptr_, *d_pruned_ptr_;
    uint *d_activeFrontierIdxs_ptr_, *d_goalSetIdxs_ptr_, *d_frontierScanIdx_ptr_, *d_goalSetScanIdx_ptr_;
    uint* d_treeInactiveIterations_ptr_;
    int *d_unexploredSamplesParentIdxs_ptr_, *d_treeXR1s_ptr_, *d_frontierNextXR1s_ptr_;
    int* d_iterations_ptr_;

    // --- Spatial hash grid for collision detection ---
    SpatialHashGrid* d_spatialHashGrid_;
    SpatialHashGrid h_spatialHashGrid_;
};

/**************************** DEVICE FUNCTIONS ****************************/

/***************************/
/* PROPAGATE FRONTIER KERNEL 1 */
/***************************/
// --- Propagates current frontier. Builds new frontier. ---
// --- One Block Per Frontier Sample ---
__global__ void KinoPaxPlusSH_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                bool* frontierNext, float* treeSampleCosts, float* minCostsR1,
                                                int* frontierNextXR1s, float* unexploredSampleCosts,
                                                SpatialHashGrid spatialHashGrid);

__global__ void KinoPaxPlusSH_propagateFrontier_kernel1V2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   bool* frontierNext, float* treeSampleCosts, float* minCostsR1,
                                                   int* frontierNextXR1s, float* unexploredSampleCosts, int bf,
                                                   SpatialHashGrid spatialHashGrid);

/***************************/
/* PROPAGATE FRONTIER KERNEL 2 */
/***************************/
// --- Iterations new samples per frontier sample ---
__global__ void KinoPaxPlusSH_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                bool* frontierNext, int iterations, float* treeSampleCosts,
                                                float* minCostsR1, int* frontierNextXR1s, float* unexploredSampleCosts,
                                                SpatialHashGrid spatialHashGrid);

/***************************/
/* TREE PRUNING KERNEL */
/***************************/
__global__ void KinoPaxPlusSH_pruningTree_kernel(int treeSize, int* treeSamplesParentIdxs, float* treeSampleCosts, bool* goalSet,
                                         float* minCostsR1, int* treeXR1s, bool* pruned, uint* inactiveIterations);

/***************************/
/* FRONTIER PRUNING KERNEL */
/***************************/
__global__ void KinoPaxPlusSH_pruningFrontier_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                             int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs,
                                             float* treeSampleCosts, float* minCostsR1, int* frontierNextXR1s,
                                             int* treeXR1s, bool* frontierNext, float* unexploredSampleCosts,
                                             bool* pruned, uint* inactiveIterations);

/***************************/
/* COMBINED PRUNING KERNEL */
/***************************/
__global__ void KinoPaxPlusSH_pruning_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize, int treeSize,
                                     int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs,
                                     float* treeSampleCosts, bool* goalSet, float* minCostsR1, int* treeXR1s,
                                     int* frontierNextXR1s, bool* frontierNext, float* unexploredSampleCosts,
                                     bool* pruned);

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// --- Adds previous frontier to the tree and builds new frontier. ---
__global__ void KinoPaxPlusSH_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs,
                                            uint frontierNextSize, float* xGoal, int treeSize, float* unexploredSamples,
                                            float* treeSamples, int* unexploredSamplesParentIdxs,
                                            int* treeSamplesParentIdxs, float* treeSampleCosts, curandState* randomSeeds,
                                            float* controlPathToGoal, bool* goalSet, int* iterations, int iteration,
                                            float* minCostsR1, int* treeXR1s, int* frontierNextXR1s, float* minCost,
                                            float* unexploredSampleCosts, bool* pruned);

/***************************/
/* GET CONTROL PATH TO GOAL KERNEL */
/***************************/
__global__ void KinoPaxPlusSH_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
                                                  int* treeSamplesParentIdxs, uint* goalSetIdxs, int goalSetSize,
                                                  float* pathCosts, float* treeSampleCosts, int* iterations,
                                                  float* minCost);

/***************************/
/* GET CONTROL PATH TO GOAL PATHS COLLECT KERNEL */
/***************************/
__global__ void KinoPaxPlusSH_getControlPathToGoalPathsCollect_kernel(float* controlPathsToGoal, float* treeSamples,
                                                              int* treeSamplesParentIdxs, uint* goalSetIdxs,
                                                              int goalSetSize, float* pathCosts,
                                                              float* treeSampleCosts, int* iterations,
                                                              float* minCost, int itr, int numSols);

/***************************/
/* KinoPaxPlusSH GET REGION DEVICE FUNCTION */
/***************************/
__device__ int KinoPaxPlusSH_getRegion(float* coord);
