#pragma once
#include "planners/Planner.cuh"
#include "graphs/Graph.cuh"
#include "collisionCheck/spatialHash.cuh"

class KinoPaxSTARancestor : public Planner
{
public:
    /**************************** CONSTRUCTORS ****************************/
    KinoPaxSTARancestor();
    ~KinoPaxSTARancestor();

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
    float h_minCost_;
    float h_maxRegression_, h_explorationBias_, h_goalBias_;  // PruneKPAX goal-progress gate tunables

    // --- Ancestor-pruning tunables (KinoPaxPlus port; set in ctor, NOT reset by resetPlanner) ---
    //   h_ancestorPrune_     0 = off (reproduces stock KinoPaxSTAR exactly)
    //                        1 = node-only  (this node beaten in its own region)
    //                        2 = memoized ancestor chain (one parent lookup, see the .cu)
    //   h_dormancyThreshold_ iterations a pruned-but-region-best node must survive before it is
    //                        un-pruned. KinoPaxPlus hardcodes 5 (KinoPaxPlus.cu:541,548).
    //   h_ancestorTol_       slack before an ancestor counts as bad:
    //                        cost > minCostsR1[r] * (1 + tol). 0 == KinoPaxPlus's strict test.
    int   h_ancestorPrune_;
    int   h_dormancyThreshold_;
    float h_ancestorTol_;

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
    // Ancestor pruning: dormancy timer per node, and the memoized "some ancestor is bad" flag.
    thrust::device_vector<uint> d_treeInactiveIterations_;
    thrust::device_vector<bool> d_ancestorBad_;
    thrust::device_vector<uint> d_goalSetIdxs_, d_goalSetScanIdx_;
    thrust::device_vector<int> d_iterations_;
    thrust::device_vector<float> d_pathCosts_, d_controlPathsToGoal_;

    // --- raw pointers ---
    float *d_unexploredSamples_ptr_, *d_goalSample_ptr_, *d_unexploredSampleCosts_ptr_;
    float *d_minCostsR1_ptr_, *d_minCost_ptr_;
    float *d_pathCosts_ptr_, *d_controlPathsToGoal_ptr_;
    bool *d_frontier_ptr_, *d_frontierNext_ptr_, *d_goalSet_ptr_, *d_pruned_ptr_, *d_ancestorBad_ptr_;
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
__global__ void KinoPaxSTARancestor_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, int* frontierNextXR1s,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid);

/***************************/
/* PROPAGATE FRONTIER KERNEL 2 */
/***************************/
__global__ void KinoPaxSTARancestor_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, int iterations,
                                                   float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, int* frontierNextXR1s,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid);

/***************************/
/* ANCESTOR (TREE) PRUNING KERNEL */
/***************************/
// Port of KinoPaxPlus_pruningTree_kernel (KinoPaxPlus.cu:525-575): retroactively tombstones tree
// nodes whose path from the root passes through a region where a cheaper route has since been
// found. Unlike the original it does not walk the ancestor chain -- see the .cu for why one
// parent lookup is exactly equivalent. One thread per tree node.
__global__ void KinoPaxSTARancestor_pruningTree_kernel(int treeSize, int* treeSamplesParentIdxs,
                                                  float* treeSampleCosts, float* minCostsR1, int* treeXR1s,
                                                  bool* pruned, bool* ancestorBad, uint* inactiveIterations,
                                                  int ancestorPrune, int dormancyThreshold, float ancestorTol);

/***************************/
/* GOAL-PROGRESS PRUNING KERNEL */
/***************************/
// Min-cost (best-in-region) candidates are exempt; non-best candidates pass the
// PruneKPAX greedy-toward-goal probabilistic gate before insertion.
__global__ void KinoPaxSTARancestor_goalProgressPrune_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                                  float* minCostsR1, int* frontierNextXR1s, float* unexploredSampleCosts,
                                                  float* unexploredSamples, int* unexploredSamplesParentIdxs,
                                                  float* treeSamples, float* xGoal, bool* frontierNext,
                                                  curandState* randomSeeds, float maxRegression, float explorationBias,
                                                  float goalBias);

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
__global__ void
KinoPaxSTARancestor_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               uint* activeFrontierRepeatCount, int* validVertexCounter, curandState* randomSeeds,
                               float* vertexScores, float fAccept,
                               float maxRegression, float explorationBias, float goalBias,
                               float* minCostsR1, int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
                               float* minCost, float* unexploredSampleCosts, bool* goalSet, bool* pruned,
                               int* iterations, int iteration);

/***************************/
/* GET CONTROL PATH TO GOAL KERNEL */
/***************************/
__global__ void KinoPaxSTARancestor_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
                                                     int* treeSamplesParentIdxs, uint* goalSetIdxs, int goalSetSize,
                                                     float* pathCosts, float* treeSampleCosts, int* iterations,
                                                     float* minCost);
