#pragma once
#include "planners/Planner.cuh"
#include "graphs/Graph.cuh"

class KPAXCap : public Planner
{
public:
    /**************************** CONSTRUCTORS ****************************/
    KPAXCap();

    /****************************    METHODS    ****************************/
    void plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount) override;
    void planDebug(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount);
    void planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr);
    void resetPlanner(float* h_initial, float* h_goal);
    void propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount);
    void updateFrontier();
    void writeDeviceVectorsToCSV(int itr);
    void writeExecutionTimeToCSV(double time);

    /****************************    FIELDS    ****************************/
    // --- host fields ---
    Graph graph_;
    uint h_frontierSize_, h_frontierNextSize_, h_activeBlockSize_, h_frontierRepeatSize_, h_propIterations_;
    float h_fAccept_;

// ---- Syclop exploration cap: the ONLY difference from stock KPAX ----
// h_syclopCap_  multiplier in (0,1] on the Syclop region score, applied at BOTH acceptance points:
//               propagate uses  curand < cap * vertexScores[r] || !activeSubVertices[sub]
//               reactivation uses  curand <= cap * vertexScores[r] + fAccept
// fAccept is deliberately NOT scaled -- it is KPAX's additive reactivation floor, and multiplying
// it by a small cap would shut dormant-node revival off entirely rather than merely throttle it.
// The R2 seeding disjunct is NOT capped either: it is a separate free pass. Same contract as
// KinoPaxSTARTrue::h_syclopCap_.
//
// cap = 1.0 (the default) reproduces KPAX exactly and costs no RNG draw either way, so it is a
// pure no-op at the default. Set in the ctor and deliberately NOT touched by resetPlanner, so a
// benchmark can override it per run.
//
// WHY THIS EXISTS. KinoPaxSTARCleanCost at w = 1, cap = 1 is NOT KPAX: its accept kernel reads
// vertexScores AFTER graph_.updateVertices() has folded in this iteration's samples, and
// computeVertexScores_kernel divides by (1 + counterArray^2) with counterArray cumulative over the
// run -- so the regions the frontier currently occupies are judged on a score already penalised for
// the very batch being judged. KPAXCap isolates the other half of that difference (the cap alone,
// still decided inside propagate on pre-jump scores), so KPAX / KPAXCap / CleanCost-at-w=1
// separates "the cap did this" from "the kernel boundary did this".
    float h_syclopCap_;

    // --- device fields ---
    thrust::device_vector<bool> d_frontier_, d_frontierNext_;
    thrust::device_vector<uint> d_activeFrontierIdxs_, d_frontierScanIdx_, d_activeFrontierRepeatCount_, d_frontierRepeatScanIdx_,
      d_activeFrontierRepeatIdxs_;
    thrust::device_vector<int> d_unexploredSamplesParentIdxs_;
    thrust::device_vector<float> d_unexploredSamples_, d_goalSample_;
    float *d_unexploredSamples_ptr_, *d_goalSample_ptr_;
    bool *d_frontier_ptr_, *d_frontierNext_ptr_;
    uint *d_activeFrontierIdxs_ptr_, *d_frontierScanIdx_ptr_, *d_activeFrontierRepeatCount_ptr_, *d_frontierRepeatScanIdx_ptr_,
      *d_activeFrontierRepeatIdxs_ptr_;
    int* d_unexploredSamplesParentIdxs_ptr_;
};

/**************************** DEVICE FUNCTIONS ****************************/

/***************************/
/* PROPAGATE FRONTIER KERNEL 1 */
/***************************/
// --- Propagates current frontier. Builds new frontier. ---
// --- One Block Per Frontier Sample ---
__global__ void KPAXCap_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples, float* unexploredSamples,
                                          uint frontierSize, curandState* randomSeeds, int* unexploredSamplesParentIdxs, float* obstacles,
                                          int obstaclesCount, int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                          int* vertexCounter, int* validVertexCounter, float* minValueInRegion, float syclopCap);

__global__ void KPAXCap_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples, float* unexploredSamples,
                                          uint frontierSize, curandState* randomSeeds, int* unexploredSamplesParentIdxs, float* obstacles,
                                          int obstaclesCount, int* activeSubVertices, float* vertexScores, bool* frontierNext,
                                          int* vertexCounter, int* validVertexCounter, int iterations, float* minValueInRegion,
                                          float syclopCap);

__global__ void
KPAXCap_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize, float* xGoal, int treeSize,
                      float* unexploredSamples, float* treeSamples, int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs,
                      float* treeSampleCosts, int* pathToGoal, uint* activeFrontierRepeatCount, int* validVertexCounter,
                      curandState* randomSeeds, float* vertexScores, float* controlPathToGoal, float fAccept, float syclopCap);