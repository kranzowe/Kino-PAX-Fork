#pragma once
#include <stdio.h>
#include <vector>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/count.h>
#include "helper/helper.cuh"

// Predicate for counting active regions (validCounterArray[r] > 0). A named functor rather than a
// thrust placeholder so it compiles the same way under every nvcc this repo targets.
//
// Lives in the header rather than file-local in Graph.cu because planners need it too:
// KinoPaxSTARCOMBO reduces d_regionCoverage_ over the SAME active set that produces the dynamic
// score floor, and a second private copy of the predicate is how those two silently drift apart.
struct IsActiveRegion
{
    __host__ __device__ bool operator()(int v) const { return v > 0; }
};

class Graph
{
public:
    // --- constructor ---
    Graph() = default;
    Graph(float h_ws);

    // --- host fields ---
    int h_numPartialSums_;
    int h_blockSize_ = 32;

    // ---- Syclop score floor ----
    // updateSampleAcceptance_kernel sets vertexScore = floor + score/total, and the score/total
    // shares sum to exactly 1 across ACTIVE regions -- so the mean share is 1/N_active.
    //
    // LEGACY (default): floor = EPSILON = 1e-2. At 27k active regions the mean share is 3.7e-5, so
    // the floor exceeds the thing it floors by ~270x. Because the shares sum to 1, at most
    // 1/EPSILON = 100 regions can rise above the floor AT ANY GRID SIZE; every other region sits at
    // exactly 0.01, and refining the grid makes it worse (same budget of 1.0, split more ways).
    //
    // DYNAMIC: floor = 1/N_active, i.e. the mean share itself. An average region then gets exactly
    // 2x the floor regardless of discretization, and the fraction of regions that discriminate stops
    // collapsing as the grid is refined.
    //
    // Opt-in per planner rather than global: KPAX must stay a fixed historical baseline, so it keeps
    // the legacy floor while KPAXCap / KinoPaxSTARTrue / KinoPaxSTARCleanCost take the dynamic one.
    // The count_if only runs in dynamic mode, so legacy planners pay nothing.
    bool  h_dynamicScoreFloor_ = false;
    float h_scoreFloor_        = EPSILON;   // value actually used last updateVertices(); logged

    // Active-region count from the last updateVertices(). A by-product of the dynamic score floor's
    // count_if, kept rather than discarded because it is exactly the denominator a planner needs to
    // turn a sum over d_regionCoverage_ into a mean over EXPLORED regions. ONLY VALID IN DYNAMIC
    // MODE -- legacy-floor planners (KPAX) skip the count entirely and leave this at 0.
    int   h_nActive_ = 0;

    // --- device fields ---
    thrust::device_vector<int> d_validCounterArray_, d_counterArray_, d_activeVerticesScanIdx_, d_activeSubVertices_;
    thrust::device_vector<float> d_vertexScoreArray_, d_minValueInRegion_, d_partialSums_, d_totalScore_;

    // Per-R1 fraction of sub-regions ever touched, in [0,1]. computeVertexScores_kernel already
    // computed this as a local and threw it away; materializing it is purely additive -- the score
    // still consumes the same local the same way, so every existing planner is bit-identical.
    thrust::device_vector<float> d_regionCoverage_;

    float *d_vertexScoreArray_ptr_, *d_minValueInRegion_ptr_, *d_partialSums_ptr_, *d_totalScore_ptr_;
    float *d_regionCoverage_ptr_;
    int *d_validCounterArray_ptr_, *d_counterArray_ptr_, *d_activeVerticesScanIdx_ptr_, *d_activeSubVertices_ptr_;

    /****************************    METHODS    ****************************/
    void updateVertices();

private:
    /**************************** METHODS ****************************/
    void initializeRegions();
};

/**************************** DEVICE FUNCTIONS ****************************/
__host__ __device__ int getRegion(float* coord);
__device__ int getSubRegion(float* coord, int r1, float* minRegion);

// --- Computes the raw Syclop desirability score for every R1 region (0 for inactive regions).
//     One thread per region; the total is summed with thrust::reduce, which is correct for any
//     region count. (The old cub reduction launched > 1024 threads/block once NUM_R1_REGIONS
//     exceeded 32768, causing an illegal-memory-access crash at the larger deltas.) ---
__global__ void computeVertexScores_kernel(int* activeSubVertices, int* validCounterArray, int* counterArray, float* vertexScores,
                                           float* regionCoverage);
__global__ void updateSampleAcceptance_kernel(int* validCounterArray, float* vertexScores, float* totalScore,
                                              float scoreFloor);

/***************************/
/* INITIALIZE REGIONS KERNEL */
/***************************/
// --- Initializes min and max values for regions ---
__global__ void initializeRegions_kernel(float* minValueInRegion);