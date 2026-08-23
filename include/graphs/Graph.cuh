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

    // --- device fields ---
    thrust::device_vector<int> d_validCounterArray_, d_counterArray_, d_activeVerticesScanIdx_, d_activeSubVertices_;
    thrust::device_vector<float> d_vertexScoreArray_, d_minValueInRegion_, d_partialSums_, d_totalScore_;

    float *d_vertexScoreArray_ptr_, *d_minValueInRegion_ptr_, *d_partialSums_ptr_, *d_totalScore_ptr_;
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
__global__ void computeVertexScores_kernel(int* activeSubVertices, int* validCounterArray, int* counterArray, float* vertexScores);
__global__ void updateSampleAcceptance_kernel(int* validCounterArray, float* vertexScores, float* totalScore,
                                              float scoreFloor);

/***************************/
/* INITIALIZE REGIONS KERNEL */
/***************************/
// --- Initializes min and max values for regions ---
__global__ void initializeRegions_kernel(float* minValueInRegion);