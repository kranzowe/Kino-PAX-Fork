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
__global__ void updateSampleAcceptance_kernel(int* validCounterArray, float* vertexScores, float* totalScore);

/***************************/
/* INITIALIZE REGIONS KERNEL */
/***************************/
// --- Initializes min and max values for regions ---
__global__ void initializeRegions_kernel(float* minValueInRegion);