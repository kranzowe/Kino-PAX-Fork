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

class OKPAXRegions
{
public:
    // --- constructor ---
    OKPAXRegions() = default;
    OKPAXRegions(float h_ws);

    // --- device fields ---
    thrust::device_vector<float> d_minCostsR1_;
    float* d_minCostsR1_ptr_;

private:
    /**************************** METHODS ****************************/
    void initializeRegions();
};

/**************************** DEVICE FUNCTIONS ****************************/
__host__ __device__ int OKPAX_getRegion(float* coord);

/***************************/
/* INITIALIZE REGIONS KERNEL */
/***************************/
// --- Initializes min costs for R1 regions ---
__global__ void OKPAXRegions_initializeRegions_kernel(float* minCostsR1);
