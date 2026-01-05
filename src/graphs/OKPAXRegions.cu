#include "graphs/OKPAXRegions.cuh"
#include "config/config.h"
#include <filesystem>

OKPAXRegions::OKPAXRegions(const float ws)
{
    if(VERBOSE)
        {
            printf("/***************************/\n");
            printf("/* OKPAX Regions Initialized */\n");
            printf("/* Number of R1 Regions: %d */\n", NUM_R1_REGIONS);
            printf("/***************************/\n");
        }

    d_minCostsR1_     = thrust::device_vector<float>(NUM_R1_REGIONS);
    d_minCostsR1_ptr_ = thrust::raw_pointer_cast(d_minCostsR1_.data());

    initializeRegions();
}

void OKPAXRegions::initializeRegions()
{
    thrust::fill(d_minCostsR1_.begin(), d_minCostsR1_.end(), MAX_FLOAT);
}
