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

/***************************/
/* OKPAX GET REGION DEVICE FUNCTION */
/***************************/
// --- Returns the R1 region index for a given coordinate ---
__host__ __device__ int OKPAX_getRegion(float* coord)
{
    // --- Workspace ---
    int wRegion = 0;
    int factor  = 1;
    int index;
    for(int i = W_DIM - 1; i >= 0; --i)
        {
            index = (int)(W_R1_LENGTH * (coord[i] - W_MIN) / (W_MAX - W_MIN));
            if(index >= W_R1_LENGTH) index = W_R1_LENGTH - 1;
            if(index < 0) index = 0;

            wRegion += factor * index;
            factor *= W_R1_LENGTH;
        }

    if(V_DIM == 1 && C_DIM == 1)
        {
            return wRegion;
        }

    // --- Attitude ---
    int aRegion = 0;
    if(C_R1_LENGTH > 1)
        {
            factor = 1;
            for(int i = C_DIM - 1; i >= 0; --i)
                {
                    index = (int)(C_R1_LENGTH * (coord[i + W_DIM] - C_MIN) / (C_MAX - C_MIN));
                    if(index >= C_R1_LENGTH) index = C_R1_LENGTH - 1;
                    if(index < 0) index = 0;

                    aRegion += factor * index;
                    factor *= C_R1_LENGTH;
                }
        }

    // --- Velocity ---
    int vRegion = 0;
    if(V_R1_LENGTH > 1)
        {
            factor = 1;
            for(int i = V_DIM - 1; i >= 0; --i)
                {
                    index = (int)(V_R1_LENGTH * (coord[i + W_DIM + C_DIM] - V_MIN) / (V_MAX - V_MIN));
                    if(index >= V_R1_LENGTH) index = V_R1_LENGTH - 1;
                    if(index < 0) index = 0;

                    vRegion += factor * index;
                    factor *= V_R1_LENGTH;
                }
        }

    return wRegion * pow(C_R1_LENGTH, C_DIM) * pow(V_R1_LENGTH, V_DIM) + aRegion * pow(V_R1_LENGTH, V_DIM) + vRegion;
}
