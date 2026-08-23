#include "graphs/Graph.cuh"
#include "config/config.h"
#include <filesystem>

// Predicate for counting active regions (validCounterArray[r] > 0) when the dynamic score floor is
// enabled. A named functor rather than a thrust placeholder so it compiles the same way under every
// nvcc this repo targets.
struct IsActiveRegion
{
    __host__ __device__ bool operator()(int v) const { return v > 0; }
};

Graph::Graph(const float ws)
{
    if(VERBOSE)
        {
            printf("/***************************/\n");
            printf("/* Grid Dimension: %d */\n", W_DIM + C_DIM + V_DIM);
            printf("/***************************/\n");
        }

    h_numPartialSums_ = iDivUp(NUM_R1_REGIONS, h_blockSize_);

    d_validCounterArray_     = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_counterArray_          = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_vertexScoreArray_      = thrust::device_vector<float>(NUM_R1_REGIONS);
    d_activeVerticesScanIdx_ = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_activeSubVertices_     = thrust::device_vector<int>(NUM_R2_REGIONS);
    d_minValueInRegion_      = thrust::device_vector<float>(NUM_R1_REGIONS * STATE_DIM);
    d_partialSums_           = thrust::device_vector<float>(h_numPartialSums_);
    d_totalScore_            = thrust::device_vector<float>(1, 0.0);

    d_validCounterArray_ptr_ = thrust::raw_pointer_cast(d_validCounterArray_.data());
    d_counterArray_ptr_      = thrust::raw_pointer_cast(d_counterArray_.data());
    d_vertexScoreArray_ptr_  = thrust::raw_pointer_cast(d_vertexScoreArray_.data());
    d_activeSubVertices_ptr_ = thrust::raw_pointer_cast(d_activeSubVertices_.data());
    d_minValueInRegion_ptr_  = thrust::raw_pointer_cast(d_minValueInRegion_.data());
    d_partialSums_ptr_       = thrust::raw_pointer_cast(d_partialSums_.data());
    d_totalScore_ptr_        = thrust::raw_pointer_cast(d_totalScore_.data());

    initializeRegions();

    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/RegionMins");

    filename.str("");
    filename << "Data/RegionMins/RegionMins_" << ws << ".csv";
    copyAndWriteVectorToCSV(d_minValueInRegion_, filename.str(), NUM_R1_REGIONS, 1, false);
}

void Graph::initializeRegions()
{
    initializeRegions_kernel<<<iDivUp(NUM_R1_REGIONS, h_blockSize_), h_blockSize_>>>(d_minValueInRegion_ptr_);
}

/***************************/
/* INITIALIZE REGIONS KERNEL */
/***************************/
// --- one thread per R1 region ---
__global__ void initializeRegions_kernel(float* minValueInRegion)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= NUM_R1_REGIONS) return;

    int wRegion = tid % (W_R1_LENGTH * W_R1_LENGTH * W_R1_LENGTH);
    int wIndex[W_DIM];
    int temp = wRegion;
    for(int i = W_DIM - 1; i >= 0; --i)
        {
            wIndex[i] = temp % W_R1_LENGTH;
            temp /= W_R1_LENGTH;
        }

    for(int i = 0; i < W_DIM; ++i)
        {
            minValueInRegion[tid * STATE_DIM + i] = W_MIN + wIndex[i] * W_R1_SIZE;
        }

    int aRegion = (tid / (W_R1_LENGTH * W_R1_LENGTH * W_R1_LENGTH)) % (C_R1_LENGTH * C_R1_LENGTH);
    int aIndex[(C_DIM > 0) ? C_DIM : 1];  // guard against zero-length array when C_DIM == 0
    temp = aRegion;
    for(int i = C_DIM - 1; i >= 0; --i)
        {
            aIndex[i] = temp % C_R1_LENGTH;
            temp /= C_R1_LENGTH;
        }
    for(int i = 0; i < C_DIM; ++i)
        {
            minValueInRegion[tid * STATE_DIM + W_DIM + i] = C_MIN + aIndex[i] * C_R1_SIZE;
        }

    int vRegion = (tid / (W_R1_LENGTH * W_R1_LENGTH * W_R1_LENGTH * C_R1_LENGTH * C_R1_LENGTH)) % V_R1_LENGTH;
    int vIndex[V_DIM];
    temp = vRegion;
    for(int i = V_DIM - 1; i >= 0; --i)
        {
            vIndex[i] = temp % V_R1_LENGTH;
            temp /= V_R1_LENGTH;
        }
    for(int i = 0; i < V_DIM; ++i)
        {
            minValueInRegion[tid * STATE_DIM + W_DIM + C_DIM + i] = V_MIN + vIndex[i] * V_R1_SIZE;
        }
}

__host__ __device__ int getRegion(float* coord)
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

__device__ int getSubRegion(float* coord, int r1, float* minRegion)
{
    // --- Workspace ---
    int wRegion = 0;
    int factor  = 1;
    int index;

    for(int i = W_DIM - 1; i >= 0; --i)
        {
            index = (int)(W_R2_LENGTH * (coord[i] - minRegion[r1 * STATE_DIM + i]) / (W_R1_SIZE));
            if(index >= W_R2_LENGTH) index = W_R2_LENGTH - 1;
            if(index < 0) index = 0;

            wRegion += factor * index;
            factor *= W_R2_LENGTH;
        }

    // --- Attitude ---
    int aRegion = 0;
    if(C_R2_LENGTH > 1)
        {
            factor = 1;
            for(int i = C_DIM - 1; i >= 0; --i)
                {
                    index = (int)(C_R2_LENGTH * (coord[i + W_DIM] - minRegion[r1 * STATE_DIM + i + W_DIM]) / (C_R1_SIZE));
                    if(index >= C_R2_LENGTH) index = C_R2_LENGTH - 1;
                    if(index < 0) index = 0;

                    aRegion += factor * index;
                    factor *= C_R2_LENGTH;
                }
        }

    // --- Velocity ---
    int vRegion = 0;
    if(V_R2_LENGTH > 1)
        {
            factor = 1;
            for(int i = V_DIM - 1; i >= 0; --i)
                {
                    index = (int)(V_R2_LENGTH * (coord[i + W_DIM + C_DIM] - minRegion[r1 * STATE_DIM + i + W_DIM + C_DIM]) / (V_R1_SIZE));
                    if(index >= V_R2_LENGTH) index = V_R2_LENGTH - 1;
                    if(index < 0) index = 0;

                    vRegion += factor * index;
                    factor *= V_R2_LENGTH;
                }
        }

    return r1 * NUM_R2_PER_R1 + (wRegion * pow(C_R2_LENGTH, C_DIM) * pow(V_R2_LENGTH, V_DIM) + aRegion * pow(V_R2_LENGTH, V_DIM) + vRegion);
}

void Graph::updateVertices()
{
    int blocks = iDivUp(NUM_R1_REGIONS, h_blockSize_);

    // --- Compute raw per-region scores (one thread per region; 0 for inactive regions). ---
    computeVertexScores_kernel<<<blocks, h_blockSize_>>>(d_activeSubVertices_ptr_, d_validCounterArray_ptr_, d_counterArray_ptr_,
                                                         d_vertexScoreArray_ptr_);

    // --- Sum scores -> totalScore. thrust::reduce is correct for any region count, unlike the
    //     old single-block cub reduction which launched > 1024 threads/block for large grids. ---
    float total = thrust::reduce(d_vertexScoreArray_.begin(), d_vertexScoreArray_.end(), 0.0f);
    cudaMemcpy(d_totalScore_ptr_, &total, sizeof(float), cudaMemcpyHostToDevice);

    // --- Score floor: the mean share (1/N_active) when dynamic, else the legacy fixed EPSILON.
    //     See the comment on h_dynamicScoreFloor_ in Graph.cuh for why the legacy floor swamps the
    //     score it is meant to floor. Skipped entirely in legacy mode, so KPAX pays no extra pass. ---
    if(h_dynamicScoreFloor_)
        {
            int nActive = (int)thrust::count_if(d_validCounterArray_.begin(), d_validCounterArray_.end(), IsActiveRegion());
            h_scoreFloor_ = (nActive > 0) ? 1.0f / (float)nActive : 1.0f;
        }
    else
        {
            h_scoreFloor_ = EPSILON;
        }

    // --- Normalize: active regions -> scoreFloor + score/total; inactive regions -> 1.0. ---
    updateSampleAcceptance_kernel<<<blocks, h_blockSize_>>>(d_validCounterArray_ptr_, d_vertexScoreArray_ptr_, d_totalScore_ptr_,
                                                            h_scoreFloor_);
}

/***************************/
/* COMPUTE VERTEX SCORES KERNEL */
/***************************/
// --- Calculates the raw Syclop desirability score for each region (0 for inactive regions).
//     The total is summed separately with thrust::reduce in Graph::updateVertices. ---
__global__ void
computeVertexScores_kernel(int* activeSubVertices, int* validCounterArray, int* counterArray, float* vertexScores)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= NUM_R1_REGIONS) return;

    if(validCounterArray[tid] > 0)
        {
            int numValidSamples = validCounterArray[tid];
            float coverage      = 0;

            // --- Thread loops through all sub vertices to determine vertex coverage. ---
            for(int i = tid * NUM_R2_PER_R1; i < (tid + 1) * NUM_R2_PER_R1; ++i)
                {
                    coverage += activeSubVertices[i];
                }
            coverage /= NUM_R2_PER_R1;

            // --- From OMPL Syclop ref: https://ompl.kavrakilab.org/classompl_1_1control_1_1Syclop.html---
            float freeVol = (EPSILON + numValidSamples) / (EPSILON + numValidSamples + (counterArray[tid] - numValidSamples)) * W_R1_VOL;
            vertexScores[tid] = pow(freeVol, 4) / ((1 + coverage) * (1 + pow(counterArray[tid], 2)));
        }
    else
        {
            vertexScores[tid] = 0.0f;  // inactive region contributes nothing to totalScore
        }
}

/***************************/
/* UPDATE SAMPLE ACCEPTANCE KERNEL */
/***************************/
// --- normalizes score for each active region ---
// scoreFloor is EPSILON in legacy mode and 1/N_active in dynamic mode (Graph::updateVertices).
// The division only runs on the active branch, and a region can only be active if some region
// scored, so totalScore > 0 whenever it is read -- no 0/0.
__global__ void updateSampleAcceptance_kernel(int* validCounterArray, float* vertexScores, float* totalScore,
                                              float scoreFloor)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= NUM_R1_REGIONS) return;
    if(validCounterArray[tid] == 0)
        {
            // Never-visited region: unconditional free pass, the main exploration drive.
            vertexScores[tid] = 1.0f;
        }
    else
        {
            vertexScores[tid] = scoreFloor + (vertexScores[tid] / *totalScore);
        }
}