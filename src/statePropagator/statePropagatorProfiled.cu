#include "statePropagator/statePropagator.cuh"
#include <cuda_runtime.h>

// Device-side timing accumulator structure
struct ProfilingData {
    unsigned long long randomGeneration;
    unsigned long long odeComputation;
    unsigned long long dynamicsCheck;
    unsigned long long workspaceCheck;
    unsigned long long bbConstruction;
    unsigned long long collisionCheck;
    unsigned long long totalIterations;
};

// Global device variable for profiling
__device__ ProfilingData d_profilingData;

__device__ bool propagateAndCheckQuadRungeKuttaProfiled(float* x0, float* x1, curandState* seed, float* obstacles, int obstaclesCount)
{
    unsigned long long start, end;

    // Random number generation
    start = clock64();
    float Zc = QUAD_MIN_Zc + curand_uniform(seed) * (QUAD_MAX_Zc - QUAD_MIN_Zc);
    float Lc = QUAD_MIN_Lc + curand_uniform(seed) * (QUAD_MAX_Lc - QUAD_MIN_Lc);
    float Mc = QUAD_MIN_Mc + curand_uniform(seed) * (QUAD_MAX_Mc - QUAD_MIN_Mc);
    float Nc = QUAD_MIN_Nc + curand_uniform(seed) * (QUAD_MAX_Nc - QUAD_MIN_Nc);
    int propagationDuration = 1 + (int)(curand_uniform(seed) * (MAX_PROPAGATION_DURATION));
    end = clock64();
    atomicAdd(&d_profilingData.randomGeneration, end - start);

    bool motionValid = true;
    float bbMin[W_DIM], bbMax[W_DIM];

    float h0[STATE_DIM];
    float h1[STATE_DIM];
    float h2[STATE_DIM];
    float h3[STATE_DIM];
    float h4[STATE_DIM];

    for(int j = 0; j < STATE_DIM; j++) h0[j] = x0[j];

    for(int i = 0; i < propagationDuration; i++)
        {
            atomicAdd(&d_profilingData.totalIterations, 1ULL);

            float x0State[W_DIM] = {h0[0], h0[1], h0[2]};

            // ODE computation (Runge-Kutta)
            start = clock64();
            ode(h1, h0, nullptr, Zc, Lc, Mc, Nc, 0);
            ode(h2, h0, h1, Zc, Lc, Mc, Nc, 1);
            ode(h3, h0, h2, Zc, Lc, Mc, Nc, 2);
            ode(h4, h0, h3, Zc, Lc, Mc, Nc, 3);
            for(int j = 0; j < STATE_DIM; j++)
                {
                    h0[j] += STEP_SIZE / 6.0f * (h1[j] + 2.0f * h2[j] + 2.0f * h3[j] + h4[j]);
                }
            end = clock64();
            atomicAdd(&d_profilingData.odeComputation, end - start);

            float x1State[W_DIM] = {h0[0], h0[1], h0[2]};

            // Vehicle Dynamics Check
            start = clock64();
            if(h0[6] < V_MIN || h0[6] > V_MAX || h0[7] < V_MIN || h0[7] > V_MAX || h0[8] < V_MIN || h0[8] > V_MAX)
                {
                    motionValid = false;
                }
            end = clock64();
            atomicAdd(&d_profilingData.dynamicsCheck, end - start);
            if(!motionValid) break;

            // Workspace Limit Check
            start = clock64();
            if(h0[0] < W_MIN || h0[0] > W_MAX || h0[1] < W_MIN || h0[1] > W_MAX || h0[2] < W_MIN || h0[2] > W_MAX)
                {
                    motionValid = false;
                }
            end = clock64();
            atomicAdd(&d_profilingData.workspaceCheck, end - start);
            if(!motionValid) break;

            // Bounding Box Construction
            start = clock64();
            for(int d = 0; d < W_DIM; d++)
                {
                    if(x0State[d] > x1State[d])
                        {
                            bbMin[d] = x1State[d];
                            bbMax[d] = x0State[d];
                        }
                    else
                        {
                            bbMin[d] = x0State[d];
                            bbMax[d] = x1State[d];
                        }
                }
            end = clock64();
            atomicAdd(&d_profilingData.bbConstruction, end - start);

            // Collision Check
            start = clock64();
            motionValid = motionValid && isMotionValid(x0State, x1State, bbMin, bbMax, obstacles, obstaclesCount);
            end = clock64();
            atomicAdd(&d_profilingData.collisionCheck, end - start);

            if(!motionValid) break;
        }

    for(int j = 0; j < STATE_DIM; j++) x1[j] = h0[j];

    x1[12] = Zc;
    x1[13] = Lc;
    x1[14] = Mc;
    x1[15] = Nc;
    x1[16] = STEP_SIZE * propagationDuration;

    return motionValid;
}

// Kernel to reset profiling data
__global__ void resetProfilingData()
{
    d_profilingData.randomGeneration = 0;
    d_profilingData.odeComputation = 0;
    d_profilingData.dynamicsCheck = 0;
    d_profilingData.workspaceCheck = 0;
    d_profilingData.bbConstruction = 0;
    d_profilingData.collisionCheck = 0;
    d_profilingData.totalIterations = 0;
}

// Kernel to copy profiling data to host
__global__ void getProfilingData(ProfilingData* hostData)
{
    *hostData = d_profilingData;
}
