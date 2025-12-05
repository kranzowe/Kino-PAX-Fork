#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include "statePropagator/statePropagator.cuh"  // For ode() function
#include "statePropagator/statePropagatorSpatialHash.cuh"
#include "collisionCheck/collisionCheck.cuh"
#include "collisionCheck/spatialHash.cuh"

// Profiling data structure
struct ProfilingData {
    unsigned long long randomGeneration;
    unsigned long long odeComputation;
    unsigned long long dynamicsCheck;
    unsigned long long workspaceCheck;
    unsigned long long bbConstruction;
    unsigned long long collisionCheck;
    unsigned long long totalIterations;
};

__device__ ProfilingData d_profilingData;

__device__ bool propagateAndCheckQuadRungeKuttaProfiledSpatialHash(float* x0, float* x1, curandState* seed, SpatialHashGrid grid, float* obstacles, int obstaclesCount)
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

        // ODE computation
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

        // Dynamics Check
        start = clock64();
        if(h0[6] < V_MIN || h0[6] > V_MAX || h0[7] < V_MIN || h0[7] > V_MAX || h0[8] < V_MIN || h0[8] > V_MAX)
        {
            motionValid = false;
        }
        end = clock64();
        atomicAdd(&d_profilingData.dynamicsCheck, end - start);
        if(!motionValid) break;

        // Workspace Check
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

        // Collision Check (with spatial hash)
        start = clock64();
        motionValid = motionValid && isMotionValidSpatialHash(x0State, x1State, bbMin, bbMax, grid, obstacles);
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

__global__ void getProfilingData(ProfilingData* hostData)
{
    *hostData = d_profilingData;
}

// Test kernel
__global__ void testPropagationKernel(float* d_states, float* d_results, curandState* d_randStates,
                                      SpatialHashGrid grid, float* d_obstacles, int obstaclesCount, int numTests)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numTests) return;

    curandState localState = d_randStates[idx];

    float x0[STATE_DIM];
    for(int i = 0; i < STATE_DIM; i++) {
        x0[i] = d_states[idx * STATE_DIM + i];
    }

    float x1[CONTROL_DIM];
    bool valid = propagateAndCheckQuadRungeKuttaProfiledSpatialHash(x0, x1, &localState, grid, d_obstacles, obstaclesCount);

    d_results[idx] = valid ? 1.0f : 0.0f;
    d_randStates[idx] = localState;
}

__global__ void initRandStates(curandState* states, int numStates, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numStates) return;
    curand_init(seed, idx, 0, &states[idx]);
}

int main(int argc, char** argv)
{
    int numTests = 10000;
    int numObstacles = 100;

    if(argc > 1) numTests = atoi(argv[1]);
    if(argc > 2) numObstacles = atoi(argv[2]);

    printf("=== Propagation Profiler (WITH SPATIAL HASH) ===\n");
    printf("Running %d propagation tests with %d obstacles\n", numTests, numObstacles);
    printf("Grid size: %dx%dx%d = %d cells\n", GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z, GRID_SIZE);
    printf("Cell size: %.2f\n\n", GRID_CELL_SIZE);

    // Allocate device memory
    float* d_states;
    float* d_results;
    float* d_obstacles;
    curandState* d_randStates;

    cudaMalloc(&d_states, numTests * STATE_DIM * sizeof(float));
    cudaMalloc(&d_results, numTests * sizeof(float));
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMalloc(&d_randStates, numTests * sizeof(curandState));

    // Initialize states
    float* h_states = new float[numTests * STATE_DIM];
    for(int i = 0; i < numTests; i++) {
        h_states[i * STATE_DIM + 0] = W_SIZE / 2.0f;
        h_states[i * STATE_DIM + 1] = W_SIZE / 2.0f;
        h_states[i * STATE_DIM + 2] = W_SIZE / 2.0f;
        for(int j = 3; j < STATE_DIM; j++) {
            h_states[i * STATE_DIM + j] = 0.0f;
        }
    }
    cudaMemcpy(d_states, h_states, numTests * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_states;

    // Initialize obstacles
    float* h_obstacles = new float[numObstacles * 2 * W_DIM];
    srand(42);
    for(int i = 0; i < numObstacles; i++) {
        for(int d = 0; d < W_DIM; d++) {
            float minVal = (float)rand() / RAND_MAX * (W_SIZE - 1.0f);
            float size = 0.5f + (float)rand() / RAND_MAX * 2.0f;
            h_obstacles[i * 2 * W_DIM + d] = minVal;
            h_obstacles[i * 2 * W_DIM + W_DIM + d] = minVal + size;
        }
    }
    cudaMemcpy(d_obstacles, h_obstacles, numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_obstacles;

    // Initialize random states
    int threadsPerBlock = 256;
    int blocks = (numTests + threadsPerBlock - 1) / threadsPerBlock;
    initRandStates<<<blocks, threadsPerBlock>>>(d_randStates, numTests, 12345ULL);
    cudaDeviceSynchronize();

    // Create and build spatial hash grid
    printf("Building spatial hash grid...\n");
    cudaEvent_t gridStart, gridStop;
    cudaEventCreate(&gridStart);
    cudaEventCreate(&gridStop);

    cudaEventRecord(gridStart);
    SpatialHashGrid* d_grid = createSpatialHashGrid();
    updateSpatialHashGrid(d_grid, d_obstacles, numObstacles);
    cudaEventRecord(gridStop);
    cudaEventSynchronize(gridStop);

    float gridBuildTime = 0;
    cudaEventElapsedTime(&gridBuildTime, gridStart, gridStop);
    printf("Spatial hash grid built in %.3f ms!\n\n", gridBuildTime);

    cudaEventDestroy(gridStart);
    cudaEventDestroy(gridStop);

    // Reset profiling data
    resetProfilingData<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Run test kernel
    printf("Running propagation kernel...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy grid structure to pass to kernel
    SpatialHashGrid h_grid;
    cudaMemcpy(&h_grid, d_grid, sizeof(SpatialHashGrid), cudaMemcpyDeviceToHost);

    cudaEventRecord(start);
    testPropagationKernel<<<blocks, threadsPerBlock>>>(d_states, d_results, d_randStates,
                                                        h_grid, d_obstacles, numObstacles, numTests);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalTime = 0;
    cudaEventElapsedTime(&totalTime, start, stop);

    // Get profiling data
    ProfilingData* d_profData;
    ProfilingData h_profData;
    cudaMalloc(&d_profData, sizeof(ProfilingData));
    getProfilingData<<<1, 1>>>(d_profData);
    cudaMemcpy(&h_profData, d_profData, sizeof(ProfilingData), cudaMemcpyDeviceToHost);

    // Get GPU clock rate
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    double clockRate = prop.clockRate * 1000.0;

    unsigned long long totalCycles = h_profData.randomGeneration + h_profData.odeComputation +
                                     h_profData.dynamicsCheck + h_profData.workspaceCheck +
                                     h_profData.bbConstruction + h_profData.collisionCheck;

    printf("\n=== Results ===\n");
    printf("Total kernel time: %.3f ms\n", totalTime);
    printf("Average time per propagation: %.3f us\n", (totalTime * 1000.0f) / numTests);
    printf("Total iterations executed: %llu\n", h_profData.totalIterations);
    printf("Average iterations per propagation: %.2f\n", (double)h_profData.totalIterations / numTests);

    printf("\n=== Breakdown (by cycles) ===\n");
    printf("Component               Cycles              Percent     Time (ms)\n");
    printf("----------------------------------------------------------------\n");

    auto printStat = [&](const char* name, unsigned long long cycles) {
        double percent = 100.0 * cycles / totalCycles;
        double timeMs = (cycles / clockRate) * 1000.0;
        printf("%-20s %15llu    %6.2f%%    %8.3f\n", name, cycles, percent, timeMs);
    };

    printStat("Random Generation", h_profData.randomGeneration);
    printStat("ODE Computation", h_profData.odeComputation);
    printStat("Dynamics Check", h_profData.dynamicsCheck);
    printStat("Workspace Check", h_profData.workspaceCheck);
    printStat("BB Construction", h_profData.bbConstruction);
    printStat("Collision Check", h_profData.collisionCheck);
    printf("----------------------------------------------------------------\n");
    printStat("TOTAL", totalCycles);

    // Cleanup
    destroySpatialHashGrid(d_grid);
    cudaFree(d_states);
    cudaFree(d_results);
    cudaFree(d_obstacles);
    cudaFree(d_randStates);
    cudaFree(d_profData);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nProfiler completed successfully!\n");
    return 0;
}
