#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

// Include the necessary headers
#include "statePropagator/statePropagator.cuh"
#include "collisionCheck/collisionCheck.cuh"

// Profiling data structure (must match the one in statePropagatorProfiled.cu)
struct ProfilingData {
    unsigned long long randomGeneration;
    unsigned long long odeComputation;
    unsigned long long dynamicsCheck;
    unsigned long long workspaceCheck;
    unsigned long long bbConstruction;
    unsigned long long collisionCheck;
    unsigned long long totalIterations;
};

// Forward declarations for profiled functions
extern __device__ bool propagateAndCheckQuadRungeKuttaProfiled(float* x0, float* x1, curandState* seed, float* obstacles, int obstaclesCount);
extern __global__ void resetProfilingData();
extern __global__ void getProfilingData(ProfilingData* hostData);

// Test kernel that calls the profiled propagation function
__global__ void testPropagationKernel(float* d_states, float* d_results, curandState* d_randStates,
                                      float* d_obstacles, int obstaclesCount, int numTests)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numTests) return;

    // Initialize random state for this thread
    curandState localState = d_randStates[idx];

    // Get initial state
    float x0[STATE_DIM];
    for(int i = 0; i < STATE_DIM; i++) {
        x0[i] = d_states[idx * STATE_DIM + i];
    }

    // Run profiled propagation
    float x1[CONTROL_DIM];
    bool valid = propagateAndCheckQuadRungeKuttaProfiled(x0, x1, &localState, d_obstacles, obstaclesCount);

    // Store result
    d_results[idx] = valid ? 1.0f : 0.0f;

    // Update random state
    d_randStates[idx] = localState;
}

// Initialize random states
__global__ void initRandStates(curandState* states, int numStates, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numStates) return;
    curand_init(seed, idx, 0, &states[idx]);
}

int main(int argc, char** argv)
{
    // Configuration
    int numTests = 10000;  // Number of propagations to test
    int numObstacles = 100; // Default number of obstacles

    if(argc > 1) {
        numTests = atoi(argv[1]);
    }
    if(argc > 2) {
        numObstacles = atoi(argv[2]);
    }

    printf("=== Propagation Profiler ===\n");
    printf("Running %d propagation tests with %d obstacles\n\n", numTests, numObstacles);

    // Allocate device memory
    float* d_states;
    float* d_results;
    float* d_obstacles;
    curandState* d_randStates;

    cudaMalloc(&d_states, numTests * STATE_DIM * sizeof(float));
    cudaMalloc(&d_results, numTests * sizeof(float));
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMalloc(&d_randStates, numTests * sizeof(curandState));

    // Initialize states (simple starting configuration)
    float* h_states = new float[numTests * STATE_DIM];
    for(int i = 0; i < numTests; i++) {
        // Start position (middle of workspace)
        h_states[i * STATE_DIM + 0] = W_SIZE / 2.0f; // x
        h_states[i * STATE_DIM + 1] = W_SIZE / 2.0f; // y
        h_states[i * STATE_DIM + 2] = W_SIZE / 2.0f; // z
        // Rest zeros
        for(int j = 3; j < STATE_DIM; j++) {
            h_states[i * STATE_DIM + j] = 0.0f;
        }
    }
    cudaMemcpy(d_states, h_states, numTests * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_states;

    // Initialize obstacles (random boxes)
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

    // Reset profiling data
    resetProfilingData<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Run test kernel
    printf("Running propagation kernel...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    testPropagationKernel<<<blocks, threadsPerBlock>>>(d_states, d_results, d_randStates,
                                                        d_obstacles, numObstacles, numTests);
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

    // Get GPU clock rate for converting cycles to time
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    double clockRate = prop.clockRate * 1000.0; // Convert to Hz

    // Calculate percentages and times
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
