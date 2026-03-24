#include <iostream>
#include <string>
#include <chrono>
#include "planners/KPAX.cuh"
#include "planners/PruneKPAX.cuh"
#include "planners/KinoPaxPlus.cuh"
#include "planners/KPAXPlus.cuh"

// Print free/total GPU memory in MB
void printGPUMemory(const char* label)
{
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("  [Memory %s] Free: %.1f MB / Total: %.1f MB\n",
           label, freeMem / (1024.0 * 1024.0), totalMem / (1024.0 * 1024.0));
}

bool checkCudaError(const char* plannerName)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        printf("  FAIL - %s: CUDA error: %s\n", plannerName, cudaGetErrorString(err));
        return false;
    }
    return true;
}

// Run a planner for a few iterations and report pass/fail
template<typename PlannerType>
bool runSmokeTest(
    const char* name,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations)
{
    printf("\n--- %s ---\n", name);
    printGPUMemory("before");

    auto wallStart = std::chrono::steady_clock::now();

    {
        PlannerType planner;
        if(!checkCudaError(name)) return false;

        planner.resetPlanner(h_initial, h_goal);
        if(!checkCudaError(name)) return false;

        int itr = 0;
        while(itr < maxIterations)
        {
            itr++;
            planner.h_itr_++;
            planner.propagateFrontier(d_obstacles, numObstacles);
            planner.graph_.updateVertices();
            planner.updateFrontier();

            if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
        cudaDeviceSynchronize();

        if(!checkCudaError(name)) return false;

        auto wallEnd = std::chrono::steady_clock::now();
        double wallSec = std::chrono::duration<double>(wallEnd - wallStart).count();

        printf("  Ran %d iterations in %.3fs, tree_size=%d\n",
               itr, wallSec, planner.h_treeSize_);
    } // planner destroyed here

    cudaDeviceSynchronize();
    printGPUMemory("after");

    if(!checkCudaError(name)) return false;

    printf("  PASS - %s\n", name);
    return true;
}

// Specialization for KinoPaxPlus which doesn't have graph_
template<>
bool runSmokeTest<KinoPaxPlus>(
    const char* name,
    float* h_initial,
    float* h_goal,
    float* d_obstacles,
    uint numObstacles,
    int maxIterations)
{
    printf("\n--- %s ---\n", name);
    printGPUMemory("before");

    auto wallStart = std::chrono::steady_clock::now();

    {
        KinoPaxPlus planner;
        if(!checkCudaError(name)) return false;

        planner.resetPlanner(h_initial, h_goal);
        if(!checkCudaError(name)) return false;

        int itr = 0;
        while(itr < maxIterations)
        {
            itr++;
            planner.h_itr_++;
            planner.propagateFrontier(d_obstacles, numObstacles);
            if(planner.h_propIterations_ == 0) break;
            planner.updateFrontier();

            if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
        cudaDeviceSynchronize();

        if(!checkCudaError(name)) return false;

        auto wallEnd = std::chrono::steady_clock::now();
        double wallSec = std::chrono::duration<double>(wallEnd - wallStart).count();

        printf("  Ran %d iterations in %.3fs, tree_size=%d\n",
               itr, wallSec, planner.h_treeSize_);
    }

    cudaDeviceSynchronize();
    printGPUMemory("after");

    if(!checkCudaError(name)) return false;

    printf("  PASS - %s\n", name);
    return true;
}

int main(void)
{
    const int MAX_ITERATIONS = 50;

    printf("=======================================================\n");
    printf("    JETSON SMOKE TEST\n");
    printf("=======================================================\n");
    printf("Running each planner for %d iterations on Empty environment\n", MAX_ITERATIONS);
    printf("Purpose: verify all algorithms run without OOM on Jetson\n");
    printf("=======================================================\n");

    printGPUMemory("initial");

    // Load empty environment (minimal obstacles)
    int numObstacles;
    float* d_obstacles;
    std::vector<float> obstacles = readObstaclesFromCSV(
        "../include/config/obstacles/empty/obstacles.csv", numObstacles, W_DIM);
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(),
               numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);
    printf("Loaded %d obstacles\n", numObstacles);

    float h_initial[SAMPLE_DIM] = {10.0, 8, 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float h_goal[SAMPLE_DIM]    = {80, 95.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    int passed = 0;
    int total  = 5;

    passed += runSmokeTest<KPAX>("KPAX",
        h_initial, h_goal, d_obstacles, numObstacles, MAX_ITERATIONS);

    passed += runSmokeTest<PruneKPAX>("KPAX_SpatialHash",
        h_initial, h_goal, d_obstacles, numObstacles, MAX_ITERATIONS);

    passed += runSmokeTest<PruneKPAX>("PruneKPAX",
        h_initial, h_goal, d_obstacles, numObstacles, MAX_ITERATIONS);

    passed += runSmokeTest<KinoPaxPlus>("KinoPaxPlus",
        h_initial, h_goal, d_obstacles, numObstacles, MAX_ITERATIONS);

    passed += runSmokeTest<KPAXPlus>("KPAXPlus",
        h_initial, h_goal, d_obstacles, numObstacles, MAX_ITERATIONS);

    cudaFree(d_obstacles);

    printf("\n=======================================================\n");
    printf("    RESULTS: %d/%d PASSED\n", passed, total);
    printf("=======================================================\n");

    return (passed == total) ? 0 : 1;
}
