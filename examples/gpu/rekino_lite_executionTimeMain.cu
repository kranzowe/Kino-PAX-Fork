#include <iostream>
#include "ReKino/ReKinoLite.cuh"

int main(void)
{
    system("rm -rf Data/*");

    float h_initial[SAMPLE_DIM] = {10.0, 8, 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          h_goal[SAMPLE_DIM]    = {80, 95.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    ReKinoLite rekinolite;

    int numObstacles;
    float* d_obstacles;

    std::vector<float> obstacles = readObstaclesFromCSV("../include/config/obstacles/quadTrees/obstacles.csv", numObstacles, W_DIM);

    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(), numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);

    printf("=== ReKinoLite Benchmark ===\n");
    printf("Running 50 planning iterations...\n");
    printf("Initial: (%.1f, %.1f, %.1f)\n", h_initial[0], h_initial[1], h_initial[2]);
    printf("Goal: (%.1f, %.1f, %.1f)\n", h_goal[0], h_goal[1], h_goal[2]);
    printf("Obstacles: %d\n", numObstacles);
    printf("Warps: 512 (16,384 threads)\n\n");

    int N = 50;
    int successCount = 0;
    double totalTime = 0.0;

    for(int i = 0; i < N; i++)
    {
        printf("Iteration %d/%d...\n", i + 1, N);

        rekinolite.plan(h_initial, h_goal, d_obstacles, numObstacles, false);  // false = don't save tree (benchmark mode)

    }

    printf("\n=== Benchmark Complete ===\n");
    printf("Total iterations: %d\n", N);
    printf("Results written to ../data/execution_time.csv\n");

    // --- Free memory ---
    cudaFree(d_obstacles);

    return 0;
}
