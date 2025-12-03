#include "ReKino/ReKinoLite.cuh"
#include "config/config.h"
#include <iostream>

int main()
{
    // Simple test: start at origin, goal at opposite corner
    float h_initial[SAMPLE_DIM] = {0};
    float h_goal[SAMPLE_DIM] = {0};

    // Set initial position
    h_initial[0] = 10.0f;  // x
    h_initial[1] = 10.0f;  // y
    h_initial[2] = 10.0f;  // z

    // Set goal position
    h_goal[0] = 90.0f;
    h_goal[1] = 90.0f;
    h_goal[2] = 90.0f;

    // Create some random obstacles
    int numObstacles = 50;
    float* h_obstacles = new float[numObstacles * 2 * W_DIM];

    srand(42);
    for(int i = 0; i < numObstacles; i++)
    {
        for(int d = 0; d < W_DIM; d++)
        {
            float minVal = 20.0f + (float)rand() / RAND_MAX * 60.0f;
            float size = 2.0f + (float)rand() / RAND_MAX * 5.0f;
            h_obstacles[i * 2 * W_DIM + d] = minVal;
            h_obstacles[i * 2 * W_DIM + W_DIM + d] = minVal + size;
        }
    }

    // Copy obstacles to device
    float* d_obstacles;
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, h_obstacles, numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Create planner and run
    printf("Starting ReKinoLite...\n");
    printf("Initial: (%.1f, %.1f, %.1f)\n", h_initial[0], h_initial[1], h_initial[2]);
    printf("Goal: (%.1f, %.1f, %.1f)\n", h_goal[0], h_goal[1], h_goal[2]);
    printf("Obstacles: %d\n", numObstacles);
    printf("Warps: 512 (16,384 threads)\n\n");

    ReKinoLite planner;
    planner.plan(h_initial, h_goal, d_obstacles, numObstacles, true);  // true = save tree to CSV

    // Cleanup
    delete[] h_obstacles;
    cudaFree(d_obstacles);

    return 0;
}
