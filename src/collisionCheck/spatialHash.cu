#include "collisionCheck/spatialHash.cuh"
#include "collisionCheck/collisionCheck.cuh"
#include <stdio.h>

/**
 * Build spatial hash grid kernel
 * Each thread processes one obstacle and inserts it into all grid cells it overlaps
 */
__global__ void buildSpatialHashGrid(
    float* obstacles,
    int obstaclesCount,
    SpatialHashGrid grid
)
{
    int obsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(obsIdx >= obstaclesCount) return;

    // Get obstacle bounds
    float obsMin[W_DIM], obsMax[W_DIM];
    for(int d = 0; d < W_DIM; d++)
    {
        obsMin[d] = obstacles[obsIdx * 2 * W_DIM + d];
        obsMax[d] = obstacles[obsIdx * 2 * W_DIM + W_DIM + d];
    }

    // Find grid cells the obstacle overlaps
    int gMinX, gMinY, gMinZ, gMaxX, gMaxY, gMaxZ;
    worldToGrid(obsMin[0], obsMin[1], obsMin[2], gMinX, gMinY, gMinZ);
    worldToGrid(obsMax[0], obsMax[1], obsMax[2], gMaxX, gMaxY, gMaxZ);

    // Insert obstacle into all overlapping cells
    for(int gz = gMinZ; gz <= gMaxZ; gz++)
    {
        for(int gy = gMinY; gy <= gMaxY; gy++)
        {
            for(int gx = gMinX; gx <= gMaxX; gx++)
            {
                int cellIdx = gridToIndex(gx, gy, gz);

                // Atomically increment cell count and get insertion position
                int insertPos = atomicAdd(&grid.cellCounts[cellIdx], 1);

                // Only insert if we haven't exceeded max obstacles per cell
                if(insertPos < MAX_OBSTACLES_PER_CELL)
                {
                    int arrayIdx = cellIdx * MAX_OBSTACLES_PER_CELL + insertPos;
                    grid.obstacleIndices[arrayIdx] = obsIdx;
                }
                // If we exceed, the obstacle won't be in this cell's list
                // This is acceptable for performance - we'll still check some obstacles
            }
        }
    }
}

/**
 * Kernel to initialize grid arrays to zero
 */
__global__ void initSpatialHashGrid(SpatialHashGrid grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < GRID_SIZE)
    {
        grid.cellStarts[idx] = idx * MAX_OBSTACLES_PER_CELL;
        grid.cellCounts[idx] = 0;
    }
}

/**
 * Create and allocate spatial hash grid on device
 */
SpatialHashGrid* createSpatialHashGrid()
{
    SpatialHashGrid* h_grid = new SpatialHashGrid();
    SpatialHashGrid* d_grid;

    // Allocate device memory for grid structure
    cudaMalloc(&d_grid, sizeof(SpatialHashGrid));

    // Allocate device memory for grid arrays
    cudaMalloc(&h_grid->cellStarts, GRID_SIZE * sizeof(int));
    cudaMalloc(&h_grid->cellCounts, GRID_SIZE * sizeof(int));
    cudaMalloc(&h_grid->obstacleIndices, GRID_SIZE * MAX_OBSTACLES_PER_CELL * sizeof(int));

    // Copy structure to device
    cudaMemcpy(d_grid, h_grid, sizeof(SpatialHashGrid), cudaMemcpyHostToDevice);

    // Initialize grid
    int threadsPerBlock = 256;
    int blocks = (GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    initSpatialHashGrid<<<blocks, threadsPerBlock>>>(*h_grid);
    cudaDeviceSynchronize();

    // Clean up host copy of structure
    delete h_grid;

    return d_grid;
}

/**
 * Free spatial hash grid memory
 */
void destroySpatialHashGrid(SpatialHashGrid* d_grid)
{
    if(!d_grid) return;

    // Copy structure back to host to get pointers
    SpatialHashGrid h_grid;
    cudaMemcpy(&h_grid, d_grid, sizeof(SpatialHashGrid), cudaMemcpyDeviceToHost);

    // Free device arrays
    cudaFree(h_grid.cellStarts);
    cudaFree(h_grid.cellCounts);
    cudaFree(h_grid.obstacleIndices);

    // Free device structure
    cudaFree(d_grid);
}

/**
 * Update spatial hash grid with new obstacles
 */
void updateSpatialHashGrid(SpatialHashGrid* d_grid, float* d_obstacles, int obstaclesCount)
{
    // Copy grid structure to host to get pointers
    SpatialHashGrid h_grid;
    cudaMemcpy(&h_grid, d_grid, sizeof(SpatialHashGrid), cudaMemcpyDeviceToHost);

    // Reset grid
    int threadsPerBlock = 256;
    int blocks = (GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    initSpatialHashGrid<<<blocks, threadsPerBlock>>>(h_grid);
    cudaDeviceSynchronize();

    // Build grid from obstacles
    blocks = (obstaclesCount + threadsPerBlock - 1) / threadsPerBlock;
    buildSpatialHashGrid<<<blocks, threadsPerBlock>>>(d_obstacles, obstaclesCount, h_grid);
    cudaDeviceSynchronize();
}

/**
 * Collision check using spatial hash
 * Only checks obstacles in cells intersected by the bounding box
 */
__device__ bool isMotionValidSpatialHash(
    float* x0,
    float* x1,
    float* bbMin,
    float* bbMax,
    SpatialHashGrid grid,
    float* obstacles
)
{
    // Find grid cells the bounding box overlaps
    int gMinX, gMinY, gMinZ, gMaxX, gMaxY, gMaxZ;
    worldToGrid(bbMin[0], bbMin[1], bbMin[2], gMinX, gMinY, gMinZ);
    worldToGrid(bbMax[0], bbMax[1], bbMax[2], gMaxX, gMaxY, gMaxZ);

    // Check obstacles in all overlapping cells
    for(int gz = gMinZ; gz <= gMaxZ; gz++)
    {
        for(int gy = gMinY; gy <= gMaxY; gy++)
        {
            for(int gx = gMinX; gx <= gMaxX; gx++)
            {
                int cellIdx = gridToIndex(gx, gy, gz);
                int count = grid.cellCounts[cellIdx];
                int start = grid.cellStarts[cellIdx];

                // Check each obstacle in this cell
                for(int i = 0; i < count; i++)
                {
                    int obsIdx = grid.obstacleIndices[start + i];

                    // Load obstacle bounds
                    float obs[2 * W_DIM];
                    for(int d = 0; d < W_DIM; d++)
                    {
                        obs[d] = obstacles[obsIdx * 2 * W_DIM + d];
                        obs[W_DIM + d] = obstacles[obsIdx * 2 * W_DIM + W_DIM + d];
                    }

                    // Perform collision check (broad phase)
                    if(!isBroadPhaseValid(bbMin, bbMax, obs))
                    {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}
