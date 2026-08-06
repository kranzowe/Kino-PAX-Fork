#pragma once

#include "config/config.h"
#include <cuda_runtime.h>

// Spatial hash grid configuration.
// GRID_CELL_SIZE is workspace-relative: (W_MAX - W_MIN)/16 yields a ~17^3 grid for any
// model's workspace scale. The old hardcoded 5.0f was tuned for a 100^3 workspace and
// collapsed Model 1's [0,1]^3 into a SINGLE cell, so every obstacle past
// MAX_OBSTACLES_PER_CELL was silently ignored and motions passed through them.
#define GRID_CELL_SIZE ((W_MAX - W_MIN) / 16.0f)
#define GRID_DIM_X ((int)((W_MAX - W_MIN) / GRID_CELL_SIZE) + 1)
#define GRID_DIM_Y ((int)((W_MAX - W_MIN) / GRID_CELL_SIZE) + 1)
#define GRID_DIM_Z ((int)((W_MAX - W_MIN) / GRID_CELL_SIZE) + 1)
#define GRID_SIZE (GRID_DIM_X * GRID_DIM_Y * GRID_DIM_Z)

// Maximum obstacles stored per cell. MUST be >= the most obstacles that can fall in a
// single cell, or buildSpatialHashGrid silently drops the overflow and collision checks
// miss them. 128 covers the current environments (house = 81 obstacles) even in the
// worst case where every obstacle overlaps one cell; raise it for denser environments.
#define MAX_OBSTACLES_PER_CELL 128

/**
 * Spatial hash grid structure
 *
 * The grid divides workspace into cells. Each cell stores indices of obstacles
 * that intersect it. During collision checking, we only check obstacles in
 * cells that the trajectory passes through.
 */
struct SpatialHashGrid
{
    // For each grid cell, stores the starting index in obstacleIndices array
    int* cellStarts;  // Size: GRID_SIZE

    // For each grid cell, stores the count of obstacles in that cell
    int* cellCounts;  // Size: GRID_SIZE

    // Flat array of obstacle indices, organized by cell
    // cellStarts[cellIdx] points to first obstacle index for that cell
    // cellCounts[cellIdx] tells how many obstacle indices follow
    int* obstacleIndices;  // Size: GRID_SIZE * MAX_OBSTACLES_PER_CELL
};

/**
 * Compute grid cell indices from world coordinates
 */
__device__ __host__ inline void worldToGrid(float x, float y, float z, int& gx, int& gy, int& gz)
{
    gx = (int)((x - W_MIN) / GRID_CELL_SIZE);
    gy = (int)((y - W_MIN) / GRID_CELL_SIZE);
    gz = (int)((z - W_MIN) / GRID_CELL_SIZE);

    // Clamp to grid bounds
    gx = max(0, min(gx, GRID_DIM_X - 1));
    gy = max(0, min(gy, GRID_DIM_Y - 1));
    gz = max(0, min(gz, GRID_DIM_Z - 1));
}

/**
 * Convert 3D grid coordinates to 1D cell index
 */
__device__ __host__ inline int gridToIndex(int gx, int gy, int gz)
{
    return gx + gy * GRID_DIM_X + gz * GRID_DIM_X * GRID_DIM_Y;
}

/**
 * Build spatial hash grid from obstacle array
 * Each obstacle is defined by 2*W_DIM floats: [minX, minY, minZ, maxX, maxY, maxZ]
 */
__global__ void buildSpatialHashGrid(
    float* obstacles,
    int obstaclesCount,
    SpatialHashGrid grid
);

/**
 * Initialize spatial hash grid (allocates device memory)
 */
SpatialHashGrid* createSpatialHashGrid();

/**
 * Free spatial hash grid memory
 */
void destroySpatialHashGrid(SpatialHashGrid* grid);

/**
 * Update spatial hash grid with new obstacle configuration
 */
void updateSpatialHashGrid(SpatialHashGrid* grid, float* d_obstacles, int obstaclesCount);

/**
 * Collision check using spatial hash grid
 * Only checks obstacles in cells intersected by the bounding box
 */
__device__ bool isMotionValidSpatialHash(
    float* x0,
    float* x1,
    float* bbMin,
    float* bbMax,
    SpatialHashGrid grid,
    float* obstacles
);
