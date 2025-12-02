# Spatial Hash Optimization

## Overview

This document describes the spatial hash grid optimization for collision detection in KinoPax.

## Problem

The original collision detection checked **every obstacle** for every propagation substep:
- 100 obstacles × 5.5 iterations avg = 550 obstacle checks per propagation
- With thousands of propagations per iteration, this dominated 40% of runtime

## Solution: Spatial Hashing

Divide the workspace into a 3D grid. Each grid cell maintains a list of obstacles that intersect it.

### Grid Configuration

```cpp
#define GRID_CELL_SIZE 5.0f           // 5x5x5 meter cells
#define GRID_DIM_X 21                 // (100/5) + 1
#define GRID_DIM_Y 21
#define GRID_DIM_Z 21
#define GRID_SIZE 9261                // 21^3 cells
#define MAX_OBSTACLES_PER_CELL 16
```

For a 100×100×100 workspace:
- **Grid**: 21×21×21 = 9,261 cells
- **Cell size**: 5×5×5 meters
- **Memory**: ~150KB per grid

## How It Works

### 1. Preprocessing (once per iteration)

```cuda
buildSpatialHashGrid(obstacles, grid)
```

- Each obstacle inserted into all grid cells it overlaps
- Parallel: one thread per obstacle
- Cost: ~0.1ms for 100 obstacles

### 2. Collision Checking (every substep)

**Old approach:**
```cuda
for each obstacle (0 to 100):
    check collision
```

**New approach:**
```cuda
find cells overlapping bounding box (typically 1-4 cells)
for each cell:
    for each obstacle in cell (typically 0-5):
        check collision
```

**Speedup**: 100 checks → ~5 checks = **20x reduction**

## Implementation Files

### Core Files
- `include/collisionCheck/spatialHash.cuh` - Data structures and API
- `src/collisionCheck/spatialHash.cu` - Grid construction and queries
- `include/statePropagator/statePropagatorSpatialHash.cuh` - Propagation API
- `src/statePropagator/statePropagatorSpatialHash.cu` - Spatial-hash-enabled propagation

### Testing
- `tools/profilePropagationSpatialHash.cu` - Profiler to measure speedup

## Building and Testing

### Build the Profiler

```bash
cd build
cmake ..
make profilePropagationSpatialHash
```

### Run Comparison

```bash
# Original (no spatial hash)
./profilePropagation 10000 100

# Optimized (with spatial hash)
./profilePropagationSpatialHash 10000 100
```

## Expected Results

**Before (original):**
- Collision Check: ~40% of total time
- Checking 100 obstacles per substep

**After (spatial hash):**
- Collision Check: ~5-10% of total time
- Checking 5-10 obstacles per substep on average
- **Overall speedup: 20-30%**

## Tuning Parameters

### Grid Cell Size

```cpp
#define GRID_CELL_SIZE 5.0f
```

**Trade-offs:**
- **Larger cells**: Fewer cells, more obstacles per cell
- **Smaller cells**: More cells, fewer obstacles per cell

**Optimal**: Slightly larger than typical obstacle size

For your workspace:
- Obstacle sizes: 0.5-2.5 meters
- Cell size: 5 meters (good balance)

### Max Obstacles Per Cell

```cpp
#define MAX_OBSTACLES_PER_CELL 16
```

If obstacles are very dense, increase this. Otherwise it's conservative.

## Integration into KinoPax

To use spatial hashing in KinoPax:

1. **Create grid** (before planning loop):
```cpp
SpatialHashGrid* d_grid = createSpatialHashGrid();
```

2. **Update grid** (once per iteration, after obstacle updates):
```cpp
updateSpatialHashGrid(d_grid, d_obstacles, obstaclesCount);
```

3. **Use spatial hash propagation** (in kernels):
```cpp
propagateAndCheckSpatialHash(x0, x1, seed, grid, obstacles, count);
```

4. **Cleanup** (after planning):
```cpp
destroySpatialHashGrid(d_grid);
```

## Memory Usage

**Per Grid:**
- `cellStarts`: 9,261 × 4 bytes = 37 KB
- `cellCounts`: 9,261 × 4 bytes = 37 KB
- `obstacleIndices`: 9,261 × 16 × 4 bytes = 592 KB
- **Total: ~666 KB**

For 400K tree nodes, this is negligible (<0.5% overhead).

## Limitations

1. **Static obstacles only**: Grid must be rebuilt if obstacles move
2. **Memory overhead**: Small but non-zero
3. **Sparse regions**: Cells with no obstacles still consume memory

## Future Optimizations

1. **Compact storage**: Use variable-length lists instead of fixed MAX_OBSTACLES_PER_CELL
2. **Hierarchical grids**: Multi-resolution for varying obstacle densities
3. **Persistent grid**: Reuse across iterations if obstacles don't change
4. **Warp-cooperative collision**: Parallelize within-cell checks across warp

## References

- Teschner et al. "Optimized Spatial Hashing for Collision Detection of Deformable Objects" (2003)
- Green "Particle Simulation using CUDA" (NVIDIA, 2010)
