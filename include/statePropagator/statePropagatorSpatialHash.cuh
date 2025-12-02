#pragma once
#include <curand_kernel.h>
#include "helper/helper.cuh"
#include "config/config.h"
#include "collisionCheck/collisionCheck.cuh"
#include "collisionCheck/spatialHash.cuh"
#include "statePropagator/statePropagator.cuh"  // For ode() function

/**
 * Spatial-hash-enabled propagation functions
 * These versions use spatial hashing for faster collision detection
 */

__device__ bool propagateAndCheckSpatialHash(float* x0, float* x1, curandState* seed, SpatialHashGrid grid, float* obstacles, int obstaclesCount);
__device__ bool propagateAndCheckUnicycleSpatialHash(float* x0, float* x1, curandState* seed, SpatialHashGrid grid, float* obstacles, int obstaclesCount);
__device__ bool propagateAndCheckDoubleIntRungeKuttaSpatialHash(float* x0, float* x1, curandState* seed, SpatialHashGrid grid, float* obstacles, int obstaclesCount);
__device__ bool propagateAndCheckDubinsAirplaneRungeKuttaSpatialHash(float* x0, float* x1, curandState* seed, SpatialHashGrid grid, float* obstacles, int obstaclesCount);
__device__ bool propagateAndCheckQuadRungeKuttaSpatialHash(float* x0, float* x1, curandState* seed, SpatialHashGrid grid, float* obstacles, int obstaclesCount);

typedef bool (*PropagateAndCheckFuncSpatialHash)(float*, float*, curandState*, SpatialHashGrid, float*, int);

__device__ PropagateAndCheckFuncSpatialHash getPropagateAndCheckFuncSpatialHash();
