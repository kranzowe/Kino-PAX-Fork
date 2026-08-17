#pragma once

#include <stdio.h>
#include <sstream>
#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>
#include <math.h>
#include <fstream>
#include "cuda.h"
#include "cuda_runtime.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iomanip>
#include "config/config.h"
#define _USE_MATH_DEFINES

// A macro for checking the error codes of cuda runtime calls
#define CUDA_ERROR_CHECK(expr)                                                \
    {                                                                         \
        cudaError_t err = expr;                                               \
        if(err != cudaSuccess)                                                \
            {                                                                 \
                printf("CUDA call failed!\n\t%s\n", cudaGetErrorString(err)); \
                exit(1);                                                      \
            }                                                                 \
    }

template <typename T>
void printDeviceVector(const T* d_ptr, int size);

__device__ void printSample(float* x, int sampleDim);
std::vector<float> readObstaclesFromCSV(const std::string& filename, int& numObstacles, int workspaceDim);

template <typename T>
void writeVectorToCSV(const thrust::host_vector<T>& vec, const std::string& filename, int rows, int cols);

template <typename T>
void copyAndWriteVectorToCSV(const thrust::device_vector<T>& d_vec, const std::string& filename, int rows, int cols);

// Implement the template functions in the header file

template <typename T>
void printDeviceVector(const T* d_ptr, int size)
{
    thrust::host_vector<T> h_vec(size);
    cudaMemcpy(thrust::raw_pointer_cast(h_vec.data()), d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    for(int i = 0; i < size; ++i)
        {
            std::cout << h_vec[i] << " ";
        }
    std::cout << std::endl;
}

template <typename T>
void writeVectorToCSV(const thrust::host_vector<T>& vec, const std::string& filename, int rows, int cols, bool append = false)
{
    std::ofstream file;
    if(append)
        {
            file.open(filename, std::ios_base::app);  // Open in append mode
        }
    else
        {
            file.open(filename);
        }
    file << std::fixed << std::setprecision(6);

    for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
                {
                    file << vec[i * cols + j];
                    if(j < cols - 1)
                        {
                            file << ",";
                        }
                }
            file << std::endl;
        }

    file.close();
}

template <typename T>
void copyAndWriteVectorToCSV(const thrust::device_vector<T>& d_vec, const std::string& filename, int rows, int cols, bool append = false)
{
    thrust::host_vector<T> h_vec(d_vec.size());
    cudaMemcpy(thrust::raw_pointer_cast(h_vec.data()), thrust::raw_pointer_cast(d_vec.data()), d_vec.size() * sizeof(T),
               cudaMemcpyDeviceToHost);
    writeVectorToCSV(h_vec, filename, rows, cols, append);
}

template <typename T>
inline void writeValueToCSV(const T& value, const std::string& filename)
{
    std::ofstream file;
    file.open(filename, std::ios_base::app);  // Open in append mode

    // Set precision for floating-point numbers
    if constexpr(std::is_floating_point_v<std::decay_t<decltype(value)>>)
        {
            file << std::fixed << std::setprecision(10);
        }

    file << value << std::endl;
    file.close();
}

__device__ __forceinline__ float atomicMinFloat(float* addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value)))
                       : __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                       : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

// Cost-prune keep-probability for a NON-best node, normalized per region so the exponent k
// keeps its dynamic range (the raw min-ratio collapses when the region min approaches 0).
//   norm 0: min-ratio (m/cost)^k          -- original; ill-conditioned as m -> 0
//   norm 1: min-max  ((M-cost)/(M-m))^k   -- bounded [0,1]; M = region max cost
//   norm 2: mean     min(1,(mean/cost)^k) -- mean = sum/cnt; robust to outliers
// All modes return ~1 at cost==m, preserving the region-best exemption. Result floored by 'floor'.
__device__ __forceinline__ float costKeepProb(int norm, float m, float M, float sum, int cnt,
                                              float cost, float k, float floor)
{
    float p;
    if(norm == 1)
        p = (M > m) ? powf((M - cost) / (M - m), k) : 1.0f;
    else if(norm == 2)
        {
            float mean = (cnt > 0) ? sum / (float)cnt : cost;
            p          = fminf(1.0f, powf(mean / cost, k));
        }
    else
        p = powf(m / cost, k);
    return fmaxf(p, floor);
}

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if(code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if(abort) exit(code);
        }
}

inline int iDivUp(int a, int b)
{
    return (a + b - 1) / b;
}

__device__ __forceinline__ float distance(float* a, float* b)
{
    if(W_DIM == 2)
        {
            return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2));
        }
    else if(W_DIM == 3)
        {
            return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2));
        }
    else
        {
            return -1;
        }
}

// --------------------------------------------------------------------------------------
// Per-edge path cost from parent state x0 to child state x1, selected by COST_MODE (config.h).
// Used at every cumulative-cost accumulation site (planner kernels + benchmark computePathCost),
// so it is both host- and device-callable.
//   COST_MODE 1 (default): pure control effort. The double-integrator child stores its control
//     accel at x1[6..8] and the edge duration at x1[9], so edge cost = (ax^2+ay^2+az^2)*dt.
//   COST_MODE 0: baseline workspace Euclidean distance (self-contained; host-callable).
// An undefined MODEL evaluates to 0 in the #if, so the distance fallback is always safe.
// --------------------------------------------------------------------------------------
#ifndef COST_MODE
#define COST_MODE 1
#endif

__host__ __device__ __forceinline__ float edgeCost(const float* x0, const float* x1)
{
#if (COST_MODE == 1) && (MODEL == 1)
    // Pure control effort: integral of ||a||^2 over the edge (no distance term).
    float ax = x1[6], ay = x1[7], az = x1[8], dt = x1[9];
    (void)x0;
    return (ax * ax + ay * ay + az * az) * dt;
#else
    // Baseline / non-Model-1 fallback: workspace Euclidean distance.
    float s = 0.0f;
    for(int d = 0; d < W_DIM; ++d)
        {
            float diff = x1[d] - x0[d];
            s += diff * diff;
        }
    return sqrtf(s);
#endif
}
