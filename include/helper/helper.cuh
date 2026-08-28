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

// Cost keep-probability for the weighted-sum variants (KinoPaxSTARWeightedCost).
//
// Exactly 1.0 at the region min, then a smooth decay in the NORMALIZED EXCESS
// (cost - m) / (mean - m). Contrast with costKeepProb's norm 2, min(1,(mean/cost)^k), which is
// also 1 at the min but stays pinned at 1 for EVERY cost at or below the mean -- so the whole
// lower half of a region gets no gradient -- and never reaches 0 (0.33 at 6x the min). Here,
// with k = 1: 1.00 at the min, 0.61 at 1.5x, 0.37 at the mean, 0.05 at 4x, 0.007 at 6x.
//
// Deliberately has NO floor: the weighted-sum form adds P_floor exactly once, in
// weightedAccept(). Dimensionless, so it behaves identically under either COST_MODE, and it
// needs only min/sum/cnt -- no outlier-sensitive running max.
__device__ __forceinline__ float costProbExp(float m, float sum, int cnt, float cost, float k)
{
    if(cnt <= 0) return 1.0f;   // no samples yet: no evidence, so do not penalize
    float mean = sum / (float)cnt;
    float d    = mean - m;
    if(d <= 0.0f) return 1.0f;  // degenerate region: every node equally good
    // The clamp guards a transient cost < m: minCostsR1 is lowered by atomicMinFloat during
    // propagation, so a thread can read a cost below the min it was compared against.
    return fminf(1.0f, __expf(-k * (cost - m) / d));
}

// Cost keep-probability with a LOCAL reference and a GLOBAL scale.
//
// costProbExp above divides by (mean_r - m_r), the region's OWN spread. That pins the typical
// candidate at x ~ 1 in EVERY region by construction -- which is what the mean means -- so
// exp(-k*x) ~ exp(-k) uniformly across the grid and k loses its grip entirely. It also makes k mean
// different things in different places: a corridor region whose costs span 2 units punishes a
// 2-unit excess exactly as hard as an open region punishes a 30-unit one.
//
// Here the reference stays the region's own minimum -- so the bias toward each region's cheapest
// node survives, which is what keeps the search from collapsing onto the root, where all the
// globally-cheapest nodes live -- and only the SCALE is global:
//
//     x = (cost - m_r) / dGlobal,   dGlobal = globalMeanCost - globalMinCost
//
// A given cost excess now means the same thing everywhere. Numerator and denominator are both in
// cost units, so this is COST_MODE-agnostic: no heuristic, no mixing of distance with effort.
__device__ __forceinline__ float costProbExpGlobal(float m, float cost, float dGlobal, float k)
{
    if(dGlobal <= 0.0f) return 1.0f;   // no global spread yet: no evidence, so do not penalize
    return fminf(1.0f, __expf(-k * (cost - m) / dGlobal));
}

// ======================================================================================
// ======================================================================================
// KinoPaxSTARCOMBO shape functions.
//
// TWO shapes, not one, and the split is the point. One rule was being asked two different
// questions:
//   ACCEPTANCE  which nodes join the tree. Cost belongs here -- it is what makes COMBO optimal.
//   FAN-OUT     where to spend propagation. Cost is counter-productive here: cost is cumulative
//               root-to-node, so "cheap" means SHALLOW, and weighting fan-out by cost pours
//               propagation into the neighbourhood of the root. That is a density mechanism where
//               KPAX's novelty rule (validVertexCounter < 10 ? 15 : 1) is a REACH mechanism, and
//               it is why COMBO grew a bigger tree than CleanCost while reaching a first solution
//               later.
// Both use the same two deltas and the same blend; they differ only in their gains and their
// blend exponent, so a caller can make fan-out novelty-driven while acceptance stays cost-driven.
//
// Each shape BLENDS coverage against cost as the run progresses -- explore first, optimise later:
//
//     u  = treeSize / MAX_TREE_SIZE                 how full is the tree, in [0,1]
//     v  = u ^ (ln 0.5 / ln mid)                    remap so v = 0.5 exactly at u = mid
//     wCov = (1-v)^g      wCst = v^g
//     shape = (sigma(-kCov*d1)*wCov + sigma(-kCst*d3)*wCst) / (wCov + wCst)
//
// NORMALISED BY THE WEIGHT SUM, not by a constant. At g = 1 the weights already partition unity, so
// a constant divisor of 2 would return 0.25 at the neutral point rather than 0.5; and at g != 1 the
// raw weights do NOT sum to 1 (at u = 0.5, g = 2 they sum to 0.5), so a constant divisor would let
// g rescale the whole shape instead of only reshaping the transition. Dividing by the sum pins a
// neutral candidate at exactly 0.5 for every g, every mid and every u -- which is what lets g be a
// pure sharpness knob and lets COMBO_NEUTRAL_SHAPE stay a compile-time constant.
//
// mid AND g DO DIFFERENT JOBS. mid sets WHERE the crossover happens (wCov == wCst at u == mid); g
// sets HOW SHARPLY it happens there. g alone cannot move the crossover: (1-u)^g and u^g are equal
// at u = 0.5 for every g. At mid = 0.5 the remap exponent is 1, v = u, and the whole thing reduces
// to the plain (1-u), u crossfade.
//
// EVERY DELTA IS SIGNED SO d > 0 MEANS UNFAVOURABLE, and each is divided by a GLOBAL reference.
// The global scale is load-bearing twice over:
//   - Raw deltas have no usable range. With NUM_R2_PER_R1 = 64 a one-sub-region coverage
//     difference is 0.0156, and sigmoid(0.0156) = 0.4961 -- the coverage term would be the constant
//     0.5 to three digits. At the other extreme (cost - mean) is in cost units, O(1e2) under
//     COST_MODE 1, so the cost term would saturate to a hard step and its gain would be inert.
//     That is exactly the failure costProbExpGlobal above was written to escape.
//   - A LOCAL scale (the region's own spread) pins the typical candidate at d ~ 1 in EVERY region
//     by construction -- see costProbExpGlobal's comment -- which is what made the old k knob inert.
//
// k ENTERS THE ARGUMENT, NOT AS AN EXPONENT. sigmoid(x)^k moves the midpoint to 2^-k and its slope
// actually FALLS past k = 2 (0.250 at k=1 and k=2, 0.188 at k=3, 0.125 at k=4) -- it squashes
// rather than sharpens. sigmoid(k*x) holds the midpoint at 0.5 for every k with slope k/4.
//
// k = 0 pins its term at 0.5 -- an exact ablation switch. With BOTH gains 0 a shape is the constant
// 0.5, which for fan-out means uniform rep: the KinoPaxPlus/CleanCost control arm.
//
// GAIN SETS THE SPREAD THE FAN-OUT THRESHOLD LIVES ON. Fan-out now favours nodes scoring above
// mu + N*sigma over the frontier's own score distribution, so what kFan controls is sigma. At low
// gain the shape crowds around 0.5, sigma is small, and mu + N*sigma sits just above the mean --
// a narrow, arbitrary slice. At high gain the sigmoid degenerates to a step, the shape goes bimodal
// {~0, ~1}, sigma is large, and the threshold lands squarely between the two modes: exactly the
// favoured MINORITY that KPAX's 15/1 gets from its hardcoded validVertexCounter < 10, but relative
// to the explored mean and recomputed every iteration. So kFan is still the headline tuning axis,
// and LOW kFan is still the failure mode -- but the threshold no longer has to chase it.
//
// COLD START is the caller's job, because the caller holds the raw arrays: pass
// r1MeanCost = nodeCost when cntCostsR1[r] == 0. That collapses d3 to 0 (neutral).
// ======================================================================================

// A neutral candidate -- every delta zero, so every sigmoid exactly 0.5 -- scores this, for every
// gain, blend exponent, midpoint and u. Used wherever a node needs an "average" shape without one
// being computed.
#define COMBO_NEUTRAL_SHAPE 0.5f

// Blend weights for one shape. Split out so the shapes and the diagnostics share one definition.
__device__ __forceinline__ void comboBlendWeights(float u, float g, float mid, float* wCov, float* wCst)
{
    u = fminf(1.0f, fmaxf(0.0f, u));
    // Remap so v = 0.5 exactly at u = mid, leaving the endpoints pinned (0 -> 0, 1 -> 1) so
    // coverage still owns the very start of a run whatever the midpoint is.
    float v;
    if(mid > 0.0f && mid < 1.0f && fabsf(mid - 0.5f) > 1e-6f)
        v = __powf(u, __logf(0.5f) / __logf(mid));
    else
        v = u;   // mid == 0.5 (or degenerate): the identity remap, i.e. the plain crossfade
    *wCov = __powf(1.0f - v, g);
    *wCst = __powf(v, g);
}

// One shape from the two deltas. Returns (0,1), exactly COMBO_NEUTRAL_SHAPE at the neutral point.
__device__ __forceinline__ float comboShape2(float d1, float d3, float kCov, float kCst,
                                             float u, float g, float mid)
{
    float wCov, wCst;
    comboBlendWeights(u, g, mid, &wCov, &wCst);
    float tCov = 1.0f / (1.0f + __expf(kCov * d1));
    float tCst = 1.0f / (1.0f + __expf(kCst * d3));
    float wSum = wCov + wCst;
    // wSum can underflow to 0 only if u is exactly 0 or 1 AND g is large enough that the surviving
    // weight also underflows; fall back to the neutral value rather than dividing by zero.
    if(!(wSum > 1e-20f)) return COMBO_NEUTRAL_SHAPE;
    return (tCov * wCov + tCst * wCst) / wSum;
}

// The two deltas, from the raw region statistics. Shared by both shapes so there is exactly one
// definition of "how far from average is this candidate".
__device__ __forceinline__ void comboDeltas(float nodeCost, float r1MeanCost, float costScale,
                                            float r1Coverage, float exploredMeanCoverage,
                                            float* d1, float* d3)
{
    // d1: prefer regions covered LESS than the explored average -- steer toward the thin places.
    *d1 = (exploredMeanCoverage > 0.0f) ? (r1Coverage - exploredMeanCoverage) / exploredMeanCoverage : 0.0f;
    // d3: prefer nodes cheaper than their OWN region's mean, measured on the GLOBAL cost scale.
    *d3 = (costScale > 0.0f) ? (nodeCost - r1MeanCost) / costScale : 0.0f;
}

// Two-level fan-out: a node is either in the favoured minority or it is not.
//
// REPLACES A PROPORTIONAL RULE THAT COULD NOT CONCENTRATE. The first form was
// rep = clamp(repTarget * shape, 1, repeatMax), and it failed for a structural reason worth
// recording. It thresholds each candidate against the MEAN of its deltas, and both deltas are
// right-skewed -- coverage is floored at -1 and unbounded above, cost has a long expensive tail --
// so mean > median and MOST candidates land on the favourable side. The measured favoured fraction
// came out above 0.5 at every gain. Raising the gain only sharpens the step; it cannot move where
// the step sits, so the "boost" was going to ~70% of the frontier. KPAX's 15/1 works precisely
// because its 15 goes to a shrinking MINORITY of under-sampled regions.
//
// THE FIX IS A SCALE-FREE THRESHOLD, NOT A TRACKED ONE. The second form fed a threshold back toward
// a target fraction phi, which put the step in the tail but bought a second failure: the fraction
// driving the feedback was measured in the accept kernel over PRE-GATE candidates against the
// PREVIOUS threshold, while this function was applied in the update kernel to post-gate survivors
// plus every Part B node against the NEW one. Different population, different threshold, one
// iteration apart -- so the block budget was sized for a favoured count that never materialised,
// repHi pinned at repeatMax, and sum(rep) came uncoupled from the budget it was solved against.
//
// Now the caller thresholds at mu + N*sigma over the score distribution of the WHOLE REALISED
// FRONTIER, measured after the frontier is compacted and therefore complete and countable. That is
// scale-free: a fixed N picks the tail of a skewed distribution whatever the gains do to its spread,
// with nothing to track and nothing to lag. See the planner's h_fanSigmaN_.
//
// THE >= 1 FLOOR IS STRUCTURAL, and now doubly so: the caller runs this over exactly the compacted
// frontier, so every frontier member receives a count by construction rather than by a clamp.
//
// Why the floor exists at all: repeatInd emits nothing for count 0, and on the kernel1 path the
// frontier bit is cleared BY THE EXPANDING BLOCK ITSELF -- so a node with no block is never
// expanded and never cleared. Its frontier bit stays true forever, inflating h_frontierSize_, and
// Part B cannot rescue it because its reactivation guard is frontier == 0.
__device__ __forceinline__ unsigned int repeatFromScore(float score, float threshold, unsigned int repHi)
{
    // NaN-safe: an unordered compare falls through to the unfavoured branch.
    if(!(score > threshold)) return 1u;
    return (repHi >= 1u) ? repHi : 1u;
}

// Weighted-sum acceptance: P = min(1, w*P_syclop + (1-w)*P_cost + P_floor).
// w = 1 recovers the KPAX rule (P_syclop = vertexScore + fAccept) plus the floor; w = 0 is pure
// cost-greedy. Replaces the multiplicative `costProb * syclop` blend, which collapses to zero in
// exactly the low-score cells that hold a narrow passage (see Graph.cu's quartic freeVol term).
__device__ __forceinline__ float weightedAccept(float w, float pSyclop, float pCost, float pFloor)
{
    return fminf(1.0f, w * pSyclop + (1.0f - w) * pCost + pFloor);
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
