#pragma once
#include "planners/Planner.cuh"
#include "collisionCheck/spatialHash.cuh"

/*
 * ReKinoLite: Ultra-Lightweight Block-Based Kinodynamic Planner
 *
 * Each block (256 threads = 8 warps) maintains ONE path cooperatively:
 * - All 256 threads propagate different random controls from current node
 * - Pick the collision-free propagation closest to goal
 * - If all collide, backtrack randomly
 * - Repeat until goal found or max iterations
 *
 * No graph, no complex data structures - just simple greedy search!
 * Goal: Fast, occasionally successful paths with more samples per iteration
 */

#define WARP_SIZE 32
#define MAX_PATH_LENGTH 200  // Maximum nodes in a path per block

class ReKinoLite : public Planner
{
public:
    /**************************** CONSTRUCTORS ****************************/
    ReKinoLite();
    ~ReKinoLite();

    /****************************    METHODS    ****************************/
    void plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, bool saveTree = false) override;
    void writeExecutionTimeToCSV(double time);
    void writeTreeToCSV(int winning_warp_id);
    void setTreeOutputPrefix(const std::string& prefix) { treeOutputPrefix_ = prefix; }

    /****************************    FIELDS    ****************************/

    // --- Host fields ---
    uint h_numWarps_;           // Number of parallel warps searching
    uint h_samplesPerThread_;   // How many controls each thread samples (default: 1)
    float h_epsilonGreedy_;     // Probability of random selection vs greedy (0.0 = pure greedy, 1.0 = pure random)
    std::string treeOutputPrefix_;  // Prefix for tree output files (default: empty)

    // --- Device fields ---

    // Per-warp path storage: [num_warps][MAX_PATH_LENGTH][SAMPLE_DIM]
    thrust::device_vector<float> d_warpPaths_;
    thrust::device_vector<int> d_pathLengths_;     // Current length for each warp's path
    float* d_warpPaths_ptr_;
    int* d_pathLengths_ptr_;

    // Goal tracking
    thrust::device_vector<int> d_goalFound_;       // 0 = searching, warp_id+1 = found by that warp
    thrust::device_vector<float> d_goalSample_;
    int* d_goalFound_ptr_;
    float* d_goalSample_ptr_;

    // Spatial hash for collision detection
    SpatialHashGrid* d_spatialHashGrid_;
    SpatialHashGrid h_spatialHashGrid_;
};

/**************************** DEVICE FUNCTIONS ****************************/

/****************************/
/* REKINO LITE BLOCK KERNEL */
/****************************/
/*
 * Ultra-lightweight block-based search
 *
 * Each block (256 threads) collaborates to extend one path:
 * 1. All 256 threads propagate random controls from current node
 * 2. Find best collision-free propagation (closest to goal)
 * 3. Extend path with that node
 * 4. If all collide, randomly backtrack 1-5 nodes
 * 5. Repeat until goal found or max iterations (or 6-second timeout)
 *
 * Block-level synchronization ensures coherent branching
 * Much better sampling density: 256 threads × samplesPerThread (vs 32 × samplesPerThread)
 */
__global__ void rekino_lite_warp_kernel(
    float* initial_state,
    float* goal_state,
    float* obstacles,
    int obstaclesCount,
    SpatialHashGrid spatialHashGrid,
    float* warpPaths,        // [num_warps][MAX_PATH_LENGTH][SAMPLE_DIM]
    int* pathLengths,        // [num_warps]
    int* goalFound,          // Global flag
    curandState* randomSeeds,
    int maxIterations,
    int samplesPerThread,    // How many controls each thread samples
    float epsilonGreedy,     // Epsilon-greedy exploration parameter
    unsigned long long maxClockCycles  // 6-second timeout in clock cycles
);
