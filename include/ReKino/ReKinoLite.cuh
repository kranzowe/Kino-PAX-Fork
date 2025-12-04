#pragma once
#include "planners/Planner.cuh"
#include "collisionCheck/spatialHash.cuh"

/*
 * ReKinoLite: Ultra-Lightweight Warp-Based Kinodynamic Planner
 *
 * Each warp (32 threads) maintains ONE branch cooperatively:
 * - All 32 threads propagate different random controls from current node
 * - Pick the collision-free propagation closest to goal
 * - If all collide, backtrack randomly
 * - Repeat until goal found or max iterations
 *
 * No graph, no complex data structures - just simple greedy search!
 * Goal: Fast, occasionally successful paths
 */

#define WARP_SIZE 32
#define MAX_PATH_LENGTH 200  // Maximum nodes in a path per warp

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

/***************************/
/* REKINO LITE WARP KERNEL */
/***************************/
/*
 * Ultra-lightweight warp-based search
 *
 * Each warp collaborates to extend one path:
 * 1. All 32 threads propagate random controls from current node
 * 2. Find best collision-free propagation (closest to goal)
 * 3. Extend path with that node
 * 4. If all collide, randomly backtrack 1-10 nodes
 * 5. Repeat until goal found or max iterations
 *
 * Warp-level synchronization ensures coherent branching
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
    float epsilonGreedy      // Epsilon-greedy exploration parameter
);
