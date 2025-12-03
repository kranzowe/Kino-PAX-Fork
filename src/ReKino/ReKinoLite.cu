#include "ReKino/ReKinoLite.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

ReKinoLite::ReKinoLite()
{
    h_numWarps_ = 512;  // 512 warps = 16,384 threads

    d_warpPaths_       = thrust::device_vector<float>(h_numWarps_ * MAX_PATH_LENGTH * SAMPLE_DIM);
    d_pathLengths_     = thrust::device_vector<int>(h_numWarps_);
    d_goalFound_       = thrust::device_vector<int>(1);
    d_goalSample_      = thrust::device_vector<float>(SAMPLE_DIM);

    d_warpPaths_ptr_   = thrust::raw_pointer_cast(d_warpPaths_.data());
    d_pathLengths_ptr_ = thrust::raw_pointer_cast(d_pathLengths_.data());
    d_goalFound_ptr_   = thrust::raw_pointer_cast(d_goalFound_.data());
    d_goalSample_ptr_  = thrust::raw_pointer_cast(d_goalSample_.data());

    d_spatialHashGrid_ = createSpatialHashGrid();

    if(VERBOSE)
    {
        printf("/* Planner Type: ReKinoLite (Warp-Based) */\n");
        printf("/* Number of Warps: %d */\n", h_numWarps_);
        printf("/* Total Threads: %d */\n", h_numWarps_ * WARP_SIZE);
        printf("/* Max Path Length: %d */\n", MAX_PATH_LENGTH);
        printf("/***************************/\n");
    }
}

ReKinoLite::~ReKinoLite()
{
    destroySpatialHashGrid(d_spatialHashGrid_);
}

void ReKinoLite::plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, bool saveTree)
{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Initialize
    thrust::fill(d_pathLengths_.begin(), d_pathLengths_.end(), 0);
    thrust::fill(d_goalFound_.begin(), d_goalFound_.end(), 0);

    // Copy initial and goal states
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize all warps with initial state
    for(uint warp_id = 0; warp_id < h_numWarps_; warp_id++)
    {
        cudaMemcpy(d_warpPaths_ptr_ + warp_id * MAX_PATH_LENGTH * SAMPLE_DIM,
                   h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Build spatial hash grid
    updateSpatialHashGrid(d_spatialHashGrid_, d_obstacles_ptr, h_obstaclesCount);
    cudaMemcpy(&h_spatialHashGrid_, d_spatialHashGrid_, sizeof(SpatialHashGrid), cudaMemcpyDeviceToHost);

    // Initialize random seeds
    initializeRandomSeeds(static_cast<unsigned int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()));

    // Launch kernel: each block = 1 warp
    int maxIterations = MAX_ITER_REKINO;
    rekino_lite_warp_kernel<<<h_numWarps_, WARP_SIZE>>>(
        h_initial,
        d_goalSample_ptr_,
        d_obstacles_ptr,
        h_obstaclesCount,
        h_spatialHashGrid_,
        d_warpPaths_ptr_,
        d_pathLengths_ptr_,
        d_goalFound_ptr_,
        d_randomSeeds_ptr_,
        maxIterations
    );

    cudaDeviceSynchronize();

    // Check if goal was found
    int goalFound;
    cudaMemcpy(&goalFound, d_goalFound_ptr_, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);

    if(goalFound > 0)
    {
        int winning_warp = goalFound - 1;
        printf("ReKinoLite: Goal found by warp %d in %.3f seconds!\n", winning_warp, milliseconds / 1000.0);

        // Copy solution path
        int pathLength;
        cudaMemcpy(&pathLength, d_pathLengths_ptr_ + winning_warp, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Path length: %d nodes\n", pathLength);

        // Write tree to CSV if requested
        if(saveTree)
        {
            writeTreeToCSV(winning_warp);
        }
    }
    else
    {
        printf("ReKinoLite: Goal not found after %.3f seconds\n", milliseconds / 1000.0);
    }
}

void ReKinoLite::writeExecutionTimeToCSV(double time)
{
    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/ExecutionTime");
    filename.str("");
    filename << "Data/ExecutionTime/executionTime.csv";
    writeValueToCSV(time, filename.str());
}

void ReKinoLite::writeTreeToCSV(int winning_warp_id)
{
    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/ReKinoLiteTree");
    filename.str("");
    filename << "Data/ReKinoLiteTree/rekino_lite_tree.csv";

    // Write all warp paths (one row per warp's path)
    // Each row contains: [node_0][node_1]...[node_MAX_PATH_LENGTH] for that warp
    copyAndWriteVectorToCSV(
        d_warpPaths_,
        filename.str(),
        h_numWarps_,                    // Number of rows (one per warp)
        MAX_PATH_LENGTH * SAMPLE_DIM,   // Columns per row (max path length × state dimension)
        false                           // Don't append, overwrite
    );

    // Write the path lengths (how deep each warp's path is)
    filename.str("");
    filename << "Data/ReKinoLiteTree/rekino_lite_depths.csv";
    copyAndWriteVectorToCSV(
        d_pathLengths_,
        filename.str(),
        h_numWarps_,  // Number of rows
        1,            // One depth value per warp
        false
    );

    printf("Tree written to Data/ReKinoLiteTree/rekino_lite_tree.csv\n");
    printf("Path lengths written to Data/ReKinoLiteTree/rekino_lite_depths.csv\n");
    printf("Winning warp: %d\n", winning_warp_id);
}


/***************************/
/* REKINO LITE WARP KERNEL */
/***************************/
__global__ void rekino_lite_warp_kernel(
    float* initial_state,
    float* goal_state,
    float* obstacles,
    int obstaclesCount,
    SpatialHashGrid spatialHashGrid,
    float* warpPaths,
    int* pathLengths,
    int* goalFound,
    curandState* randomSeeds,
    int maxIterations
)
{
    // Warp-level organization
    int warp_id = blockIdx.x;
    int lane_id = threadIdx.x;  // 0-31

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

    // Get pointers to this warp's path
    float* myPath = &warpPaths[warp_id * MAX_PATH_LENGTH * SAMPLE_DIM];
    int* myPathLength = &pathLengths[warp_id];

    // Each thread gets its own random seed
    curandState localSeed = randomSeeds[blockIdx.x * blockDim.x + threadIdx.x];

    // Shared memory for warp collaboration
    __shared__ float s_candidateStates[WARP_SIZE * SAMPLE_DIM];  // 32 candidate propagations
    __shared__ float s_distances[WARP_SIZE];                     // Distance to goal for each
    __shared__ bool s_valid[WARP_SIZE];                          // Is collision-free?
    __shared__ int s_bestIdx;                                    // Index of best candidate
    __shared__ int s_currentDepth;                               // Current path length

    // Initialize
    if(lane_id == 0)
    {
        s_currentDepth = 0;
        s_bestIdx = -1;
    }
    warp.sync();

    // Main loop
    for(int iter = 0; iter < maxIterations; iter++)
    {
        // Check if any warp found goal
        if(*goalFound > 0) return;

        // Get current node (last node in path)
        float currentNode[SAMPLE_DIM];
        for(int d = 0; d < SAMPLE_DIM; d++)
        {
            currentNode[d] = myPath[s_currentDepth * SAMPLE_DIM + d];
        }

        // STEP 1: Each thread propagates a random control
        float* myCandidate = &s_candidateStates[lane_id * SAMPLE_DIM];
        bool valid = propagateAndCheckSpatialHash(currentNode, myCandidate, &localSeed, spatialHashGrid, obstacles, obstaclesCount);

        // STEP 2: Compute distance to goal for valid propagations
        float distToGoal = INFINITY;
        if(valid)
        {
            distToGoal = 0.0f;
            for(int d = 0; d < W_DIM; d++)
            {
                float diff = myCandidate[d] - goal_state[d];
                distToGoal += diff * diff;
            }
            distToGoal = sqrtf(distToGoal);
        }

        s_distances[lane_id] = distToGoal;
        s_valid[lane_id] = valid;
        warp.sync();

        // STEP 3: Warp leader finds best valid candidate
        if(lane_id == 0)
        {
            float bestDist = INFINITY;
            int bestIdx = -1;

            for(int i = 0; i < WARP_SIZE; i++)
            {
                if(s_valid[i] && s_distances[i] < bestDist)
                {
                    bestDist = s_distances[i];
                    bestIdx = i;
                }
            }

            s_bestIdx = bestIdx;

            // Check if we reached goal
            if(bestIdx >= 0 && bestDist < GOAL_THRESH)
            {
                // Goal found!
                atomicCAS(goalFound, 0, warp_id + 1);  // Mark this warp as winner
            }
        }
        warp.sync();

        // STEP 4: Handle result
        if(s_bestIdx >= 0)
        {
            // Success! Extend path with best candidate
            if(lane_id == 0)
            {
                s_currentDepth++;
                if(s_currentDepth >= MAX_PATH_LENGTH)
                {
                    // Path too long, restart from beginning
                    s_currentDepth = 0;
                }
            }
            warp.sync();

            // Copy best candidate to path
            float* bestCandidate = &s_candidateStates[s_bestIdx * SAMPLE_DIM];
            if(lane_id < SAMPLE_DIM)
            {
                myPath[s_currentDepth * SAMPLE_DIM + lane_id] = bestCandidate[lane_id];
            }
            warp.sync();
        }
        else
        {
            // All 32 propagations collided! Backtrack randomly
            if(lane_id == 0)
            {
                if(s_currentDepth > 0)
                {
                    // Random backtrack 1-5 nodes
                    int backtrackAmount = 1 + (int)(curand_uniform(&localSeed) * 5);
                    s_currentDepth = max(0, s_currentDepth - backtrackAmount);
                }
                // If already at root, stay there and try again
            }
            warp.sync();
        }

        // Update path length
        if(lane_id == 0)
        {
            *myPathLength = s_currentDepth;
        }
    }

    // Save random seed
    randomSeeds[blockIdx.x * blockDim.x + threadIdx.x] = localSeed;
}
