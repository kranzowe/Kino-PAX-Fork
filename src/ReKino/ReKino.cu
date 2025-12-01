#include "ReKino/ReKino.cuh"
#include "config/config.h"
#include <thread>
#include <chrono>


ReKino::ReKino()
{
    // compute the number of threads to launch. 
    // we have 64 SMs and will use 256 per sm. though i think they have more. 
    // this gives 16k parallel branches.
    h_numThreads_ = 64 * 256;
    
    // how many nodes any one branch can have. 
    h_maxBranchLength_ = 500;

    // To track explored regions, we dont use a graph. is just a coarse grid
    // explored_regions[region_idx] = true if explored / false else
    // its a very lightweight tracker. Hopefully its enough
    d_exploredRegions_ = thrust::device_vector<bool>(NUM_R1_REGIONS);
    d_exploredRegions_ptr_ = thrust::raw_pointer_cast(d_exploredRegions_.data());
    
    // each thread needs a way to track nodes! No need for parents since each branch is linear
    // Layout: [thread_0_branch][thread_1_branch]...[thread_N_branch]
    // where each branch is: [node_0][node_1]...[node_depth]
    d_allBranches_ = thrust::device_vector<float>(h_numThreads_ * h_maxBranchLength_ * SAMPLE_DIM);
    d_allControls_ = thrust::device_vector<float>(h_numThreads_ * h_maxBranchLength_ * CONTROL_DIM);
    d_branchDepths_ = thrust::device_vector<int>(h_numThreads_);
    
    d_allBranches_ptr_ = thrust::raw_pointer_cast(d_allBranches_.data());
    d_allControls_ptr_ = thrust::raw_pointer_cast(d_allControls_.data());
    d_branchDepths_ptr_ = thrust::raw_pointer_cast(d_branchDepths_.data());
    
    // need a way for threads to chit chat. This is for seeing goal have been found
    // and tracking which branch found it
    d_goalFound_ = thrust::device_vector<int>(1);
    d_solutionThreadId_ = thrust::device_vector<int>(1);
    
    d_goalFound_ptr_ = thrust::raw_pointer_cast(d_goalFound_.data());
    d_solutionThreadId_ptr_ = thrust::raw_pointer_cast(d_solutionThreadId_.data());
    
    // goal state storage for all to check
    d_goalSample_ = thrust::device_vector<float>(SAMPLE_DIM);
    d_goalSample_ptr_ = thrust::raw_pointer_cast(d_goalSample_.data());
    
    if(VERBOSE)
    {
        printf("/***************************/\n");
        printf("/* Planner Type: ReKino */\n");
        printf("/* Number of parallel threads: %d */\n", h_numThreads_);
        printf("/* Max branch length per thread: %d */\n", h_maxBranchLength_);
        printf("/* Number of regions: %d */\n", NUM_R1_REGIONS);
        printf("/* Total branch storage: %.2f MB */\n", 
               (h_numThreads_ * h_maxBranchLength_ * SAMPLE_DIM * sizeof(float)) / (1024.0f * 1024.0f));
        printf("/***************************/\n");
    }
}

/*
 * ReKino::plan
 * 
 * Main planning function. Launches persistent kernel and monitors for goal.
 * 
 * Key differences from KPAX:
 * - Single kernel launch (no iteration loop with multiple kernel calls)
 * - CPU monitors goal_found flag and times out if needed
 * - No explicit synchronization during search (kernel runs continuously)
 */
void ReKino::plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
{
    // for benchmarking
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    
    // Clear all tracking arrays
    // set the explored region vector to false, 
    // the depth of each branch to 0
    // whether the goal has been found to 0
    // solution thread id to -1
    thrust::fill(d_exploredRegions_.begin(), d_exploredRegions_.end(), false);
    thrust::fill(d_branchDepths_.begin(), d_branchDepths_.end(), 0);
    thrust::fill(d_goalFound_.begin(), d_goalFound_.end(), 0);
    thrust::fill(d_solutionThreadId_.begin(), d_solutionThreadId_.end(), -1);
    
    // Initialize solution storage (inherited from Planner)
    thrust::fill(d_treeSamples_.begin(), d_treeSamples_.end(), 0.0f);
    thrust::fill(d_controlPathToGoal_.begin(), d_controlPathToGoal_.end(), 0.0f);
    
    // Copy initial and goal states to device
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize all thread branches with root node
    // Each thread starts from the same initial state
    for(uint i = 0; i < h_numThreads_; i++)
    {
        cudaMemcpy(
            &d_allBranches_ptr_[i * h_maxBranchLength_ * SAMPLE_DIM],
            h_initial,
            SAMPLE_DIM * sizeof(float),
            cudaMemcpyHostToDevice
        );
    }
    
    // Initialize random seeds (one per thread for diversity)
    initializeRandomSeeds(static_cast<unsigned int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    ));
    
    // ========================================================================
    // LAUNCH PERSISTENT KERNEL
    // ========================================================================
    
    // Launch configuration:
    // - Use enough blocks to saturate GPU (one block per SM is typical)
    // - Threads per block = 256 (good balance for most GPUs)
    int blocks = h_numThreads_ / 256;  // e.g., 16384 / 256 = 64 blocks
    int threads_per_block = 256;
    
    if(VERBOSE)
    {
        printf("Launching ReKino kernel: %d blocks × %d threads = %d threads\n",
               blocks, threads_per_block, h_numThreads_);
    }
    
    // Launch the persistent kernel (runs until goal found or max iterations)
    rekino_persistent_kernel<<<blocks, threads_per_block>>>(
        &d_allBranches_ptr_[0],          // Initial state (root) at start of branches array
        d_goalSample_ptr_,               // Goal state
        d_obstacles_ptr,                 // Obstacles for collision checking
        h_obstaclesCount,                // Number of obstacles
        d_exploredRegions_ptr_,          // Shared exploration tracker
        d_allBranches_ptr_,              // All thread branches
        d_allControls_ptr_,              // All thread controls
        d_branchDepths_ptr_,             // Current depth per thread
        d_goalFound_ptr_,                // Global flag: goal found?
        d_solutionThreadId_ptr_,         // Which thread found it?
        d_randomSeeds_ptr_,              // Random number generators
        MAX_ITER,                        // Max iterations before giving up
        h_maxBranchLength_               // Max branch depth
    );
    
    // ========================================================================
    // MONITOR FOR COMPLETION
    // ========================================================================
    
    // CPU polls the goal_found flag while GPU kernel runs
    // This is more efficient than synchronizing repeatedly
    int h_goalFound = 0;
    int poll_count = 0;
    
    while(h_goalFound == 0 && poll_count < 10000)  // Timeout after ~10 seconds
    {
        // Check goal flag every 1ms (balance between responsiveness and CPU usage)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        cudaMemcpy(&h_goalFound, d_goalFound_ptr_, sizeof(int), cudaMemcpyDeviceToHost);
        poll_count++;
        
        // Optional: print progress every second
        if(VERBOSE && poll_count % 1000 == 0)
        {
            printf("Still searching... (%.1f seconds)\n", poll_count / 1000.0f);
        }
    }
    
    // Wait for kernel to fully complete
    cudaDeviceSynchronize();
    
    // ========================================================================
    // EXTRACT SOLUTION
    // ========================================================================
    
    if(h_goalFound == 1)
    {
        // Success! Extract the winning thread's path
        int h_solutionThreadId;
        cudaMemcpy(&h_solutionThreadId, d_solutionThreadId_ptr_, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Get the depth of the solution
        int solution_depth;
        cudaMemcpy(&solution_depth, &d_branchDepths_ptr_[h_solutionThreadId], sizeof(int), cudaMemcpyDeviceToHost);
        
        if(VERBOSE)
        {
            printf("Goal found by thread %d at depth %d!\n", h_solutionThreadId, solution_depth);
        }
        
        // Copy solution path to output
        // The winning thread's branch contains the path from root to goal
        int branch_offset = h_solutionThreadId * h_maxBranchLength_ * SAMPLE_DIM;
        cudaMemcpy(
            h_controlPathToGoal_,
            &d_allBranches_ptr_[branch_offset],
            (solution_depth + 1) * SAMPLE_DIM * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        
        h_pathToGoal_ = solution_depth + 1;
    }
    else
    {
        if(VERBOSE)
        {
            printf("Timeout: Goal not found within time limit\n");
        }
        h_pathToGoal_ = 0;
    }
    
    // ========================================================================
    // CLEANUP AND REPORTING
    // ========================================================================
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    
    std::cout << "ReKino execution time: " << milliseconds / 1000.0 
              << " seconds. Path length: " << h_pathToGoal_ << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// PERSISTENT KERNEL IMPLEMENTATION
// ============================================================================

/*
 * Main ReKino Kernel
 * 
 * Each thread runs this loop continuously until:
 * 1. Any thread finds the goal (goalFound flag set to 1)
 * 2. Max iterations reached (give up)
 * 3. Branch becomes too deep (max_branch_length exceeded)
 * 
 * Algorithm per thread:
 * Loop:
 *   - Get current node from my branch
 *   - Sample a control and propagate
 *   - Check collision
 *   - If valid:
 *       - Add to branch
 *       - Mark region as explored
 *       - Check if goal reached
 *   - If collision:
 *       - Calculate neighborhood exploration density
 *       - Probabilistically decide how far to backtrack
 *       - Backtrack and retry from ancestor
 */
__global__ void rekino_persistent_kernel(
    float* initial_state,
    float* goal_state,
    float* obstacles,
    int obstaclesCount,
    bool* exploredRegions,
    float* allBranches,
    float* allControls,
    int* branchDepths,
    int* goalFound,
    int* solutionThreadId,
    curandState* randomSeeds,
    int maxIterations,
    int maxBranchLength
)
{
    // ========================================================================
    // THREAD IDENTIFICATION
    // ========================================================================
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread gets its own slice of the branches array
    // Layout: [my_branch][my_controls] where each has maxBranchLength entries
    float* my_branch = &allBranches[tid * maxBranchLength * SAMPLE_DIM];
    float* my_controls = &allControls[tid * maxBranchLength * CONTROL_DIM];
    int current_depth = 0;  // Start at root (index 0)
    
    // Initialize random state for this thread
    curandState local_rand_state = randomSeeds[tid];
    
    // ========================================================================
    // LOAD GOAL STATE INTO SHARED MEMORY (optimization)
    // ========================================================================
    
    // All threads in block share the same goal, so load once to fast memory
    __shared__ float s_goal[SAMPLE_DIM];
    if(threadIdx.x < SAMPLE_DIM)
    {
        s_goal[threadIdx.x] = goal_state[threadIdx.x];
    }
    __syncthreads();  // Wait for goal to be loaded
    
    // ========================================================================
    // MAIN PROPAGATION LOOP
    // ========================================================================
    
    for(int iter = 0; iter < maxIterations && *goalFound == 0; iter++)
    {
        // ====================================================================
        // STEP 1: Get current node to propagate from
        // ====================================================================
        
        float* current_node = &my_branch[current_depth * SAMPLE_DIM];
        
        // ====================================================================
        // STEP 2: Sample control and propagate
        // ====================================================================
        
        // Allocate space for new state
        float new_state[SAMPLE_DIM];
        float control[CONTROL_DIM];
        
        // Propagate using existing propagateAndCheck function (from KPAX)
        // This samples a random control and simulates the dynamics
        bool is_valid = propagateAndCheck(
            current_node,     // Starting state
            new_state,        // Output: new state after propagation
            &local_rand_state, // Random number generator
            obstacles,         // For collision checking
            obstaclesCount
        );
        
        // ====================================================================
        // STEP 3a: VALID SAMPLE - Extend branch
        // ====================================================================
        
        if(is_valid)
        {
            // Extend branch (move one level deeper)
            current_depth++;
            
            // Check if branch is getting too deep
            if(current_depth >= maxBranchLength)
            {
                // Branch full - reset to root and start over
                // This prevents memory overflow
                current_depth = 0;
                continue;
            }
            
            // Store the new node in our branch
            float* new_node_slot = &my_branch[current_depth * SAMPLE_DIM];
            for(int d = 0; d < SAMPLE_DIM; d++)
            {
                new_node_slot[d] = new_state[d];
            }
            
            // TODO: Store control as well (currently controls aren't extracted in propagateAndCheck)
            // Would need to modify propagateAndCheck to return the control used
            
            // ================================================================
            // STEP 3b: Mark region as explored
            // ================================================================
            
            // Figure out which region this new state is in
            int region = getRegion(new_state);
            
            // Mark as explored (no atomic needed - all threads just write 1)
            // Writing 1 multiple times is harmless and faster than atomics
            exploredRegions[region] = true;
            
            // ================================================================
            // STEP 3c: Check if we reached the goal
            // ================================================================
            
            float dist_to_goal = distance(new_state, s_goal);
            
            if(dist_to_goal < GOAL_THRESH)
            {
                // WE FOUND THE GOAL! 🎉
                
                // Use atomic compare-and-swap to ensure only ONE thread claims victory
                // This prevents race condition where multiple threads reach goal simultaneously
                int already_found = atomicCAS(goalFound, 0, 1);
                
                if(already_found == 0)
                {
                    // We're the first! Save our thread ID so CPU can extract our path
                    *solutionThreadId = tid;
                    branchDepths[tid] = current_depth;
                    
                    // Exit immediately - we're done!
                    return;
                }
                else
                {
                    // Another thread beat us to it, just exit
                    return;
                }
            }
            
            // Continue to next iteration (propagate from this new node)
        }
        
        // ====================================================================
        // STEP 4: COLLISION - Adaptive backtracking
        // ====================================================================
        
        else
        {
            // We hit an obstacle. Time to backtrack.
            // Question: How far should we backtrack?
            // Answer: Depends on how explored the current region is.
            //         Highly explored → backtrack far (try elsewhere)
            //         Barely explored → backtrack a little (keep trying here)
            
            // ================================================================
            // STEP 4a: Backtrack loop
            // ================================================================
            
            while(current_depth > 0)
            {
                // Get ancestor node at current depth
                float* ancestor = &my_branch[current_depth * SAMPLE_DIM];
                int region = getRegion(ancestor);
                
                // Calculate how explored this region's neighborhood is
                float exploration_density = get_neighborhood_exploration(
                    region,
                    exploredRegions
                );
                
                // ============================================================
                // STEP 4b: Probabilistically decide to stay or go back further
                // ============================================================
                
                // Adaptive probability formula:
                // High exploration → low probability of staying (backtrack more)
                // Low exploration → high probability of staying (stay nearby)
                
                // Exponential formula for stronger bias:
                // prob_stay = exp(-alpha * exploration_density) + epsilon
                float alpha = 3.0f;  // Tuning parameter (higher = more aggressive backtracking)
                float epsilon = 0.05f;  // Minimum probability (always some chance)
                
                float prob_stay = expf(-alpha * exploration_density) + epsilon;
                
                // Roll the dice
                if(curand_uniform(&local_rand_state) < prob_stay)
                {
                    // Accept this depth - will propagate from here next iteration
                    break;
                }
                
                // Reject - go back one more level
                current_depth--;
            }
            
            // If we backtracked all the way to root (current_depth == 0), that's okay
            // We'll try propagating from root again in the next iteration
            
            // Add occasional random restarts for diversity
            // This prevents all threads from getting stuck in the same region
            if(curand_uniform(&local_rand_state) < 0.01f)  // 1% chance
            {
                current_depth = 0;  // Jump back to root
            }
        }
        
        // Save updated random state back to global memory
        // (needed because random state is stateful)
        randomSeeds[tid] = local_rand_state;
    }
    
    // If we get here, we hit max iterations without finding goal
    // Just exit - goalFound will still be 0
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/*
 * Calculate neighborhood exploration density
 * 
 * Returns the fraction of neighboring regions that have been explored.
 * Used to determine how aggressively to backtrack on collision.
 * 
 * Example in 2D grid (8 neighbors):
 *   X X X
 *   X R O    R = current region, X = explored, O = unexplored
 *   O X X
 *   
 *   Result: 6/8 = 0.75 (75% of neighbors explored)
 */
__device__ float get_neighborhood_exploration(
    int region_idx,
    bool* exploredRegions
)
{
    // Get indices of all neighboring regions
    int neighbors[27];  // Max 26 neighbors in 3D (3^3 - 1) + self
    int num_neighbors = 0;
    
    get_neighbor_indices(region_idx, neighbors, &num_neighbors);
    
    // Count how many are explored
    int explored_count = 0;
    for(int i = 0; i < num_neighbors; i++)
    {
        if(neighbors[i] >= 0 && exploredRegions[neighbors[i]])
        {
            explored_count++;
        }
    }
    
    // Return fraction explored
    if(num_neighbors == 0) return 0.0f;
    return (float)explored_count / (float)num_neighbors;
}

/*
 * Get indices of neighboring regions in the grid
 * 
 * For simplicity, this implementation assumes a regular grid where regions
 * are organized in a multi-dimensional array. Neighbors are all adjacent cells.
 * 
 * Note: This needs to be customized based on your actual grid structure.
 * The implementation below is for a 3D grid, but can be adapted.
 */
__device__ void get_neighbor_indices(
    int region_idx,
    int* neighbors,
    int* num_neighbors
)
{
    // Convert linear index to 3D coordinates
    int coords[W_DIM];
    linear_to_multi_index(region_idx, coords);
    
    // Check all 3^W_DIM neighbors (including self)
    int neighbor_count = 0;
    
    // For each dimension, check -1, 0, +1 offset
    // This creates a 3×3 (2D) or 3×3×3 (3D) neighborhood
    
    // NOTE: This is a simplified version for W_DIM = 2 or 3
    // You may need to adapt for higher dimensions
    
    if(W_DIM == 2)
    {
        // 2D case: 8 neighbors + self
        for(int dx = -1; dx <= 1; dx++)
        {
            for(int dy = -1; dy <= 1; dy++)
            {
                if(dx == 0 && dy == 0) continue;  // Skip self
                
                int nx = coords[0] + dx;
                int ny = coords[1] + dy;
                
                // Check bounds (assuming NUM_R1_REGIONS_PER_DIM divisions per dimension)
                int divisions_per_dim = (int)powf((float)NUM_R1_REGIONS, 1.0f / W_DIM);
                
                if(nx >= 0 && nx < divisions_per_dim &&
                   ny >= 0 && ny < divisions_per_dim)
                {
                    int neighbor_coords[2] = {nx, ny};
                    neighbors[neighbor_count++] = multi_to_linear_index(neighbor_coords);
                }
            }
        }
    }
    else if(W_DIM == 3)
    {
        // 3D case: 26 neighbors
        for(int dx = -1; dx <= 1; dx++)
        {
            for(int dy = -1; dy <= 1; dy++)
            {
                for(int dz = -1; dz <= 1; dz++)
                {
                    if(dx == 0 && dy == 0 && dz == 0) continue;  // Skip self
                    
                    int nx = coords[0] + dx;
                    int ny = coords[1] + dy;
                    int nz = coords[2] + dz;
                    
                    int divisions_per_dim = (int)powf((float)NUM_R1_REGIONS, 1.0f / W_DIM);
                    
                    if(nx >= 0 && nx < divisions_per_dim &&
                       ny >= 0 && ny < divisions_per_dim &&
                       nz >= 0 && nz < divisions_per_dim)
                    {
                        int neighbor_coords[3] = {nx, ny, nz};
                        neighbors[neighbor_count++] = multi_to_linear_index(neighbor_coords);
                    }
                }
            }
        }
    }
    
    *num_neighbors = neighbor_count;
}

/*
 * Convert linear region index to multi-dimensional coordinates
 * 
 * Example for 10×10 2D grid:
 *   Region 42 → [4, 2]  (row 4, column 2)
 * 
 * Example for 10×10×10 3D grid:
 *   Region 523 → [5, 2, 3]
 */
__device__ void linear_to_multi_index(
    int linear_idx,
    int* coords
)
{
    // Assumes uniform grid with equal divisions per dimension
    int divisions_per_dim = (int)powf((float)NUM_R1_REGIONS, 1.0f / W_DIM);
    
    int temp = linear_idx;
    for(int d = W_DIM - 1; d >= 0; d--)
    {
        coords[d] = temp % divisions_per_dim;
        temp /= divisions_per_dim;
    }
}

/*
 * Convert multi-dimensional coordinates to linear region index
 * 
 * Inverse of linear_to_multi_index
 */
__device__ int multi_to_linear_index(
    int* coords
)
{
    int divisions_per_dim = (int)powf((float)NUM_R1_REGIONS, 1.0f / W_DIM);
    
    int linear = 0;
    int factor = 1;
    
    for(int d = W_DIM - 1; d >= 0; d--)
    {
        linear += coords[d] * factor;
        factor *= divisions_per_dim;
    }
    
    return linear;
}

/*
 * CSV writing utility (unchanged from KPAX)
 */
void ReKino::writeExecutionTimeToCSV(double time)
{
    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/ExecutionTime");
    filename.str("");
    filename << "Data/ExecutionTime/rekino_execution_time.csv";
    
    std::ofstream file(filename.str(), std::ios::app);
    if(file.is_open())
    {
        file << time << "\n";
        file.close();
    }
}
