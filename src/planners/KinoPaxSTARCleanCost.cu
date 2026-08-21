// KinoPaxSTARCleanCost -- KinoPaxSTARWeightedCost folded down to ONE acceptance decision.
//
// WHAT WAS WRONG. KinoPaxSTARWeightedCost ran two acceptance decisions that composed as an AND for
// every non-best candidate:
//   1. propagate time: isBest || (rand < min(vertexScore, h_acceptCap_)) || !activeSubVertices[sub]
//   2. gate time:      weightedAccept(w, vertexScore + fAccept, costProbExp, floor)
// so the effective admission probability was [<= h_acceptCap_, or fresh-subregion] * weightedAccept.
// Three consequences: h_acceptCap_ (0.1) sat silently upstream of w, so w was never the single knob
// it was documented as; the R2 seeding "free pass" was not free, since a fresh-sub-region node still
// had to survive the weighted roll; and the tree grew ~10x slower than the weighted rule alone
// implies.
//
// WHY THE DECISION BELONGS AT THE GATE, NOT IN PROPAGATE. Two things are invalid inside the
// propagate kernel and become valid at the next kernel boundary:
//   - Region cost statistics are MID-FLIGHT. minCostsR1 / sumCostsR1 / cntCostsR1 are being updated
//     by atomics from the very threads that would read them, so costProbExp computed there would
//     use a partial mean over whichever threads happened to land first -- two identical-cost nodes
//     in the same region would draw different P_cost purely from scheduling. (This is tolerable for
//     isBest, where being wrong only ever keeps an extra node; it is not tolerable for the
//     distribution the weighted rule is trying to impose.)
//   - vertexScores are one iteration STALE in propagate: graph_.updateVertices() runs between
//     propagateFrontier and updateFrontier, so only the gate sees scores that include this
//     iteration's samples.
// fAccept is a third: it is computed from h_frontierNextSize_, which does not exist until propagate
// has finished.
//
// SO: propagate is a pure candidate producer (mark every collision-free sample, record its cost,
// region and sub-region freshness, draw no random numbers), and the gate -- renamed _accept_kernel,
// because nothing is in the tree yet and it was never a prune -- makes the single decision:
//
//     P_combined = cap * min(1, w*P_syclop + (1-w)*P_cost + P_floor)
//
// applied identically at the gate and at Part-B reactivation, with P_syclop = vertexScore + fAccept
// (the full KPAX rule). Region-best candidates are exempt, and so are candidates that claimed a
// virgin R2 sub-region -- seeding is now an actual free pass. Knobs: w (1 = KPAX's acceptance,
// 0 = pure cost-greedy), k (P_cost decay), cap in (0,1] (flat throttle on the final probability,
// replacing h_acceptCap_).
//
// P_cost is costProbExp (helper.cuh), not costKeepProb: exp(-k*(cost-m)/(mean-m)) is exactly 1 at
// the region min AND has a real gradient across the whole range, where min(1,(mean/cost)^k) is
// pinned at 1 for every cost at or below the mean. It carries no floor -- P_floor is added once,
// in weightedAccept().
//
// Carries NO retroactive pruning: see KinoPaxSTARTrueWeightedCost for a cost-guarded version.
#include "planners/KinoPaxSTARCleanCost.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"

KinoPaxSTARCleanCost::KinoPaxSTARCleanCost()
{
    graph_ = Graph(W_SIZE);

    // KPAX exploration vectors
    d_frontier_                    = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_frontierNext_                = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_activeFrontierIdxs_          = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_activeFrontierRepeatIdxs_    = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_unexploredSamples_           = thrust::device_vector<float>(MAX_TREE_SIZE * SAMPLE_DIM);
    d_unexploredSamplesParentIdxs_ = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_frontierScanIdx_             = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_frontierRepeatScanIdx_       = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_goalSample_                  = thrust::device_vector<float>(SAMPLE_DIM);
    d_activeFrontierRepeatCount_   = thrust::device_vector<uint>(MAX_TREE_SIZE);

    // KinoPaxPlus optimization vectors
    d_minCostsR1_             = thrust::device_vector<float>(NUM_R1_REGIONS);
    d_sumCostsR1_             = thrust::device_vector<float>(NUM_R1_REGIONS);
    d_cntCostsR1_             = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_bestNodeIdxPerR1_       = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_treeXR1s_               = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_frontierNextXR1s_       = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_unexploredSampleCosts_  = thrust::device_vector<float>(MAX_TREE_SIZE);
    d_goalSet_                = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_frontierNextFresh_      = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_goalSetIdxs_            = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_goalSetScanIdx_         = thrust::device_vector<uint>(MAX_TREE_SIZE);
    d_iterations_             = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_pathCosts_              = thrust::device_vector<float>(MAX_TREE_SIZE * 3);
    d_controlPathsToGoal_     = thrust::device_vector<float>(MAX_ITER * SAMPLE_DIM);

    // Raw pointers
    d_frontier_ptr_                    = thrust::raw_pointer_cast(d_frontier_.data());
    d_frontierNext_ptr_                = thrust::raw_pointer_cast(d_frontierNext_.data());
    d_activeFrontierIdxs_ptr_          = thrust::raw_pointer_cast(d_activeFrontierIdxs_.data());
    d_activeFrontierRepeatIdxs_ptr_    = thrust::raw_pointer_cast(d_activeFrontierRepeatIdxs_.data());
    d_unexploredSamples_ptr_           = thrust::raw_pointer_cast(d_unexploredSamples_.data());
    d_unexploredSamplesParentIdxs_ptr_ = thrust::raw_pointer_cast(d_unexploredSamplesParentIdxs_.data());
    d_frontierScanIdx_ptr_             = thrust::raw_pointer_cast(d_frontierScanIdx_.data());
    d_frontierRepeatScanIdx_ptr_       = thrust::raw_pointer_cast(d_frontierRepeatScanIdx_.data());
    d_goalSample_ptr_                  = thrust::raw_pointer_cast(d_goalSample_.data());
    d_activeFrontierRepeatCount_ptr_   = thrust::raw_pointer_cast(d_activeFrontierRepeatCount_.data());

    d_minCostsR1_ptr_             = thrust::raw_pointer_cast(d_minCostsR1_.data());
    d_sumCostsR1_ptr_             = thrust::raw_pointer_cast(d_sumCostsR1_.data());
    d_cntCostsR1_ptr_             = thrust::raw_pointer_cast(d_cntCostsR1_.data());
    d_bestNodeIdxPerR1_ptr_       = thrust::raw_pointer_cast(d_bestNodeIdxPerR1_.data());
    d_treeXR1s_ptr_               = thrust::raw_pointer_cast(d_treeXR1s_.data());
    d_frontierNextXR1s_ptr_       = thrust::raw_pointer_cast(d_frontierNextXR1s_.data());
    d_unexploredSampleCosts_ptr_  = thrust::raw_pointer_cast(d_unexploredSampleCosts_.data());
    d_goalSet_ptr_                = thrust::raw_pointer_cast(d_goalSet_.data());
    d_goalSetIdxs_ptr_            = thrust::raw_pointer_cast(d_goalSetIdxs_.data());
    d_goalSetScanIdx_ptr_         = thrust::raw_pointer_cast(d_goalSetScanIdx_.data());
    d_frontierNextFresh_ptr_      = thrust::raw_pointer_cast(d_frontierNextFresh_.data());
    d_iterations_ptr_             = thrust::raw_pointer_cast(d_iterations_.data());
    d_pathCosts_ptr_              = thrust::raw_pointer_cast(d_pathCosts_.data());
    d_controlPathsToGoal_ptr_     = thrust::raw_pointer_cast(d_controlPathsToGoal_.data());

    cudaMalloc(&d_minCost_ptr_, sizeof(float));

    // Spatial hash grid for fast collision detection
    d_spatialHashGrid_ = createSpatialHashGrid();

    // Host buffer for the best (min-cost) reconstructed trajectory
    h_controlPathsToGoal_ = new float[MAX_ITER * SAMPLE_DIM];

    // CleanCost tunables. P_floor defaults to EPSILON, matching the additive floor already
    // baked into the Syclop score itself (Graph.cu: vertexScores = EPSILON + score/total).
    h_costWeight_   = 0.5f;
    h_costPruneExp_ = 1.0f;
    h_probFloor_    = EPSILON;
    // cap = 1.0 means the weighted rule alone decides. Because the propagate-time filter is gone,
    // this planner admits far more per iteration than KinoPaxSTARWeightedCost at the same w --
    // cap is the knob that buys that throttle back, explicitly and downstream of w rather than
    // upstream of it. Note it multiplies P_floor too, so the effective floor is EPSILON*cap.
    h_acceptCapMul_ = 1.0f;

    h_activeBlockSize_ = 32;

    if(VERBOSE)
        {
            printf("/* Planner Type: KinoPaxSTARCleanCost (Hybrid) */\n");
            printf("/* Number of R1 Vertices: %d */\n", NUM_R1_REGIONS);
            printf("/* Number of R2 Vertices: %d */\n", NUM_R2_REGIONS);
            printf("/***************************/\n");
        }
}

KinoPaxSTARCleanCost::~KinoPaxSTARCleanCost()
{
    destroySpatialHashGrid(d_spatialHashGrid_);
    delete[] h_controlPathsToGoal_;
}

void KinoPaxSTARCleanCost::resetPlanner(float* h_initial, float* h_goal)
{
    // KPAX exploration state
    thrust::fill(d_frontier_.begin(), d_frontier_.end(), false);
    thrust::fill(d_frontierNext_.begin(), d_frontierNext_.end(), false);
    thrust::fill(d_activeFrontierIdxs_.begin(), d_activeFrontierIdxs_.end(), 0);
    thrust::fill(d_unexploredSamples_.begin(), d_unexploredSamples_.end(), 0.0f);
    thrust::fill(d_unexploredSamplesParentIdxs_.begin(), d_unexploredSamplesParentIdxs_.end(), -1);
    thrust::fill(d_frontierScanIdx_.begin(), d_frontierScanIdx_.end(), 0);
    thrust::fill(d_frontierRepeatScanIdx_.begin(), d_frontierRepeatScanIdx_.end(), 0);
    thrust::fill(d_goalSample_.begin(), d_goalSample_.end(), 0.0f);
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.begin() + 1, 5);

    // Graph state
    thrust::fill(graph_.d_activeSubVertices_.begin(), graph_.d_activeSubVertices_.end(), false);
    thrust::fill(graph_.d_vertexScoreArray_.begin(), graph_.d_vertexScoreArray_.end(), 0.0f);
    thrust::fill(graph_.d_counterArray_.begin(), graph_.d_counterArray_.end(), 0);
    thrust::fill(graph_.d_validCounterArray_.begin(), graph_.d_validCounterArray_.end(), 0);

    // Tree state
    thrust::fill(d_treeSamples_.begin(), d_treeSamples_.end(), 0.0f);
    thrust::fill(d_treeSamplesParentIdxs_.begin(), d_treeSamplesParentIdxs_.end(), -1);
    thrust::fill(d_treeSampleCosts_.begin(), d_treeSampleCosts_.end(), 0.0f);
    thrust::fill(d_frontier_.begin(), d_frontier_.begin() + 1, true);

    // KinoPaxPlus optimization state
    thrust::fill(d_minCostsR1_.begin(), d_minCostsR1_.end(), MAX_FLOAT);
    thrust::fill(d_sumCostsR1_.begin(), d_sumCostsR1_.end(), 0.0f);
    thrust::fill(d_cntCostsR1_.begin(), d_cntCostsR1_.end(), 0);
    thrust::fill(d_bestNodeIdxPerR1_.begin(), d_bestNodeIdxPerR1_.end(), -1);
    thrust::fill(d_treeXR1s_.begin(), d_treeXR1s_.end(), 0);
    thrust::fill(d_frontierNextXR1s_.begin(), d_frontierNextXR1s_.end(), 0);
    thrust::fill(d_unexploredSampleCosts_.begin(), d_unexploredSampleCosts_.end(), 0.0f);
    thrust::fill(d_goalSet_.begin(), d_goalSet_.end(), false);
    thrust::fill(d_frontierNextFresh_.begin(), d_frontierNextFresh_.end(), false);
    thrust::fill(d_iterations_.begin(), d_iterations_.end(), 0);
    thrust::fill(d_pathCosts_.begin(), d_pathCosts_.end(), 0.0f);
    thrust::fill(d_controlPathsToGoal_.begin(), d_controlPathsToGoal_.end(), 0.0f);

    h_treeSize_     = 1;
    h_itr_          = 0;
    h_costToGoal_   = 0;
    h_pathToGoal_   = 0;
    h_frontierSize_ = 0;
    h_minCost_      = MAX_FLOAT;
    h_solSetSize_   = 0;
    // Must be nonzero before iteration 1: the plan/benchmark loop breaks on
    // h_propIterations_ == 0, and propagateFrontier only assigns it on the tree-full path.
    // Adding the cost-stat members shifted the object layout and flipped this uninitialized
    // value to 0 (the sibling STAR planners survive only by lucky layout).
    h_propIterations_ = 1;

    cudaMemcpy(d_treeSamples_ptr_, h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costToGoal_ptr_, &h_costToGoal_, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pathToGoal_ptr_, &h_pathToGoal_, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minCost_ptr_, &h_minCost_, sizeof(float), cudaMemcpyHostToDevice);

    initializeRandomSeeds(static_cast<unsigned int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
}

void KinoPaxSTARCleanCost::plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    resetPlanner(h_initial, h_goal);

    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            graph_.updateVertices();
            updateFrontier();

            // Run to MAX_ITER / tree-full, continuing to improve minCost (no first-solution break).
            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "KinoPaxSTARCleanCost execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void KinoPaxSTARCleanCost::planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr)
{
    double t_start = std::clock();
    resetPlanner(h_initial, h_goal);

    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            printf("Iteration: %d, Tree Size: %d, Frontier Size: %d\n", h_itr_, h_treeSize_, h_frontierSize_);
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            graph_.updateVertices();
            updateFrontier();

            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    double executionTime = (std::clock() - t_start) / (double)CLOCKS_PER_SEC;
    std::cout << "KinoPaxSTARCleanCost execution time: " << executionTime << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
}

float KinoPaxSTARCleanCost::planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    resetPlanner(h_initial, h_goal);

    // Run to MAX_ITER / tree-full, continuing to improve minCost after the first solution.
    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            graph_.updateVertices();
            updateFrontier();
            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "KinoPaxSTARCleanCost execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return h_minCost_;
}

void KinoPaxSTARCleanCost::propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount)
{
    // --- Build spatial hash grid for fast collision detection ---
    updateSpatialHashGrid(d_spatialHashGrid_, d_obstacles_ptr, h_obstaclesCount);
    cudaMemcpy(&h_spatialHashGrid_, d_spatialHashGrid_, sizeof(SpatialHashGrid), cudaMemcpyDeviceToHost);

    // --- Find indices and size of frontier ---
    thrust::exclusive_scan(d_frontier_.begin(), d_frontier_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontier_[MAX_TREE_SIZE - 1]) ? ++h_frontierSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontier_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Build frontier repeat vector ---
    // Safety net: any position repeatInd does not write must not expose a stale index from an
    // earlier iteration/cycle. Seeding with 0 (the root) makes a missed slot degrade to a
    // redundant root expansion instead of fathering nodes from uninitialised tree slots. With a
    // consistent repeat count this fill is a no-op, since [0, h_frontierRepeatSize_) is fully written.
    thrust::fill(d_activeFrontierRepeatIdxs_.begin(), d_activeFrontierRepeatIdxs_.end(), 0);
    thrust::exclusive_scan(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), d_frontierRepeatScanIdx_.begin(), 0,
                           thrust::plus<uint>());
    repeatInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_activeFrontierIdxs_ptr_, d_activeFrontierRepeatCount_ptr_,
                                             d_frontierRepeatScanIdx_ptr_, d_activeFrontierRepeatIdxs_ptr_);
    h_frontierRepeatSize_ = d_frontierRepeatScanIdx_[MAX_TREE_SIZE - 1];
    (d_activeFrontierRepeatCount_[MAX_TREE_SIZE - 1]) ? h_frontierRepeatSize_ += d_activeFrontierRepeatCount_[MAX_TREE_SIZE - 1] : 0;

    // Cap the expanded frontier to the tree buffer: h_frontierRepeatSize_ (sum of the x15/x1
    // repeat weights) is otherwise unbounded and would overrun the MAX_TREE_SIZE-length repeat
    // index buffer and propagate grid near a full tree.
    if(h_frontierRepeatSize_ > (uint)MAX_TREE_SIZE) h_frontierRepeatSize_ = MAX_TREE_SIZE;

    if(h_frontierRepeatSize_ * h_activeBlockSize_ > (MAX_TREE_SIZE - h_treeSize_))
        {
            h_propIterations_ = std::min(int(float(MAX_TREE_SIZE - h_treeSize_) / float(h_frontierRepeatSize_)), int(h_activeBlockSize_));

            if(h_propIterations_ == 0)
                {
                    h_propIterations_   = 1;
                    h_frontierNextSize_ = MAX_TREE_SIZE - h_treeSize_;
                    thrust::fill(d_frontierNext_.begin(), d_frontierNext_.end(), false);
                }

            KinoPaxSTARCleanCost_propagateFrontier_kernel2<<<iDivUp(h_propIterations_ * h_frontierRepeatSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              h_propIterations_, graph_.d_minValueInRegion_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
              d_frontierNextXR1s_ptr_, d_frontierNextFresh_ptr_, d_unexploredSampleCosts_ptr_, h_spatialHashGrid_);
        }
    else
        {
            KinoPaxSTARCleanCost_propagateFrontier_kernel1<<<iDivUp(h_frontierRepeatSize_ * h_activeBlockSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              d_frontierNext_ptr_, graph_.d_counterArray_ptr_, graph_.d_validCounterArray_ptr_,
              graph_.d_minValueInRegion_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
              d_frontierNextXR1s_ptr_, d_frontierNextFresh_ptr_, d_unexploredSampleCosts_ptr_, h_spatialHashGrid_);
        }
}

/***************************/
/* PROPAGATE FRONTIER KERNEL 1 */
/***************************/
// One Block Per Frontier Sample — CANDIDATE PRODUCER ONLY. No acceptance decision, no RNG draw:
// every collision-free sample is marked and its cost / region / sub-region freshness recorded, and
// the accept kernel decides once the region statistics and vertex scores have converged.
__global__ void KinoPaxSTARCleanCost_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, bool* frontierNextFresh,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid)
{
    if(blockIdx.x >= frontierSize) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= MAX_TREE_SIZE) return;

    // --- Load Frontier Sample Idx and cost into shared memory ---
    __shared__ int s_x0Idx;
    __shared__ float s_x0Cost;
    if(threadIdx.x == 0)
        {
            s_x0Idx           = activeFrontierIdxs[blockIdx.x];
            s_x0Cost          = treeSampleCosts[s_x0Idx];
            frontier[s_x0Idx] = false;
        }
    __syncthreads();

    // --- Load Frontier Sample into shared memory ---
    __shared__ float s_x0[SAMPLE_DIM];
    if(threadIdx.x < SAMPLE_DIM) s_x0[threadIdx.x] = treeSamples[s_x0Idx * SAMPLE_DIM + threadIdx.x];
    __syncthreads();

    // --- Propagate Sample ---
    float* x1                        = &unexploredSamples[tid * SAMPLE_DIM];
    unexploredSamplesParentIdxs[tid] = s_x0Idx;
    curandState randSeed             = randomSeeds[tid];
    bool valid                       = propagateAndCheckSpatialHash(s_x0, x1, &randSeed, spatialHashGrid, obstacles, obstaclesCount);
    int x1Vertex                     = getRegion(x1);
    int x1SubVertex                  = getSubRegion(x1, x1Vertex, minValueInRegion);

    // --- Update Graph statistics ---
    atomicAdd(&vertexCounter[x1Vertex], 1);
    if(valid)
        {
            atomicAdd(&validVertexCounter[x1Vertex], 1);

            // Cumulative cost from root
            float cost = s_x0Cost + edgeCost(s_x0, x1);

            // --- Region cost statistics. These are what the accept kernel reads once the launch
            // has finished; reading them HERE would see them mid-flight. ---
            if(minCostsR1[x1Vertex] > cost) atomicMinFloat(&minCostsR1[x1Vertex], cost);
            atomicAdd(&sumCostsR1[x1Vertex], cost);
            atomicAdd(&cntCostsR1[x1Vertex], 1);

            // --- R2 seeding: read-then-set, so an entire launch landing in one virgin sub-region
            // all record the pass. That is KPAX's behaviour and is deliberately preserved; using
            // atomicExch's return value instead would grant it to exactly one thread. ---
            bool wasFresh = (activeSubVertices[x1SubVertex] == 0);
            if(wasFresh) atomicExch(&activeSubVertices[x1SubVertex], 1);

            // --- Record the candidate. No acceptance decision, no RNG draw. ---
            unexploredSampleCosts[tid] = cost;
            frontierNextXR1s[tid]      = x1Vertex;
            frontierNextFresh[tid]     = wasFresh;
            frontierNext[tid]          = true;
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* PROPAGATE FRONTIER KERNEL 2 */
/***************************/
// Iterations mode — CANDIDATE PRODUCER ONLY (see kernel 1).
__global__ void KinoPaxSTARCleanCost_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, int iterations,
                                                   float* minValueInRegion,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, bool* frontierNextFresh,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid)
{
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= MAX_TREE_SIZE) return;
    frontier[tid] = false;
    if(tid >= frontierSize * iterations) return;

    int activeFrontierIdx = tid / iterations;
    int x0Idx             = activeFrontierIdxs[activeFrontierIdx];
    float x0Cost          = treeSampleCosts[x0Idx];

    // --- Load Frontier Sample ---
    float* x0 = &treeSamples[x0Idx * SAMPLE_DIM];

    // --- Propagate Sample ---
    float* x1                        = &unexploredSamples[tid * SAMPLE_DIM];
    unexploredSamplesParentIdxs[tid] = x0Idx;
    curandState randSeed             = randomSeeds[tid];
    bool valid                       = propagateAndCheckSpatialHash(x0, x1, &randSeed, spatialHashGrid, obstacles, obstaclesCount);
    int x1Vertex                     = getRegion(x1);
    int x1SubVertex                  = getSubRegion(x1, x1Vertex, minValueInRegion);

    // --- Update Graph statistics ---
    atomicAdd(&vertexCounter[x1Vertex], 1);
    if(valid)
        {
            atomicAdd(&validVertexCounter[x1Vertex], 1);

            float cost = x0Cost + edgeCost(x0, x1);

            // --- Region cost statistics (see kernel 1). ---
            if(minCostsR1[x1Vertex] > cost) atomicMinFloat(&minCostsR1[x1Vertex], cost);
            atomicAdd(&sumCostsR1[x1Vertex], cost);
            atomicAdd(&cntCostsR1[x1Vertex], 1);

            // --- R2 seeding: read-then-set, KPAX semantics (see kernel 1). ---
            bool wasFresh = (activeSubVertices[x1SubVertex] == 0);
            if(wasFresh) atomicExch(&activeSubVertices[x1SubVertex], 1);

            // --- Record the candidate. No acceptance decision, no RNG draw. ---
            unexploredSampleCosts[tid] = cost;
            frontierNextXR1s[tid]      = x1Vertex;
            frontierNextFresh[tid]     = wasFresh;
            frontierNext[tid]          = true;
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* ACCEPT KERNEL — the ONLY acceptance decision */
/***************************/
// Runs over EVERY collision-free candidate the propagate kernels produced. This is the first point
// in the iteration where both halves of the weighted sum are valid: the propagate launch has
// finished, so the region statistics are converged rather than mid-flight, and
// graph_.updateVertices() has run, so vertexScores include this iteration's samples.
//
// Two exemptions, then the single rule:
//     P = cap * min(1, w*(vertexScore + fAccept) + (1-w)*costProbExp(...) + P_floor)
__global__ void KinoPaxSTARCleanCost_accept_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                                  float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                  int* frontierNextXR1s, bool* frontierNextFresh,
                                                  float* unexploredSampleCosts,
                                                  bool* frontierNext, curandState* randomSeeds,
                                                  float* vertexScores, float fAccept,
                                                  float costWeight, float costPruneExp, float probFloor,
                                                  float acceptCapMul)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierNextSize) return;

    int idx    = activeFrontierNextIdxs[tid];
    float cost = unexploredSampleCosts[idx];
    int xR1    = frontierNextXR1s[idx];
    float m    = minCostsR1[xR1];

    // --- Exemption 1: min-cost candidates are always inserted (every region best stays in the
    // frontier). LOAD-BEARING, and must not be folded into the formula: at cost == m, P_cost == 1
    // but P_combined == cap*min(1, w*P_syclop + (1-w) + floor), which is below 1 whenever
    // P_syclop < 1 or cap < 1 -- so a region best could be rejected. ---
    if(cost <= m) return;

    // --- Exemption 2: this thread claimed a virgin R2 sub-region. In KinoPaxSTARWeightedCost the
    // seeding pass only cleared the propagate-time filter and then still faced the weighted roll,
    // so it was never actually free; here it is. The flag has to come from propagate because by
    // now every sub-region touched this iteration is already marked active. ---
    if(frontierNextFresh[idx]) return;

    // --- Everything else: weighted sum of the Syclop and cost probabilities, throttled by cap. ---
    float pCost   = costProbExp(m, sumCostsR1[xR1], cntCostsR1[xR1], cost, costPruneExp);
    float pSyclop = vertexScores[xR1] + fAccept;
    float acceptanceProbability = acceptCapMul * weightedAccept(costWeight, pSyclop, pCost, probFloor);

    curandState seed = randomSeeds[idx];
    bool accept      = curand_uniform(&seed) < acceptanceProbability;
    randomSeeds[idx] = seed;

    if(!accept) frontierNext[idx] = false;
}

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// Hybrid: adds new frontier nodes to tree, re-activates best-per-region + probabilistic others
__global__ void
KinoPaxSTARCleanCost_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               uint* activeFrontierRepeatCount, int* validVertexCounter, curandState* randomSeeds,
                               float* vertexScores, float fAccept,
                               float costWeight, float costPruneExp, float probFloor, float acceptCapMul,
                               float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                               int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
                               float* minCost, float* unexploredSampleCosts, bool* goalSet,
                               int* iterations, int iteration)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_xGoal[SAMPLE_DIM];
    if(threadIdx.x < SAMPLE_DIM) s_xGoal[threadIdx.x] = xGoal[threadIdx.x];
    __syncthreads();

    // --- Part A: Add new frontier nodes to tree ---
    if(tid < frontierNextSize)
        {
            int x1TreeIdx       = treeSize + tid;
            int x1UnexploredIdx = activeFrontierNextIdxs[tid];
            frontierNext[activeFrontierNextIdxs[tid]] = false;

            float* x1   = &unexploredSamples[x1UnexploredIdx * SAMPLE_DIM];
            int x0Idx   = unexploredSamplesParentIdxs[x1UnexploredIdx];
            float cost   = unexploredSampleCosts[x1UnexploredIdx];
            int xR1      = frontierNextXR1s[x1UnexploredIdx];

            // Transfer to tree
            treeSamplesParentIdxs[x1TreeIdx] = x0Idx;
            for(int i = 0; i < SAMPLE_DIM; i++)
                treeSamples[x1TreeIdx * SAMPLE_DIM + i] = x1[i];
            treeSampleCosts[x1TreeIdx] = cost;
            treeXR1s[x1TreeIdx]        = xR1;

            // Always add to frontier (passed dual filter + pruning)
            frontier[x1TreeIdx] = true;

            // Repeat count based on region exploration (KPAX logic)
            if(validVertexCounter[xR1] < 10)
                activeFrontierRepeatCount[x1TreeIdx] = 15;
            else
                activeFrontierRepeatCount[x1TreeIdx] = 1;

            // Update best-node index if this is the new region best
            if(cost <= minCostsR1[xR1])
                atomicExch(&bestNodeIdxPerR1[xR1], x1TreeIdx);

            // Goal criteria check — accumulate goal nodes into goalSet; the min-cost
            // path is reconstructed afterwards by getControlPathToGoal.
            if(distance(x1, s_xGoal) < GOAL_THRESH && cost <= *minCost)
                {
                    atomicMinFloat(minCost, cost);
                    goalSet[x1TreeIdx]    = true;
                    frontier[x1TreeIdx]   = false;
                    // The repeat count MUST be cleared with the frontier flag. repeatInd expands
                    // activeFrontierRepeatCount over the compacted frontier list, which findInd
                    // builds from frontier==true only. A node left with count>0 but frontier==false
                    // owns a slice of d_activeFrontierRepeatIdxs_ that no thread ever writes, yet
                    // h_frontierRepeatSize_ still spans it -- so propagateFrontier reads whatever
                    // stale tree indices that slice held (from a prior, much larger tree) and
                    // expands from not-yet-created nodes whose cost still reads 0. That produced
                    // phantom-parented, artificially cheap nodes that then won minCost.
                    activeFrontierRepeatCount[x1TreeIdx] = 0;
                    iterations[x1TreeIdx] = iteration;
                }
        }

    // --- Part B: Re-activate existing tree nodes ---
    else if(tid < frontierNextSize + treeSize)
        {
            int treeIdx = tid - frontierNextSize;
            if(goalSet[treeIdx]) return;

            int xR1 = treeXR1s[treeIdx];

            // GUARANTEE: Best node per region is ALWAYS in the frontier -- unconditionally, with
            // no dice roll and no cap. This is KinoPaxPlus's invariant and the reason cap can be
            // driven arbitrarily low without stalling cost improvement.
            if(treeIdx == bestNodeIdxPerR1[xR1])
                {
                    frontier[treeIdx]                  = true;
                    activeFrontierRepeatCount[treeIdx] = 1;
                    return;
                }

            // REACTIVATION: weighted sum of the Syclop and cost probabilities, throttled by cap --
            // the SAME rule the accept kernel uses, so the planner has exactly one acceptance
            // formula. At w = 1, cap = 1 this is min(1, vertexScore + fAccept + P_floor), i.e.
            // KPAX's rule, which is the control arm for the exploration-speed question.
            //
            // (KinoPaxSTARWeightedCost tested a pruned[] tombstone here. This variant runs no
            // retroactive pruning, so nothing ever set it; the array and the branch are gone.)
            if(frontier[treeIdx] == 0)
                {
                    float pCost   = costProbExp(minCostsR1[xR1], sumCostsR1[xR1], cntCostsR1[xR1],
                                                treeSampleCosts[treeIdx], costPruneExp);
                    float pSyclop = vertexScores[xR1] + fAccept;
                    float reactivationProb = acceptCapMul * weightedAccept(costWeight, pSyclop, pCost, probFloor);

                    curandState seed = randomSeeds[treeIdx];
                    if(curand_uniform(&seed) < reactivationProb)
                        {
                            frontier[treeIdx]                  = true;
                            activeFrontierRepeatCount[treeIdx] = 1;
                        }
                    randomSeeds[treeIdx] = seed;
                }
        }
}

void KinoPaxSTARCleanCost::updateFrontier()
{
    // --- Find indices and size of the next frontier ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Compute fAccept (KPAX re-activation boost) ---
    // Must precede the accept kernel: the weighted rule needs P_syclop = vertexScore + fAccept at
    // both acceptance points and the gate is the first of them. treeAddSize is therefore based on
    // the PRE-acceptance candidate count, which here is EVERY collision-free sample rather than a
    // pre-filtered subset -- so fAccept runs smaller than in KinoPaxSTARWeightedCost. One value for
    // the whole iteration is also more consistent than recomputing between the two uses.
    float treeAddSize = 1 - (float(h_treeSize_ + h_frontierNextSize_) / (MAX_TREE_SIZE));
    h_fAccept_        = (h_itr_ * EPSILON) * pow(treeAddSize, 5);

    // --- THE acceptance decision: region-best and fresh-sub-region candidates exempt, everything
    // else kept with cap * weightedAccept(...). ---
    KinoPaxSTARCleanCost_accept_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
      d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
      d_frontierNextXR1s_ptr_, d_frontierNextFresh_ptr_,
      d_unexploredSampleCosts_ptr_, d_frontierNext_ptr_, d_randomSeeds_ptr_,
      graph_.d_vertexScoreArray_ptr_, h_fAccept_,
      h_costWeight_, h_costPruneExp_, h_probFloor_, h_acceptCapMul_);

    // --- Re-scan after the accept kernel ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Check tree capacity ---
    if(h_treeSize_ + h_frontierNextSize_ >= MAX_TREE_SIZE)
        {
            h_propIterations_ = 0;
            return;
        }

    // --- Update Frontier (fAccept already computed above, before the gate) ---
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    KinoPaxSTARCleanCost_updateFrontier_kernel<<<iDivUp(h_frontierNextSize_ + h_treeSize_, h_blockSize_), h_blockSize_>>>(
      d_frontier_ptr_, d_frontierNext_ptr_, d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_goalSample_ptr_, h_treeSize_,
      d_unexploredSamples_ptr_, d_treeSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, d_activeFrontierRepeatCount_ptr_, graph_.d_validCounterArray_ptr_, d_randomSeeds_ptr_,
      graph_.d_vertexScoreArray_ptr_, h_fAccept_,
      h_costWeight_, h_costPruneExp_, h_probFloor_, h_acceptCapMul_,
      d_minCostsR1_ptr_, d_sumCostsR1_ptr_, d_cntCostsR1_ptr_,
      d_treeXR1s_ptr_, d_frontierNextXR1s_ptr_, d_bestNodeIdxPerR1_ptr_,
      d_minCost_ptr_, d_unexploredSampleCosts_ptr_, d_goalSet_ptr_,
      d_iterations_ptr_, h_itr_);

    // --- Sync goal state ---
    cudaMemcpy(&h_minCost_, d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);

    // --- Update Tree Size ---
    h_treeSize_ += h_frontierNextSize_;
}

/***************************/
/* GET CONTROL PATH TO GOAL */
/***************************/
void KinoPaxSTARCleanCost::getControlPathToGoal()
{
    thrust::exclusive_scan(d_goalSet_.begin(), d_goalSet_.end(), d_goalSetScanIdx_.begin(), 0, thrust::plus<uint>());
    h_solSetSize_ = d_goalSetScanIdx_[MAX_TREE_SIZE - 1];
    (d_goalSet_[MAX_TREE_SIZE - 1]) ? ++h_solSetSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_goalSet_ptr_, d_goalSetScanIdx_ptr_, d_goalSetIdxs_ptr_);

    if(h_solSetSize_ == 0) return;

    KinoPaxSTARCleanCost_getControlPathToGoal_kernel<<<iDivUp(h_solSetSize_, h_blockSize_), h_blockSize_>>>(
      d_controlPathsToGoal_ptr_, d_treeSamples_ptr_, d_treeSamplesParentIdxs_ptr_, d_goalSetIdxs_ptr_, h_solSetSize_,
      d_pathCosts_ptr_, d_treeSampleCosts_ptr_, d_iterations_ptr_, d_minCost_ptr_);

    cudaMemcpy(&h_minCost_, d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_controlPathsToGoal_, d_controlPathsToGoal_ptr_, MAX_ITER * SAMPLE_DIM * sizeof(float),
               cudaMemcpyDeviceToHost);
    printf("Cost to Goal: %f\n", h_minCost_);
}

/***************************/
/* GET CONTROL PATH TO GOAL KERNEL */
/***************************/
// Every goal thread records (idx, cost, iteration); only the min-cost goal reconstructs its full path.
__global__ void KinoPaxSTARCleanCost_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
                                                     int* treeSamplesParentIdxs, uint* goalSetIdxs, int goalSetSize,
                                                     float* pathCosts, float* treeSampleCosts, int* iterations,
                                                     float* minCost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= MAX_TREE_SIZE || tid >= goalSetSize) return;

    int goalIdx = goalSetIdxs[tid];
    int x0Idx   = goalIdx;
    float cost  = treeSampleCosts[goalIdx];

    int pathCostsIdx            = 3 * tid;
    pathCosts[pathCostsIdx]     = goalIdx;
    pathCosts[pathCostsIdx + 1] = cost;
    pathCosts[pathCostsIdx + 2] = iterations[goalIdx];

    if(cost != *minCost) return;
    int i = 0;
    // controlPathsToGoal holds MAX_ITER nodes; guard so a maximal-depth path can't write
    // one node past the buffer.
    while(x0Idx != -1 && i < MAX_ITER)
        {
            for(int j = 0; j < SAMPLE_DIM; j++)
                controlPathsToGoal[SAMPLE_DIM * i + j] = treeSamples[x0Idx * SAMPLE_DIM + j];
            i++;
            x0Idx = treeSamplesParentIdxs[x0Idx];
        }
}

void KinoPaxSTARCleanCost::writeExecutionTimeToCSV(double time)
{
    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/ExecutionTime");
    filename.str("");
    filename << "Data/ExecutionTime/executionTime.csv";
    writeValueToCSV(time, filename.str());
}
