#pragma once
#include "planners/Planner.cuh"
#include "graphs/Graph.cuh"
#include "collisionCheck/spatialHash.cuh"

// Which door a candidate/node came in by. Stored per node so the CSV can answer "what built this
// tree", and consumed by the fan-out rule, which gives each door a different block count.
//
// PREFIXED, and the prefix is load-bearing. These are FILE-SCOPE names in a header, and the sweep
// pulls several planner headers into ONE translation unit -- an unprefixed copy is a redefinition
// error in exactly the one .cu that matters and nowhere else. Same discipline as the kernel
// prefixes, different mechanism: kernels collide at LINK time under CUDA_SEPARABLE_COMPILATION,
// header constants collide at COMPILE time.
static const int CS_DOOR_NONE    = 0;
static const int CS_DOOR_COST    = 1;   // region best: cost <= minCostsR1[r]
static const int CS_DOOR_EXPLORE = 2;   // claimed a virgin R2 cell and won the region's quota
static const int CS_DOOR_REACT   = 3;   // existing tree node put back in the frontier

class CountingStars : public Planner
{
public:
    /**************************** CONSTRUCTORS ****************************/
    CountingStars();
    ~CountingStars();

    /****************************    METHODS    ****************************/
    void plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount) override;
    float planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount);
    void planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr);
    void resetPlanner(float* h_initial, float* h_goal);
    void propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount);
    void updateFrontier();
    void getControlPathToGoal();
    void writeExecutionTimeToCSV(double time);

    /****************************    FIELDS    ****************************/
    // --- host fields ---
    Graph graph_;
    uint h_frontierSize_, h_frontierNextSize_, h_activeBlockSize_, h_frontierRepeatSize_, h_propIterations_, h_solSetSize_;
    float h_minCost_;

    // ==================================================================================
    // THE THREE COUNTS. This is the whole tuning surface, and the whole extension surface.
    //
    // COMBO admitted nodes with one global probability, min(pMax, shape * pTargetAccept), with
    // pTargetAccept solved so the EXPECTED admission count hit a growth target. That normalisation
    // is what this planner exists to remove. The shape is a normalised blend of sigmoids -- neutral
    // 0.5, ceiling 1.0 -- so it can multiply acceptance by at most 2x; and pTargetAccept divides by
    // the measured mean shape, so sharpening the shape lowers the mean, raises pTarget, and hands
    // the gain straight back. Acceptance was a REALLOCATION mechanism at a fixed total. It could not
    // concentrate, so nodes came out dense everywhere instead of sparse where it mattered.
    //
    // Here a count says how many nodes come in by each door. Probability survives ONLY as the way a
    // counted quota is filled inside one region -- never as a global normalised score.
    //
    // Each count ramps linearly in u = treeSize / MAX_TREE_SIZE, from its *0 value to its *1 value.
    // That ramp is the growth controller's seat: replacing these six numbers with values derived
    // from a per-iteration target is a change to ONE function, updateCounts().
    // ==================================================================================

    // Nodes per R1 region per iteration admitted for NOVELTY. A candidate is eligible only if it won
    // the atomic claim on a virgin R2 sub-region; this caps how many of those winners are taken.
    // Self-limiting even before the cap: there are only NUM_R2_PER_R1 cells in a region, ever.
    float h_exploreCount0_, h_exploreCount1_;

    // Nodes per R1 region per iteration admitted for COST. Pinned at 1 in v1, where it is exactly
    // the region-best rule the atomicMin on minCostsR1 already computes -- zero new machinery.
    // Rises FASTER than explore: optimality matters more as the tree fills. Anything above 1 needs a
    // per-region top-K, which does not exist in this repo yet; see the .cu for what that would take.
    float h_costCount0_, h_costCount1_;

    // GLOBAL cap on how many EXISTING tree nodes are put back in the frontier this iteration, on top
    // of the nodes admitted this iteration. Every non-goal tree node outside the frontier is drawn
    // with p = reactCount / treeSize, so the expected number added is exactly reactCount, whatever
    // the tree size.
    //
    // THIS IS THE MOST IMPORTANT KNOB IN THE PLANNER, because it sets F, and F sets
    // propagations-per-node. KinoPaxPlus wins by dividing the whole budget over a TINY frontier --
    // bf = MAX_TREE_SIZE/(F*32), so 40,000 propagations per node at F = 10. COMBO's frontier was
    // pinned at F >= nActive by an unconditional region-best reactivation, which put it near 32
    // propagations per node: three orders of magnitude adrift, and no fan-out weighting closes that
    // gap. Only shrinking F does. 0 is legal and interesting -- the frontier is then exactly this
    // iteration's admissions.
    float h_reactCount0_, h_reactCount1_;

    // ==================================================================================
    // FAN-OUT. Blocks are decided AT ADMISSION and stored per node; propagateFrontier only reads
    // them, so it stays the single writer of activeFrontierRepeatCount.
    //
    //   REACTIVATE  1
    //   EXPLORE     max(maxBlocks >> (ordinal / halfLife), 1)   ordinal = the node's position in its
    //                                                            region's history, cumulative
    //   COST        max(maxBlocks / (novelThisIter[r] + 1), 1)   quiet region -> optimise hard
    //
    // GEOMETRIC, NOT LINEAR, and deliberately so. A linear ramp max(maxBlocks - ordinal, 1) spends
    // 15+14+13+12+11 = 65 blocks on a region's first five nodes. KPAX's realised behaviour is far
    // sparser than its own rule reads: validVertexCounter is CUMULATIVE and gains ~32 per frontier
    // node per iteration, so a region crosses `< 10` almost immediately and the x15 is a ONE-SHOT
    // BURST, not a ramp. Halving gives 15, 7, 3, 1, 1 ... -- 27 blocks over the same five nodes,
    // with almost everything on the first two.
    //
    // halfLife is the one knob spanning both: 1 is the burst, large is near-flat.
    // ==================================================================================
    int h_maxBlocks_;
    int h_fanHalfLife_;

    // ==================================================================================
    // DERIVED PER-ITERATION SCALARS -- measured, never set by a caller, all logged so the planner
    // can be audited from the CSV rather than trusted.
    // ==================================================================================

    // This iteration's counts, after the ramp. Logged because they are the interface every later
    // extension replaces.
    float h_exploreCount_, h_costCount_, h_reactCount_;

    // Admissions by door this iteration, counted exactly on the device and copied back once.
    // h_reactivated_ is the realised Part B output, which should track h_reactCount_ in expectation
    // -- a persistent gap means the draw is not seeing the population it should.
    enum DoorSlot { CS_SLOT_EXPLORE = 0, CS_SLOT_COST, CS_SLOT_REACT, CS_NUM_DOOR_SLOTS };
    thrust::device_vector<unsigned long long> d_doorCounts_;
    unsigned long long* d_doorCounts_ptr_;
    unsigned long long  h_doorCounts_[CS_NUM_DOOR_SLOTS];
    uint h_admittedExplore_, h_admittedCost_, h_reactivated_;

    // Blocks the buffer allows, and the scale applied to make the frontier fit inside it.
    //
    //   blockCeiling = 0.8 * remaining / activeBlockSize
    //
    // The 0.8 is the margin against the kernel1 condition (frontierRepeatSize * activeBlockSize <=
    // MAX_TREE_SIZE - treeSize). h_blockScale_ < 1 means the ceiling bound and every node's boost
    // was shrunk proportionally ABOVE THE rep >= 1 FLOOR, so sum(rep) fits exactly.
    //
    // Kernel2 is still forced once activeBlockSize * F > remaining, since rep >= 1 is a correctness
    // floor -- but that is a property of F, and F is what h_reactCount_ controls.
    float h_blockCeiling_;
    float h_blockScale_;

    // Collision-free fraction, kept purely as the diagnostic window into propagation efficiency now
    // that nothing consumes it. Reduced from the graph's cumulative counter arrays.
    float h_globalCollisionFrac_;

    // Two denominators the planner already knows and would otherwise discard.
    //   h_propAttempted_     propagation attempts this iteration, INCLUDING collisions
    //   h_candidatesPreGate_ collision-free candidates the accept kernel judges, captured before the
    //                        post-gate re-scan overwrites h_frontierNextSize_. Exact on both
    //                        propagate paths, unlike any reconstruction from h_propAttempted_
    //                        (whose formula differs by branch).
    uint h_propAttempted_;
    uint h_candidatesPreGate_;

    float* h_controlPathsToGoal_;

    // --- device fields (frontier / compaction) ---
    thrust::device_vector<bool> d_frontier_, d_frontierNext_;
    thrust::device_vector<uint> d_activeFrontierIdxs_, d_frontierScanIdx_, d_activeFrontierRepeatCount_,
      d_frontierRepeatScanIdx_, d_activeFrontierRepeatIdxs_;
    thrust::device_vector<int> d_unexploredSamplesParentIdxs_;
    thrust::device_vector<float> d_unexploredSamples_, d_goalSample_;

    // --- device fields (cost / region stats) ---
    thrust::device_vector<float> d_minCostsR1_, d_sumCostsR1_;
    thrust::device_vector<int>   d_cntCostsR1_;
    thrust::device_vector<float> d_unexploredSampleCosts_;
    thrust::device_vector<int> d_bestNodeIdxPerR1_;
    thrust::device_vector<int> d_treeXR1s_, d_frontierNextXR1s_;
    thrust::device_vector<bool> d_goalSet_;

    // ==================================================================================
    // COUNTINGSTARS' OWN R1 MIN-CORNER TABLE, and the reason it exists.
    //
    // getSubRegion measures a candidate's offset from its R1 region's minimum corner to find its R2
    // sub-cell. Graph.cu's initializeRegions_kernel computes those corners with a decode that does
    // NOT invert getRegion's encode: the digit order is reversed (wRegion is the most significant
    // group in the encode, the least significant in the decode) AND the group moduli use hardcoded
    // exponents C_R1_LENGTH^2 / V_R1_LENGTH^1 where the encode uses C_R1_LENGTH^C_DIM /
    // V_R1_LENGTH^V_DIM. The collapse factor is C_R1_LENGTH^(C_DIM-2) * V_R1_LENGTH^(V_DIM-1) --
    // 8x at the checked-in config, and it GROWS with a finer discretisation.
    //
    // So the corner belongs to a different region than the index names, getSubRegion's clamps pin
    // the offset to an edge cell, and every R2 identity is wrong. Nothing fails; the signal is just
    // scrambled -- which would be fatal here, because the explore door IS the R2 novelty test.
    //
    // Graph.cu is deliberately left alone: every existing baseline was measured against it. This
    // planner carries a corrected copy and passes it wherever the others pass
    // graph_.d_minValueInRegion_. scripts/check_region_math.py proves the corrected decode is a
    // bijection and measures the shared one's collapse.
    thrust::device_vector<float> d_minCornerCS_;

    // --- per-R1, RESET EVERY ITERATION. The repo has no other per-iteration per-region array; every
    // other NUM_R1_REGIONS array is cumulative or a full recompute. ---
    //   d_novelCounts_  distinct virgin R2 cells claimed in this region this iteration. THE
    //                   DENOMINATOR of the explore door's per-region fill probability, and the
    //                   divisor of the cost door's block count.
    //   d_candCounts_   collision-free candidates in this region. Diagnostic only.
    thrust::device_vector<int> d_novelCounts_, d_candCounts_;

    // --- per-R1, CUMULATIVE. atomicAdd's RETURN VALUE gives a node its ordinal within its region,
    // which is what the geometric fan-out ramp indexes. The only other place in this repo that uses
    // an atomic's return value is spatialHash.cu's cell insertion. ---
    thrust::device_vector<int> d_regionNodeCount_;

    // --- per-node, TREE-INDEXED, written once at admission ---
    thrust::device_vector<int> d_nodeBlocks_, d_nodeDoor_;

    // --- per-candidate, UNEXPLORED-SAMPLE-SLOT indexed. Carries propagate's findings to the accept
    // kernel, and the accept kernel's verdict to Part A. NOT interchangeable with the tree-indexed
    // arrays above: propagate reclaims these slots for the next candidate batch every iteration. ---
    thrust::device_vector<bool> d_candNovel_;
    thrust::device_vector<int>  d_candDoor_;

    thrust::device_vector<uint> d_goalSetIdxs_, d_goalSetScanIdx_;
    thrust::device_vector<int> d_iterations_;
    thrust::device_vector<float> d_pathCosts_, d_controlPathsToGoal_;

    // --- raw pointers ---
    float *d_unexploredSamples_ptr_, *d_goalSample_ptr_, *d_unexploredSampleCosts_ptr_;
    float *d_minCostsR1_ptr_, *d_sumCostsR1_ptr_, *d_minCost_ptr_;
    float *d_minCornerCS_ptr_;
    int   *d_cntCostsR1_ptr_;
    int   *d_novelCounts_ptr_, *d_candCounts_ptr_, *d_regionNodeCount_ptr_;
    int   *d_nodeBlocks_ptr_, *d_nodeDoor_ptr_, *d_candDoor_ptr_;
    bool  *d_candNovel_ptr_;
    float *d_pathCosts_ptr_, *d_controlPathsToGoal_ptr_;
    bool *d_frontier_ptr_, *d_frontierNext_ptr_, *d_goalSet_ptr_;
    uint *d_activeFrontierIdxs_ptr_, *d_frontierScanIdx_ptr_, *d_activeFrontierRepeatCount_ptr_,
      *d_frontierRepeatScanIdx_ptr_, *d_activeFrontierRepeatIdxs_ptr_;
    uint *d_goalSetIdxs_ptr_, *d_goalSetScanIdx_ptr_;
    int *d_unexploredSamplesParentIdxs_ptr_, *d_treeXR1s_ptr_, *d_frontierNextXR1s_ptr_;
    int *d_bestNodeIdxPerR1_ptr_, *d_iterations_ptr_;

    // --- Spatial hash grid for collision detection ---
    SpatialHashGrid* d_spatialHashGrid_;
    SpatialHashGrid h_spatialHashGrid_;
};

/**************************** DEVICE FUNCTIONS ****************************/

/***************************/
/* R1 MIN-CORNER INITIALISATION */
/***************************/
// The exact inverse of getRegion's encode, written entirely in config macros so it stays correct at
// any discretisation. See d_minCornerCS_ above for why this exists rather than reusing Graph.cu's.
__global__ void CountingStars_initializeRegions_kernel(float* minCorner);

/***************************/
/* PROPAGATE FRONTIER KERNEL 1 */
/***************************/
// Pure candidate producer plus COUNTING. It records every collision-free propagation, claims R2
// cells exclusively, and accumulates the per-region counts the accept kernel divides by -- but makes
// no admission decision. Counting with atomics is exact and order-independent, so it does not
// violate the invariant that put the decision after propagate: that rule is about cost STATISTICS
// being mid-flight, and minCostsR1 is still only read afterwards.
__global__ void CountingStars_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minCorner,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, bool* candNovel, int* candDoor,
                                                   int* novelCounts, int* candCounts,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid);

/***************************/
/* PROPAGATE FRONTIER KERNEL 2 */
/***************************/
__global__ void CountingStars_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, int iterations,
                                                   float* minCorner,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, bool* candNovel, int* candDoor,
                                                   int* novelCounts, int* candCounts,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid);

/***************************/
/* ACCEPT KERNEL — the ONLY admission decision */
/***************************/
// Two doors, both counted. COST takes precedence, so a candidate that is both its region's best and
// a novel claimant is recorded as COST and does not consume an explore slot.
//
//   COST     cost <= minCostsR1[r]                          quota costCount, == 1 in v1
//   EXPLORE  candNovel and rand() < exploreCount / novelCounts[r]
//
// The explore fill is a PER-REGION, COUNT-CALIBRATED probability: expected admissions per region are
// exactly exploreCount, and when novelCounts <= exploreCount the probability is >= 1 and all are
// taken. That is a different object from the global normalised score this planner replaced -- it
// carries no cross-region normalisation at all.
__global__ void CountingStars_accept_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                            float* minCostsR1, int* frontierNextXR1s, float* unexploredSampleCosts,
                                            bool* frontierNext, bool* candNovel, int* candDoor,
                                            curandState* randomSeeds,
                                            int* novelCounts, float exploreCount,
                                            unsigned long long* doorCounts);

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// Part A inserts admitted candidates, stamps each with its region ordinal and its block count.
// Part B draws reactivations uniformly over the tree at p = reactCount/treeSize.
//
// EVERY BRANCH THAT SETS frontier[i] = true MUST WRITE nodeBlocks[i]. A missed one leaves the node
// carrying whatever block count the previous occupant of its tree slot had.
__global__ void
CountingStars_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               curandState* randomSeeds,
                               int* candDoor, int* nodeDoor, int* nodeBlocks,
                               int* regionNodeCount, int* novelCounts,
                               float* minCostsR1, int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
                               float* minCost, float* unexploredSampleCosts, bool* goalSet,
                               int* iterations, int iteration,
                               float pReactivate, int maxBlocks, int fanHalfLife,
                               unsigned long long* doorCounts);

/***************************/
/* FAN-OUT ASSIGNMENT KERNEL */
/***************************/
// The ONLY writer of activeFrontierRepeatCount, run over exactly the compacted frontier so every
// member gets a block by construction and nothing outside it gets one. That is what makes rep >= 1
// structural rather than a clamp, and it is why a goal node -- which clears its frontier bit -- needs
// no count clearing and imposes no ordering constraint on Part A.
//
// `scale` shrinks each node's boost ABOVE THE FLOOR when the block ceiling binds, so sum(rep) fits
// exactly while no frontier node is ever left blockless.
__global__ void CountingStars_assignFanout_kernel(uint frontierSize, uint* activeFrontierIdxs,
                                                  int* nodeBlocks, float scale,
                                                  uint* activeFrontierRepeatCount);

/***************************/
/* GET CONTROL PATH TO GOAL KERNEL */
/***************************/
__global__ void CountingStars_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
                                                     int* treeSamplesParentIdxs, uint* goalSetIdxs, int goalSetSize,
                                                     float* pathCosts, float* treeSampleCosts, int* iterations,
                                                     float* minCost);
