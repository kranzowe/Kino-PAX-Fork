#pragma once
#include "planners/Planner.cuh"
#include "graphs/Graph.cuh"
#include "collisionCheck/spatialHash.cuh"

// Which door a candidate/node came in by. Stored per node so the CSV can answer "what built this
// tree", and consumed by the fan-out rule, which gives the optimal door a different block count.
//
// PREFIXED, and the prefix is load-bearing. These are FILE-SCOPE names in a header, and the sweep
// pulls several planner headers into ONE translation unit -- an unprefixed copy is a redefinition
// error in exactly the one .cu that matters and nowhere else. Same discipline as the kernel
// prefixes, different mechanism: kernels collide at LINK time under CUDA_SEPARABLE_COMPILATION,
// header constants collide at COMPILE time.
static const int CS_DOOR_NONE    = 0;
static const int CS_DOOR_COST    = 1;   // OPTIMAL: distance 0, i.e. cost <= minCostsR1[r]
static const int CS_DOOR_EXPLORE = 2;   // freshest: from a region whose ordinality beat the cutoff
static const int CS_DOOR_REACT   = 3;   // existing tree node put back in the frontier by the draw
static const int CS_DOOR_BEST    = 4;   // existing tree node put back because it is its region's best

// Ordinality buckets in the freshness histogram. A candidate's ordinality is its REGION's node
// count, clamped into [0, CS_ORD_BUCKETS): "how full is the region this candidate landed in".
// 256 is chosen because it is the point past which the answer stops mattering -- a region holding
// 255 nodes is not fresh by any reading, and every such region shares the top bucket. The histogram
// plus an exclusive scan then gives the exact top-X cutoff in two O(n) passes, with no sort and no
// rank; see the .cu for why a rank was never needed.
static const int CS_ORD_BUCKETS = 256;

// A region is "thin" -- and every node landing in it earns the fan-out burst -- until it has seen
// this many collision-free propagations. THE SAME RULE AND THE SAME NUMBER KPAXCap AND CleanCost
// USE (`validVertexCounter[region] < 10 ? 15 : 1`), and it is ported deliberately: those two are
// the runtime targets, and a like-for-like fan-out rule is what makes the comparison decidable.
//
// KEYED ON validVertexCounter, NOT regionNodeCount, and the distinction is the whole rule.
// validVertexCounter counts PROPAGATIONS -- it gains ~32 per frontier node per iteration, so a
// region crosses 10 almost as soon as it is touched and the burst stays a one-shot on genuinely
// new ground. regionNodeCount counts ADMITTED NODES: roughly 0.4 per region per iteration at
// B = 10000 spread over 27,000 regions, so a threshold of 10 there would leave nearly every region
// "thin" for hundreds of iterations and the rule would concentrate nothing at all.
static const int CS_NOVEL_THRESH = 10;

// WHICH SIGNAL THE SECOND DOOR RANKS ON. Prefixed for the same reason as CS_DOOR_* -- the sweep
// pulls several planner headers into one translation unit, so an unprefixed file-scope constant is
// a redefinition error in exactly the one .cu that matters.
//
//   ORDINALITY  regionNodeCount[r] -- how many nodes this candidate's region has ever admitted.
//               Prefers the least-populated regions: spatial spreading by node count.
//   DISTANCE    (cost - minCostsR1[r]) / costScale -- how far above its OWN region's best.
//               Prefers near-optimal candidates: quality within each region.
//
// NOT greedy best-first: distance is per-region normalised, so a node in a far, expensive region
// still scores 0 if it is that region's best. Every region that received candidates contributes its
// best at distance 0, so spatial coverage survives structurally and distance only ranks the rest.
static const int CS_KEY_ORDINALITY = 0;
static const int CS_KEY_DISTANCE   = 1;

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
    // THE BUDGET. This is the whole tuning surface, and it is TWO numbers.
    //
    // v1 admitted by PER-REGION QUOTAS (explore_count, cost_count, react_count) and the global
    // frontier size was whatever those happened to produce. That is backwards for the thing that
    // actually matters: GPU throughput is a function of frontier size, so frontier size should be
    // the INPUT.
    //
    // v2 inverts it. goal_frontier_size is the PRIMITIVE, tunable to whatever this GPU is fast at,
    // and the doors fill it in priority order. The budget is then met BY CONSTRUCTION, not tracked
    // -- the same discipline that already makes the block ceiling work, where F and the frontier's
    // block demand are both counted before the launch so the total is known.
    //
    // This also retires the pattern that has failed three times in this line: steering a GLOBAL
    // quantity through PER-REGION knobs with feedback. COMBO fed back its fan-out threshold, COMBO
    // fed back its surplus repHi, and a throughput controller over v1's counts would have been the
    // third. None of them could concentrate, because each handed the gain straight back through its
    // own normalisation.
    // ==================================================================================

    // B -- the node budget for ONE iteration's frontier. Set it where the GPU is fast; everything
    // else in this planner is downstream of it. The doors fill it in priority order:
    //
    //   1. OPTIMAL      every candidate at distance 0 from its region's best. UNCAPPED.
    //   2. FRESHEST     explore_frac of what is LEFT, taken from the least-populated regions.
    //   3. GUARANTEE    each active region's best node, if no optimal admission covered it.
    //   4. DRAW         uniform over the rest of the tree, filling whatever the budget has left.
    //
    // DOORS 1 AND 3 ARE BOTH UNCAPPED, AND BOTH ARE BOUNDED BY NUM_R1_REGIONS -- one node per region
    // can be a region best in an iteration, and one node per uncovered region can be guaranteed. So
    // B ONLY BINDS WHILE B > NUM_R1_REGIONS. Below that the budget is a SOFT TARGET -- still no cap,
    // deliberately -- and budget_used runs over B, held near the active-region count by the
    // guarantee. That is a legitimate operating point and the sweep visits it on purpose: the gap
    // between budget_used and B is the direct measurement of how much of the frontier the priority
    // doors account for before the draw is offered anything at all.
    int h_goalFrontierSize_;

    // Share of the REMAINING budget (B - optimalCount) handed to freshness. The one place novelty
    // still costs something: at 0 the frontier is optimal nodes plus reactivations and nothing
    // else, at 1 the freshness door takes everything the optimal door left.
    float h_exploreFrac_;

    // Which signal the second door ranks on -- CS_KEY_ORDINALITY or CS_KEY_DISTANCE. A swept axis,
    // not a replacement: both keys spend exactly the same budget by construction, so any difference
    // in outcome is the CHOICE OF WHICH CANDIDATES and nothing else. That is what makes the two
    // arms comparable in one figure.
    int h_selectionKey_;

    // ==================================================================================
    // FAN-OUT, REGION-KEYED. Blocks are decided AT ADMISSION and stored per node;
    // propagateFrontier only reads them, so it stays the single writer of
    // activeFrontierRepeatCount.
    //
    //   ADMITTED, region thin   -> maxBlocks     validVertexCounter[r] < CS_NOVEL_THRESH
    //   ADMITTED, region filled -> 1
    //   REACTIVATED (both arms) -> 1
    //
    // INDEPENDENT OF THE DOOR, which is the point. An earlier rule gave the OPTIMAL door maxBlocks
    // and split a `maxBlocks * B` design budget over everyone else -- and that split was INERT in
    // the nominal case, because the divisor came out at B - optimalCount and every frontier node
    // received maxBlocks regardless. So the planner spent its whole block budget uniformly while
    // KPAXCap and CleanCost were concentrating theirs 15-to-1 on new ground. At B = 10000,
    // maxBlocks = 16 that was ~224 candidates produced per node admitted against their ~32, and it
    // was the bulk of the runtime gap -- the propagation kernel, not bookkeeping.
    //
    // maxBlocks AND B ARE INDEPENDENT KNOBS: B sets the frontier's SIZE, maxBlocks sets
    // propagations PER NODE (32 * maxBlocks) for the nodes that earn the burst. maxBlocks = 1 makes
    // every node rep 1, which is KPAXCap's steady state exactly.
    //
    // THE GEOMETRIC RAMP IS GONE with the explore door that indexed it, and ordinality is a
    // SELECTION signal (which candidates are admitted at all) rather than a WEIGHTING one --
    // running it as both would double-count the same fact. Region thinness, which is what the
    // ancestors weight on, is a different measurement and is read straight from the graph.
    // ==================================================================================
    int h_maxBlocks_;

    // ==================================================================================
    // DERIVED PER-ITERATION SCALARS -- measured, never set by a caller, all logged so the planner
    // can be audited from the CSV rather than trusted.
    // ==================================================================================

    // Candidates at distance 0 this iteration, counted exactly on the device by accept pass 1. This
    // is what the host divides the budget against, so it is on the critical path, not a diagnostic.
    uint h_optimalCount_;

    // The freshness cutoff and its boundary probability, solved on the host from the ordinality
    // histogram. A candidate is admitted by the freshness door when its region's ordinality is
    // BELOW the cutoff, or EQUAL to it and it wins the boundary roll.
    //
    // RISING over a run is expected: regions fill, so freshness gets scarce. PINNED AT 0 means no
    // non-optimal candidate is ever fresh enough and explore_frac is doing nothing.
    int   h_ordCutoff_;
    float h_pBoundary_;

    // The realised distance threshold, for the CSV. RISING over a run means the candidate pool is
    // drifting away from the region bests -- the tree is filling with mediocre nodes. PINNED near 0
    // means the door is only admitting near-optimal candidates and has become a second optimal
    // door. NaN in ordinality mode.
    float h_distCutoff_;

    // The guarantee's PLANNED size: active regions whose best node no optimal admission covered.
    // Read against reactivated_best, which is the REALISED count -- the gap is the guaranteed nodes
    // Part B skipped because they were already in the frontier or had reached the goal.
    uint h_guaranteedReact_;

    // What the doors actually committed this iteration: admissions + guarantee + draw. THE NUMBER
    // TO READ AGAINST goal_frontier_size, in the same row rather than shifted by an iteration the
    // way frontier_size is (frontier_size is measured at the top of the NEXT propagateFrontier).
    uint h_budgetUsed_;

    // Admissions by door this iteration, counted exactly on the device and copied back once.
    enum DoorSlot { CS_SLOT_EXPLORE = 0, CS_SLOT_COST, CS_SLOT_REACT, CS_SLOT_BEST, CS_NUM_DOOR_SLOTS };
    thrust::device_vector<unsigned long long> d_doorCounts_;
    unsigned long long* d_doorCounts_ptr_;
    unsigned long long  h_doorCounts_[CS_NUM_DOOR_SLOTS];
    uint h_admittedExplore_, h_admittedCost_, h_reactivated_, h_reactivatedBest_;

    // Blocks the buffer allows, and the scale applied to make the frontier fit inside it.
    //
    //   blockCeiling = 0.8 * remaining / activeBlockSize
    //
    // A DIFFERENT CONSTRAINT FROM blockBudget ABOVE, and both must hold: blockBudget is the design
    // budget, this is the buffer bound. The 0.8 is the margin against the kernel1 condition
    // (frontierRepeatSize * activeBlockSize <= MAX_TREE_SIZE - treeSize). h_blockScale_ < 1 means
    // the ceiling bound and every node's boost was shrunk proportionally ABOVE THE rep >= 1 FLOOR,
    // so sum(rep) fits exactly.
    float h_blockCeiling_;
    float h_blockScale_;

    // Collision-free fraction, kept purely as the diagnostic window into propagation efficiency now
    // that nothing consumes it. Reduced from the graph's cumulative counter arrays.
    float h_globalCollisionFrac_;

    // CleanCost's global cost scale: (mean cost over all valid samples) - (min over regions). It is
    // the DENOMINATOR of a candidate's distance, so it is what makes "distance 0" a scale-free
    // statement rather than one in raw cost units. Logged, because the distance test is the top
    // door and a collapsed scale would otherwise be invisible.
    float h_costScale_;

    // R2 sub-cells claimed so far, as a RUNNING TOTAL rather than a swept count.
    //
    // THE R2 DOOR IS GONE; THE R2 MARKING SURVIVES. Novelty is ordinality now, so nothing in the
    // admission path reads a sub-cell. The claim is kept purely so r2_coverage_pct stays comparable
    // with the KPAX-family baselines, and it is kept in THIS form -- read-then-CAS, one increment
    // per cell ever -- because the alternative is a thrust::count over d_activeSubVertices_ every
    // iteration, which is O(NUM_R2_REGIONS): 2.1M elements at the coarse delta, 37.9M at `tiny`.
    thrust::device_vector<uint> d_touchedR2Count_;
    uint* d_touchedR2Count_ptr_;
    uint  h_touchedR2_;

    // Two denominators the planner already knows and would otherwise discard.
    //   h_propAttempted_     propagation attempts this iteration, INCLUDING collisions
    //   h_candidatesPreGate_ collision-free candidates the accept passes judge, captured before the
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
    // the offset to an edge cell, and every R2 identity is wrong. It no longer decides anything here
    // -- v2's doors never read a sub-cell -- but r2_coverage_pct is only comparable with the
    // baselines if the cells being counted are the right ones, so the corrected table stays.
    //
    // Graph.cu is deliberately left alone: every existing baseline was measured against it. This
    // planner carries a corrected copy and passes it wherever the others pass
    // graph_.d_minValueInRegion_. scripts/check_region_math.py proves the corrected decode is a
    // bijection and measures the shared one's collapse.
    thrust::device_vector<float> d_minCornerCS_;

    // --- per-R1, RESET EVERY ITERATION ---
    //   d_regionCovered_  did an OPTIMAL admission land in this region this iteration? The
    //                     guarantee in Part B is deduplicated against it, so a region whose best
    //                     was already re-admitted through the top door does not also spend a
    //                     guarantee slot on the older node it superseded.
    //
    // d_candCounts_ USED TO LIVE HERE and is gone: it took an atomicAdd from every collision-free
    // candidate -- millions per iteration -- and was never read by anything.
    thrust::device_vector<bool> d_regionCovered_;

    // --- per-R1, CUMULATIVE. THE ORDINALITY SOURCE, and the reason freshness costs nothing.
    //
    // ORDINALITY IS PER-REGION, NOT PER-CANDIDATE. Every candidate in region r shares
    // regionNodeCount[r], so "freshest" means "from the least-populated region" -- which is the
    // novelty signal we want, and it is a single read with no per-candidate counter behind it. Ties
    // break arbitrarily, which is exactly what the boundary roll is there to resolve. ---
    thrust::device_vector<int> d_regionNodeCount_;

    // --- the freshness histogram, and the optimal count that sizes the budget it spends.
    //
    // NEITHER SELECTION NEEDS A RANK, which is why there is no sort here. distance 0 is a
    // THRESHOLD, not an order. Top-X-freshest IS an order, but ordinality is a small non-negative
    // integer, so a histogram plus an exclusive scan gives the exact cutoff in two O(n) atomic
    // passes -- cheaper than a sort, and it reuses atomics that are already everywhere in this
    // file. ---
    // ONE BUFFER, CS_ORD_BUCKETS + 1 ENTRIES. Slot [CS_ORD_BUCKETS] holds the OPTIMAL COUNT, which
    // is not a bucket -- optimal candidates never enter the histogram, because the freshness door
    // spends what is LEFT of the budget after them. It rides here so the host round trip between
    // the two accept passes costs ONE synchronising memcpy instead of two, and that stall sits in
    // the middle of the iteration where it serialises everything behind it.
    thrust::device_vector<int>  d_ordHistogram_;
    int h_ordHistogram_[CS_ORD_BUCKETS + 1];

    // --- THE DISTANCE PATH'S RANKING BUFFER. Dense (compacted-position indexed, NOT slot indexed),
    // written by accept pass 1 and sorted in place, so sortBuf[k] is the k-th smallest distance in
    // this iteration's candidate pool.
    //
    // A SORT RATHER THAN A HISTOGRAM, and the reason is the key's type. Ordinality is a small
    // non-negative integer, which is what made a histogram exact and free; distance is continuous,
    // and for a continuous key binning buys an approximation plus a scale tunable nobody can guess
    // correctly. thrust::sort dispatches to CUB radix -- ~0.2 ms on a 250k candidate pool against a
    // ~15 ms iteration.
    //
    // Every optimal candidate writes 0 here before returning, so the zeros occupy [0, optimalCount)
    // after the sort and the X-th admissible non-optimal sits at optimalCount + X.
    thrust::device_vector<float> d_sortBuf_;

    // The rank the distance cutoff sits at, computed on the DEVICE by a one-thread kernel and read
    // by pass 2 from device memory. That is the point: it removes the mid-iteration host round trip
    // entirely on this path, and that stall is the one that serialises the iteration into two
    // dependent halves. Ordinality mode still does its 257-int readback.
    thrust::device_vector<int> d_distCutoffIdx_;

    // --- per-node, TREE-INDEXED, written once at admission ---
    thrust::device_vector<int> d_nodeBlocks_, d_nodeDoor_;

    // --- per-candidate, UNEXPLORED-SAMPLE-SLOT indexed. Carries accept pass 1's findings to accept
    // pass 2, and pass 2's verdict to Part A. NOT interchangeable with the tree-indexed arrays
    // above: propagate reclaims these slots for the next candidate batch every iteration.
    //
    //   d_candDistance_  (cost - minCostsR1[r]) / costScale, and 0 IS THE OPTIMAL MARK. Written by
    //                    pass 1 over the compacted candidate list and read by pass 2 over the same
    //                    list, so every slot pass 2 touches was written this iteration.
    //   d_candDoor_      pass 2's verdict, and pass 2 is its ONLY writer among the accept passes --
    //                    which is what keeps the door counters free of double counting. ---
    thrust::device_vector<float> d_candDistance_;
    thrust::device_vector<int>   d_candDoor_;

    thrust::device_vector<uint> d_goalSetIdxs_, d_goalSetScanIdx_;
    thrust::device_vector<int> d_iterations_;
    thrust::device_vector<float> d_pathCosts_, d_controlPathsToGoal_;

    // --- raw pointers ---
    float *d_unexploredSamples_ptr_, *d_goalSample_ptr_, *d_unexploredSampleCosts_ptr_;
    float *d_minCostsR1_ptr_, *d_sumCostsR1_ptr_, *d_minCost_ptr_;
    float *d_minCornerCS_ptr_;
    int   *d_cntCostsR1_ptr_;
    int   *d_regionNodeCount_ptr_, *d_ordHistogram_ptr_, *d_distCutoffIdx_ptr_;
    float *d_sortBuf_ptr_;
    bool  *d_regionCovered_ptr_;
    int   *d_nodeBlocks_ptr_, *d_nodeDoor_ptr_, *d_candDoor_ptr_;
    float *d_candDistance_ptr_;
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
// Pure candidate producer plus COUNTING. It records every collision-free propagation, marks R2 cells
// for the coverage metric, and accumulates the per-region cost statistics the accept passes divide
// by -- but makes no admission decision. Counting with atomics is exact and order-independent, so it
// does not violate the invariant that put the decision after propagate: that rule is about cost
// STATISTICS being mid-flight, and minCostsR1 is still only read afterwards.
__global__ void CountingStars_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* vertexCounter, int* validVertexCounter, float* minCorner,
                                                   float* treeSampleCosts, float* minCostsR1, float* sumCostsR1, int* cntCostsR1,
                                                   int* frontierNextXR1s, int* candDoor,
                                                   uint* touchedR2Count,
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
                                                   int* frontierNextXR1s, int* candDoor,
                                                   uint* touchedR2Count,
                                                   float* unexploredSampleCosts, SpatialHashGrid spatialHashGrid);

/***************************/
/* ACCEPT PASS 1 - measure, do not decide */
/***************************/
// One thread per candidate. Computes the two quantities the budget is spent against and NOTHING
// else: the candidate's scale-free distance from its region's best, and -- for the ones that are not
// already optimal -- a vote in the ordinality histogram.
//
// IT STAMPS NO DOOR. The cutoff is not known until this launch has finished and the host has scanned
// the histogram, so a door written here would be a decision taken without the number that decides
// it. Two launches, both O(candidates), and the split is what makes the budget exact.
__global__ void CountingStars_acceptPass1_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                                 float* minCostsR1, int* frontierNextXR1s, float* unexploredSampleCosts,
                                                 int* regionNodeCount, float costScale,
                                                 float* candDistance, int* ordHistogram, float* sortBuf);

/***************************/
/* DISTANCE CUTOFF - one thread, so pass 2 never waits on the host */
/***************************/
// Turns the optimal count (riding in ordHistogram[CS_ORD_BUCKETS]) into the RANK at which the
// distance cutoff sits. One thread because it is three arithmetic ops on two scalars, and the whole
// reason it exists is to keep that arithmetic on the device: doing it on the host would mean a
// synchronising copy in the middle of the iteration, which is exactly the stall this path removes.
__global__ void CountingStars_solveDistanceCutoff_kernel(const int* ordHistogram, int nCand,
                                                         int goalFrontierSize, float exploreFrac,
                                                         int* cutoffIdx);

/***************************/
/* ACCEPT PASS 2 - the ONLY admission decision */
/***************************/
// Admits in priority order against the cutoff the host solved from pass 1's histogram:
//
//   OPTIMAL   distance == 0                                      door = COST     (uncapped)
//   FRESHEST  ordinality <  cutoff                               door = EXPLORE
//             ordinality == cutoff, with probability pBoundary   door = EXPLORE
//
// The boundary roll is what makes the count EXACT rather than approximately right: the X-th freshest
// node almost never falls on a bucket edge, and admitting the whole boundary bucket would overshoot
// the budget by up to one bucket's width -- which, at a coarse ordinality, is most of the frontier.
//
// Also marks regionCovered for every optimal admission, which is what Part B's guarantee is
// deduplicated against.
__global__ void CountingStars_acceptPass2_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                                 int* frontierNextXR1s, int* regionNodeCount,
                                                 float* candDistance, bool* frontierNext, int* candDoor,
                                                 bool* regionCovered, curandState* randomSeeds,
                                                 int selectionKey,
                                                 int ordCutoff, float pBoundary,
                                                 const float* sortBuf, const int* cutoffIdx,
                                                 unsigned long long* doorCounts);

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// Part A inserts admitted candidates and stamps each with its block count. Part B fills what the
// budget has left: the region-best guarantee first, then a uniform draw at pReactivate.
//
// EVERY BRANCH THAT SETS frontier[i] = true MUST WRITE nodeBlocks[i]. A missed one leaves the node
// carrying whatever block count the previous occupant of its tree slot had.
__global__ void
CountingStars_updateFrontier_kernel(bool* frontier, bool* frontierNext, uint* activeFrontierNextIdxs, uint frontierNextSize,
                               float* xGoal, int treeSize, float* unexploredSamples, float* treeSamples,
                               int* unexploredSamplesParentIdxs, int* treeSamplesParentIdxs, float* treeSampleCosts,
                               curandState* randomSeeds,
                               int* candDoor, int* nodeDoor, int* nodeBlocks,
                               int* regionNodeCount, bool* regionCovered, int* validVertexCounter,
                               float* minCostsR1, int* treeXR1s, int* frontierNextXR1s, int* bestNodeIdxPerR1,
                               float* minCost, float* unexploredSampleCosts, bool* goalSet,
                               int* iterations, int iteration,
                               float pReactivate, int maxBlocks,
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
