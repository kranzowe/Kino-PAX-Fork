// CountingStars v3 -- KinoPAX*, with a DERIVED NODE BUDGET split by three fixed shares.
//
// WHAT CHANGED FROM v1, AND WHY IT IS AN INVERSION RATHER THAN A TWEAK. v1 admitted by PER-REGION
// QUOTAS (explore_count, cost_count, react_count) and the global frontier size F was whatever those
// happened to produce. That is backwards for the thing that actually matters: GPU throughput is a
// function of frontier size, so frontier size should be the INPUT.
//
// v2 makes goal_frontier_size B the PRIMITIVE, tunable to whatever this GPU is fast at, and the
// doors fill it in priority order. The budget is met BY CONSTRUCTION, not tracked -- the same
// discipline that already makes the block ceiling work, where F and the frontier's block demand are
// both counted before the launch so the total is known before a single block runs.
//
// ================================================================================================
// WHAT v3 CHANGES. Two things, and the second is the substantive one.
//
// 1. B IS DERIVED, NOT HAND-SWEPT.  B = floor(fill_frac * MAX_TREE_SIZE / MAX_ITER), which is "the
//    frontier size that fills the tree exactly at MAX_ITER" scaled by fill_frac -- so the remaining
//    (1 - fill_frac) of the buffer is what the uncapped optimal door is left to spend. v2 swept B
//    over {200, 2000, 6000, 10000} with no derivation behind any of them; the buffer and the
//    iteration count are the only things that actually constrain a frontier size.
//
// 2. THERE IS NOW A WAY TO BE ADMITTED FOR BEING CHEAP, not only for being THE cheapest. v2's only
//    cost-driven admission was the OPTIMAL door, `cost <= minCostsR1[r]` -- a threshold with nothing
//    behind it, so a candidate one part in 1e6 above its region's minimum was treated exactly like
//    one at ten times the minimum. The new CHEAPEST door selects the top-X smallest cost distances
//    the same way the freshness door selects the top-X smallest ordinalities: a histogram, an
//    exclusive scan, and a boundary roll. An earlier branch did it with a sort over the distances
//    and kept breaking; nothing here needs a rank.
//
// AND THE BUDGET SPLITS THREE WAYS BY FIXED FRACTION rather than "one share plus a remainder":
// explore_frac to freshness, cost_frac to cheapness, and 1 - explore - cost to the draw. All three
// are fractions of B ITSELF; the optimal door and the region-best guarantee remain uncapped and
// spend on top of them.
//
// WHERE THAT LEAVES B. Both uncapped doors are bounded by NUM_R1_REGIONS rather than by B -- at most
// one region best per region per iteration, at most one guaranteed node per uncovered region -- so
// whenever B < nActive the two fractions steer a minority of the frontier and B binds early in a run
// and then stops. That is the honest limit on this design and is exactly what "tree growth is less
// controlled once min cost is always accepted" amounts to. budget_used against goal_frontier_size,
// read as a curve, is the measurement; the lever if it matters is capping the guarantee, not B.
// ================================================================================================
//
// This also retires the pattern that has failed three times in this line: steering a GLOBAL quantity
// through PER-REGION knobs with feedback. COMBO fed back its fan-out threshold; COMBO fed back its
// surplus repHi; a throughput controller over v1's three counts would have been the third. Each one
// normalised, and each one therefore handed its gain straight back.
//
// WHAT IT KEEPS FROM THE STAR LINE. Propagate makes no admission decision; the accept passes run
// after it, once the region statistics have converged. That ordering is load-bearing and must not be
// relaxed: minCostsR1 / maxCostsR1 / sumCostsR1 are updated by atomics from the very threads that
// would read them, so a decision taken inside propagate would see a partial mean and two identical
// candidates would draw different answers purely from scheduling. v3's distMax has the same
// property and the same answer: it is reduced on the host after the launch, before pass 1.
//
// Propagate does still do COUNTING -- candidate counts, R2 cell claims. That is not a relaxation of
// the rule: counting with atomics is exact and order-independent, and the rule is about STATISTICS
// being mid-flight. Nothing reads minCostsR1 until the launch has finished.
//
// THE FIVE DOORS.
//
//   1. OPTIMAL    distance 0, i.e. cost <= minCostsR1[r]. UNCAPPED, and it has first claim every
//                 iteration -- a stronger optimality guarantee than v1's region-best reactivation,
//                 which only put a region's best back AFTER the fact. Safe uncapped while
//                 B > NUM_R1_REGIONS, since NUM_R1_REGIONS is the ceiling on how many nodes can be
//                 a region best in one iteration.
//   2. FRESHEST   explore_frac * B, taken from the least-populated regions.
//   3. CHEAPEST   cost_frac * B, taken from the smallest cost distances.               (v3, NEW)
//   4. GUARANTEE  each active region's best node, if no optimal admission already covered it.
//   5. DRAW       uniform over the rest of the tree at p = react_frac * B / treeSize.
//
// 2 AND 3 ARE A UNION, NOT A PRIORITY CHAIN. They select over the same candidate pool on independent
// signals, so a candidate can clear both -- and it is still one tree node, so the second admission is
// spent as fan-out (CS_DOOR_BOTH takes maxBlocks). Chaining them would make the second door's
// realised count depend on the first door's picks, so neither would meet its share.
//
// NO SELECTION NEEDS A SORT, AND THAT IS NOT A PERFORMANCE ARGUMENT -- IT IS A STRUCTURAL ONE.
// distance 0 is a THRESHOLD, not an order: it is exactly v1's `cost <= minCostsR1[r]`. The other two
// ARE orders, but both are orders over a bucketed value, so a HISTOGRAM plus an exclusive scan gives
// each exact cutoff in two O(n) atomic passes, reusing atomics that are already everywhere in this
// file. (For the record, a sort would also have been affordable: thrust::sort_by_key dispatches to
// CUB radix sort at ~1-2 G keys/s on Pascal, so 1e5-3e5 candidates is 0.05-0.3 ms against a ~15 ms
// iteration. It is not here because it is not needed, not because it is slow. It is also what the
// earlier cost-door branch used, and what kept breaking.)
//
// ORDINALITY IS PER-REGION, NOT PER-CANDIDATE. Every candidate in region r shares regionNodeCount[r],
// so "freshest" means "from the least-populated region". That is the novelty signal we want and it
// costs a single read -- no per-candidate counter, no extra array. Ties break arbitrarily, which is
// what the boundary roll resolves.
//
// COST DISTANCE IS PER-CANDIDATE and is the one signal in this planner that is: it is
// (cost - minCostsR1[r]) / costScale, so it distinguishes two candidates in the same region. Being
// dimensionless it piles up near 0 with a long tail, which is why its bucket map is LOG and anchored
// at an exactly computed distMax -- see CS_COST_BUCKETS in the header.
//
// TWO ACCEPT PASSES, and the split is what makes each budget exact. The histograms must be COMPLETE
// before either cutoff is known, and both cutoffs must be known before anything is admitted. Pass 1
// measures and stamps no door; pass 2 decides and is the only door writer. Both histograms and the
// optimal count share ONE buffer, so v3's second signal costs no second round trip.
//
// THE R2 DOOR IS GONE; THE R2 MARKING SURVIVES. Novelty is ordinality now, so no door reads a
// sub-cell. The claim is kept purely to feed h_touchedR2_ so r2_coverage_pct stays comparable with
// the KPAX-family baselines at O(1) -- the alternative is a thrust::count over d_activeSubVertices_
// every iteration, which is 2.1M elements at the coarse delta and 37.9M at `tiny`.
//
// THE R2 MAPPING IS FIXED HERE AND ONLY HERE. Graph.cu's initializeRegions_kernel does not invert
// getRegion, so its min-corners are wrong and every R2 identity built on them is scrambled. This
// planner carries a corrected copy so the coverage metric counts the right cells; Graph.cu is left
// alone so the existing baselines stay comparable. See the header, and check_region_math.py.
//
// Opts into Graph's dynamic score floor (1/N_active rather than a fixed EPSILON); see Graph.cuh.
// Carries NO retroactive pruning.
#include "planners/CountingStars.cuh"
#include "config/config.h"
#include "statePropagator/statePropagatorSpatialHash.cuh"
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
// distMax's reduction walks minCostsR1 and maxCostsR1 together, so it needs the zip iterator, the
// tuple it yields, and thrust::maximum as the combining op.
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

CountingStars::CountingStars()
{
    graph_ = Graph(W_SIZE);
    // Opt into the mean-share score floor (1/N_active) instead of the legacy fixed EPSILON, which
    // exceeds the score it floors by ~270x at 27k regions and caps the number of discriminated
    // regions at 1/EPSILON = 100 regardless of grid size. KPAX deliberately keeps the legacy floor
    // so it remains a fixed baseline.
    graph_.h_dynamicScoreFloor_ = true;

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
    // v3: the cost histogram's anchor. Per-iteration, unlike minCostsR1 -- see the header.
    d_maxCostsR1_             = thrust::device_vector<float>(NUM_R1_REGIONS);
    d_bestNodeIdxPerR1_       = thrust::device_vector<int>(NUM_R1_REGIONS);
    d_treeXR1s_               = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_frontierNextXR1s_       = thrust::device_vector<int>(MAX_TREE_SIZE);
    d_unexploredSampleCosts_  = thrust::device_vector<float>(MAX_TREE_SIZE);
    d_goalSet_                = thrust::device_vector<bool>(MAX_TREE_SIZE);
    d_doorCounts_             = thrust::device_vector<unsigned long long>(CS_NUM_DOOR_SLOTS, 0ULL);
    d_touchedR2Count_         = thrust::device_vector<uint>(1, 0u);

    // CountingStars' OWN min-corner table -- see the header for why Graph.cu's is not usable.
    d_minCornerCS_            = thrust::device_vector<float>(NUM_R1_REGIONS * STATE_DIM);

    // Per-R1, reset every iteration.
    d_regionCovered_          = thrust::device_vector<bool>(NUM_R1_REGIONS, false);
    // Per-R1, cumulative. THE ORDINALITY SOURCE: how many nodes this region has ever taken.
    d_regionNodeCount_        = thrust::device_vector<int>(NUM_R1_REGIONS);
    // v3: BOTH selection histograms plus the optimal count, in ONE buffer. The mid-iteration
    // readback between the two accept passes stays ONE synchronising memcpy -- 513 ints instead of
    // 257 -- so adding the cost door costs no extra round trip. See CS_HIST_* for the slot layout.
    d_acceptHistogram_        = thrust::device_vector<int>(CS_HIST_SIZE, 0);
    // Per-node, tree-indexed, written once at admission.
    d_nodeBlocks_             = thrust::device_vector<int>(MAX_TREE_SIZE, 1);
    d_nodeDoor_               = thrust::device_vector<int>(MAX_TREE_SIZE, CS_DOOR_NONE);
    // Per-candidate, unexplored-sample-slot indexed.
    d_candDistance_           = thrust::device_vector<float>(MAX_TREE_SIZE, 0.0f);
    d_candDoor_               = thrust::device_vector<int>(MAX_TREE_SIZE, CS_DOOR_NONE);
    // v3.1: the reactivation population, written by the scan and read by Part B's cost arm.
    d_reactEligible_          = thrust::device_vector<bool>(MAX_TREE_SIZE, false);

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
    d_maxCostsR1_ptr_             = thrust::raw_pointer_cast(d_maxCostsR1_.data());
    d_bestNodeIdxPerR1_ptr_       = thrust::raw_pointer_cast(d_bestNodeIdxPerR1_.data());
    d_treeXR1s_ptr_               = thrust::raw_pointer_cast(d_treeXR1s_.data());
    d_frontierNextXR1s_ptr_       = thrust::raw_pointer_cast(d_frontierNextXR1s_.data());
    d_unexploredSampleCosts_ptr_  = thrust::raw_pointer_cast(d_unexploredSampleCosts_.data());
    d_goalSet_ptr_                = thrust::raw_pointer_cast(d_goalSet_.data());
    d_goalSetIdxs_ptr_            = thrust::raw_pointer_cast(d_goalSetIdxs_.data());
    d_goalSetScanIdx_ptr_         = thrust::raw_pointer_cast(d_goalSetScanIdx_.data());
    d_doorCounts_ptr_             = thrust::raw_pointer_cast(d_doorCounts_.data());
    d_touchedR2Count_ptr_         = thrust::raw_pointer_cast(d_touchedR2Count_.data());
    d_minCornerCS_ptr_            = thrust::raw_pointer_cast(d_minCornerCS_.data());
    d_regionCovered_ptr_          = thrust::raw_pointer_cast(d_regionCovered_.data());
    d_reactEligible_ptr_          = thrust::raw_pointer_cast(d_reactEligible_.data());
    d_regionNodeCount_ptr_        = thrust::raw_pointer_cast(d_regionNodeCount_.data());
    d_acceptHistogram_ptr_        = thrust::raw_pointer_cast(d_acceptHistogram_.data());
    d_nodeBlocks_ptr_             = thrust::raw_pointer_cast(d_nodeBlocks_.data());
    d_nodeDoor_ptr_               = thrust::raw_pointer_cast(d_nodeDoor_.data());
    d_candDistance_ptr_           = thrust::raw_pointer_cast(d_candDistance_.data());
    d_candDoor_ptr_               = thrust::raw_pointer_cast(d_candDoor_.data());
    d_iterations_ptr_             = thrust::raw_pointer_cast(d_iterations_.data());
    d_pathCosts_ptr_              = thrust::raw_pointer_cast(d_pathCosts_.data());
    d_controlPathsToGoal_ptr_     = thrust::raw_pointer_cast(d_controlPathsToGoal_.data());

    cudaMalloc(&d_minCost_ptr_, sizeof(float));

    // The corrected R1 min-corner table. Computed ONCE here, exactly as Graph does for its own --
    // the corners are a pure function of the discretisation, not of the run. Everything that would
    // otherwise pass graph_.d_minValueInRegion_ passes this instead; see the header for why.
    CountingStars_initializeRegions_kernel<<<iDivUp(NUM_R1_REGIONS, h_blockSize_), h_blockSize_>>>(d_minCornerCS_ptr_);

    // Spatial hash grid for fast collision detection
    d_spatialHashGrid_ = createSpatialHashGrid();

    // Host buffer for the best (min-cost) reconstructed trajectory
    h_controlPathsToGoal_ = new float[MAX_ITER * SAMPLE_DIM];

    // ================================================================================
    // THE BUDGET. v3: THREE FRACTIONS, and B is DERIVED from the first of them. See the header for
    // why a hand-swept B was the wrong primitive and why the cost door is a histogram.
    // ================================================================================

    // Share of the tree buffer B is sized to consume per iteration; the remaining quarter is left
    // for the uncapped OPTIMAL door. THE ONLY KNOB BEHIND B.
    h_fillFrac_ = 0.75f;

    // "Fill the tree at THIS iteration." MAX_ITER is the whole run, which is what the derivation
    // means; a benchmark running to a different cap can override it before resetPlanner.
    h_fillIters_ = MAX_ITER;

    // Set in resetPlanner from the two above, and never by a caller. Assigned here only so a
    // freshly constructed planner that is somehow read before its first reset shows the same value
    // its first reset would produce.
    h_goalFrontierSize_ = (int)floorf(h_fillFrac_ * float(MAX_TREE_SIZE) / float(h_fillIters_));
    if(h_goalFrontierSize_ < 1) h_goalFrontierSize_ = 1;

    // Share of B given to the FRESHEST door, and to the CHEAPEST door. The draw gets whatever the
    // two leave -- h_reactFrac_ = 0.3 at these values.
    //
    // THESE ARE NOT THE SWEEP'S DERIVED OPERATING POINT, and that is a change from v2, where
    // h_exploreFrac_ was deliberately kept equal to CS_DERIVED_EXPLORE_FRAC so a standalone plan()
    // run and a --single-point sweep pass were the same planner. The sweep's grid is
    // {0, 0.2, 0.4} on both axes -- chosen so each has a 0 ablation arm -- and neither 0.1 nor 0.6
    // is a member, so no derived point could be both these values and a grid point. These stay at
    // the algorithm's stated defaults; --single-point runs (0.75, 0.2, 0.4) instead.
    h_exploreFrac_ = 0.1f;
    h_costFrac_    = 0.6f;
    h_reactFrac_   = fmaxf(0.0f, 1.0f - h_exploreFrac_ - h_costFrac_);

    // v3.1: the completeness floor. NOT A TUNING KNOB and deliberately not an axis -- its job is to
    // be non-zero. See h_reactFloor_ in the header for why a pure top-K reactivation is not
    // probabilistically complete and why the failure is permanent rather than transient.
    h_reactFloor_ = 1e-5f;

    // ---- Fan-out. Blocks a node gets are decided at admission; see the header for the rule. ----
    // rep is a plain COUNT OF BLOCKS with no alignment constraint -- repeatInd writes rep integer
    // entries and kernel1 launches one 32-thread block per entry, so a node at 4 gets
    // 4 x 32 = 128 propagations.
    //
    // SWEPT, and independent of B: this is propagations-per-node for the nodes that earn the burst,
    // where B is frontier size. 4 matches the sweep's derived point.
    h_maxBlocks_   = 4;

    // ---- Derived per-iteration scalars. All recomputed before they are read; these are only the
    // values the CSV would show if a run somehow logged iteration 0. ----
    h_optimalCount_        = 0;
    h_ordCutoff_           = 0;
    h_pBoundary_           = 0.0f;
    h_costCutoff_          = 0;
    h_pCostBoundary_       = 0.0f;
    h_costCutoffDist_      = 0.0f;
    h_distMax_             = 0.0f;
    h_budgetUsed_          = 0;
    h_admittedExplore_     = 0;
    h_admittedCost_        = 0;
    h_admittedCostDist_    = 0;
    h_admittedBoth_        = 0;
    h_reactivated_         = 0;
    h_reactivatedBest_     = 0;
    h_reactivatedCost_     = 0;
    h_reactCutoff_         = 0;
    h_pReactBoundary_      = 0.0f;
    h_reactCutoffDist_     = 0.0f;
    h_dormantCount_        = 0;
    h_blockCeiling_        = 0.0f;
    h_blockScale_          = 1.0f;
    h_globalCollisionFrac_ = 0.1f;
    h_costScale_           = 0.0f;
    h_touchedR2_           = 0;
    h_propAttempted_       = 0;
    h_candidatesPreGate_   = 0;
    for(int i = 0; i < CS_NUM_DOOR_SLOTS; i++) h_doorCounts_[i] = 0ULL;
    for(int i = 0; i < CS_HIST_SIZE; i++) h_acceptHistogram_[i] = 0;

    h_activeBlockSize_ = 32;

    if(VERBOSE)
        {
            printf("/* Planner Type: CountingStars v3 (derived budget, three fixed shares) */\n");
            printf("/* Number of R1 Vertices: %d */\n", NUM_R1_REGIONS);
            printf("/* Number of R2 Vertices: %d */\n", NUM_R2_REGIONS);
            printf("/***************************/\n");
        }
}

CountingStars::~CountingStars()
{
    destroySpatialHashGrid(d_spatialHashGrid_);
    delete[] h_controlPathsToGoal_;
}

void CountingStars::resetPlanner(float* h_initial, float* h_goal)
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
    // No root seed here. propagateFrontier zeroes this array and assigns every count itself over
    // the compacted frontier, so a seed written here would be overwritten before it was read. The
    // root still opens wide: d_nodeBlocks_ is filled with h_maxBlocks_ below.
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);

    // Graph state
    thrust::fill(graph_.d_activeSubVertices_.begin(), graph_.d_activeSubVertices_.end(), false);
    thrust::fill(graph_.d_vertexScoreArray_.begin(), graph_.d_vertexScoreArray_.end(), 0.0f);
    thrust::fill(graph_.d_regionCoverage_.begin(), graph_.d_regionCoverage_.end(), 0.0f);
    thrust::fill(graph_.d_counterArray_.begin(), graph_.d_counterArray_.end(), 0);
    thrust::fill(graph_.d_validCounterArray_.begin(), graph_.d_validCounterArray_.end(), 0);
    graph_.h_nActive_ = 0;

    // Tree state
    thrust::fill(d_treeSamples_.begin(), d_treeSamples_.end(), 0.0f);
    thrust::fill(d_treeSamplesParentIdxs_.begin(), d_treeSamplesParentIdxs_.end(), -1);
    thrust::fill(d_treeSampleCosts_.begin(), d_treeSampleCosts_.end(), 0.0f);
    thrust::fill(d_frontier_.begin(), d_frontier_.begin() + 1, true);

    // KinoPaxPlus optimization state
    thrust::fill(d_minCostsR1_.begin(), d_minCostsR1_.end(), MAX_FLOAT);
    thrust::fill(d_sumCostsR1_.begin(), d_sumCostsR1_.end(), 0.0f);
    // 0, not -MAX_FLOAT: costs are cumulative sums of a non-negative edge cost under either
    // COST_MODE, so 0 is below every real value and an untouched region reads as spread 0 once the
    // MAX_FLOAT min is subtracted -- which the host's transform already skips.
    thrust::fill(d_maxCostsR1_.begin(), d_maxCostsR1_.end(), 0.0f);
    thrust::fill(d_bestNodeIdxPerR1_.begin(), d_bestNodeIdxPerR1_.end(), -1);
    thrust::fill(d_treeXR1s_.begin(), d_treeXR1s_.end(), 0);
    thrust::fill(d_frontierNextXR1s_.begin(), d_frontierNextXR1s_.end(), 0);
    thrust::fill(d_unexploredSampleCosts_.begin(), d_unexploredSampleCosts_.end(), 0.0f);
    thrust::fill(d_goalSet_.begin(), d_goalSet_.end(), false);
    // Region node counts are CUMULATIVE over a run, so this is the one place they are cleared.
    // Carrying them across a reset would start every region already looking full, and the freshness
    // door would admit nothing from iteration 1 onward.
    thrust::fill(d_regionNodeCount_.begin(), d_regionNodeCount_.end(), 0);
    thrust::fill(d_regionCovered_.begin(), d_regionCovered_.end(), false);
    thrust::fill(d_acceptHistogram_.begin(), d_acceptHistogram_.end(), 0);
    // maxBlocks, not 1: the root is admitted by no door, so nothing else would ever write its count.
    thrust::fill(d_nodeBlocks_.begin(), d_nodeBlocks_.end(), h_maxBlocks_);
    thrust::fill(d_nodeDoor_.begin(), d_nodeDoor_.end(), CS_DOOR_NONE);
    thrust::fill(d_candDistance_.begin(), d_candDistance_.end(), 0.0f);
    thrust::fill(d_candDoor_.begin(), d_candDoor_.end(), CS_DOOR_NONE);
    thrust::fill(d_reactEligible_.begin(), d_reactEligible_.end(), false);
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

    // Every derived scalar is reset, not just the ones the constructor happened to set. CleanCost
    // reset NONE of these, so a planner object reused across runs -- which every benchmark does --
    // carried the previous run's final values into iteration 1.
    thrust::fill(d_doorCounts_.begin(), d_doorCounts_.end(), 0ULL);
    for(int i = 0; i < CS_NUM_DOOR_SLOTS; i++) h_doorCounts_[i] = 0ULL;
    for(int i = 0; i < CS_HIST_SIZE; i++) h_acceptHistogram_[i] = 0;
    h_propAttempted_        = 0;
    h_candidatesPreGate_    = 0;
    h_frontierNextSize_     = 0;
    h_frontierRepeatSize_   = 0;
    h_globalCollisionFrac_  = 0.1f;
    h_costScale_            = 0.0f;
    thrust::fill(d_touchedR2Count_.begin(), d_touchedR2Count_.end(), 0u);
    h_touchedR2_            = 0;
    h_optimalCount_         = 0;
    h_ordCutoff_            = 0;
    h_pBoundary_            = 0.0f;
    h_costCutoff_           = 0;
    h_pCostBoundary_        = 0.0f;
    h_costCutoffDist_       = 0.0f;
    h_distMax_              = 0.0f;
    h_budgetUsed_           = 0;
    h_admittedExplore_      = 0;
    h_admittedCost_         = 0;
    h_admittedCostDist_     = 0;
    h_admittedBoth_         = 0;
    h_reactivated_          = 0;
    h_reactivatedBest_      = 0;
    h_reactivatedCost_      = 0;
    h_reactCutoff_          = 0;
    h_pReactBoundary_       = 0.0f;
    h_reactCutoffDist_      = 0.0f;
    h_dormantCount_         = 0;
    h_blockCeiling_         = 0.0f;
    h_blockScale_           = 1.0f;

    // ================================================================================
    // v3: B IS DERIVED HERE, and this is the correct point for it. The tunables are deliberately
    // NOT reset -- that is what lets a benchmark set them once at entry -- and every caller sets
    // them BEFORE calling resetPlanner, exactly as d_nodeBlocks_'s fill from h_maxBlocks_ above
    // already relies on.
    //
    //     B = floor(fill_frac * MAX_TREE_SIZE / fill_iters)
    //
    // "the frontier size that fills the tree exactly at fill_iters", scaled by fill_frac. The
    // remaining (1 - fill_frac) of the buffer is what the uncapped OPTIMAL door is left to spend.
    // ================================================================================
    if(h_fillIters_ < 1) h_fillIters_ = MAX_ITER;
    h_goalFrontierSize_ = (int)floorf(h_fillFrac_ * float(MAX_TREE_SIZE) / float(h_fillIters_));
    if(h_goalFrontierSize_ < 1) h_goalFrontierSize_ = 1;

    // The draw's share is whatever the two selection doors leave. Floored at 0 so a caller setting
    // explore + cost > 1 switches the draw off rather than producing a negative probability; the
    // sweep's cross-check asserts the sum never exceeds 1 in the first place.
    h_reactFrac_ = fmaxf(0.0f, 1.0f - h_exploreFrac_ - h_costFrac_);

    cudaMemcpy(d_treeSamples_ptr_, h_initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalSample_ptr_, h_goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costToGoal_ptr_, &h_costToGoal_, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pathToGoal_ptr_, &h_pathToGoal_, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minCost_ptr_, &h_minCost_, sizeof(float), cudaMemcpyHostToDevice);

    initializeRandomSeeds(static_cast<unsigned int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
}

void CountingStars::plan(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
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
    std::cout << "CountingStars execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CountingStars::planBench(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount, int benchItr)
{
    double t_start = std::clock();
    resetPlanner(h_initial, h_goal);

    while(h_itr_ < MAX_ITER)
        {
            h_itr_++;
            printf("Iteration: %d, Tree Size: %d, Frontier Size: %d\n", h_itr_, h_treeSize_, h_frontierSize_);
            propagateFrontier(d_obstacles_ptr, h_obstaclesCount);
            updateFrontier();

            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    double executionTime = (std::clock() - t_start) / (double)CLOCKS_PER_SEC;
    std::cout << "CountingStars execution time: " << executionTime << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
}

float CountingStars::planOptimize(float* h_initial, float* h_goal, float* d_obstacles_ptr, uint h_obstaclesCount)
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
            updateFrontier();
            if(h_propIterations_ == 0) break;
            if(h_treeSize_ >= MAX_TREE_SIZE - 1) break;
        }
    getControlPathToGoal();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    writeExecutionTimeToCSV(milliseconds / 1000.0);
    std::cout << "CountingStars execution time: " << milliseconds / 1000.0 << " seconds. Iterations: " << h_itr_
              << ". Tree Size: " << h_treeSize_ << ". Best Cost: " << h_minCost_ << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return h_minCost_;
}

// Gathers a frontier node's admission-time block count by TREE INDEX, so the host can total the
// frontier's demand before launching anything and shrink it if the buffer cannot take it. Reduced
// into 64-bit: the frontier can be ~1e5 nodes at up to maxBlocks each, and this is summed every
// iteration -- the same class of overflow the graph's int counter arrays have.
struct CountingStars_BlocksOf
{
    const int* nodeBlocks;
    __host__ __device__ long long operator()(uint treeIdx) const { return (long long)nodeBlocks[treeIdx]; }
};

// CountingStars_UncoveredBest IS GONE with v3's fixed reactivation share. It counted the regions
// the GUARANTEE would have to cover, and it existed only because v2's draw probability was a
// REMAINDER -- (B - admitted - guaranteed)/treeSize -- so `guaranteed` had to be known on the host
// before the kernel that spent it launched. v3's draw is react_frac * B / treeSize, a fixed share
// that reads nothing about the guarantee, so a transform_reduce over NUM_R1_REGIONS leaves every
// iteration and reactivated_best (counted exactly on the device) is the only number anyone read
// anyway.
//
// regionCovered is still written by accept pass 2 rather than by Part A, and still for a reason:
// Part B's guarantee is deduplicated against it and Part B runs in the same launch as Part A.

// v3: the cost histogram's anchor, reduced over the per-region cost spreads. maxCostsR1 is
// per-iteration and minCostsR1 cumulative, so (max - min) is the TIGHT exact bound on this
// iteration's distances. A region no candidate has ever reached has min == MAX_FLOAT and contributes
// 0 rather than a garbage spread; a region touched in an earlier iteration but not this one has
// max == 0 and is clamped to 0 the same way.
struct CountingStars_SpreadOf
{
    __host__ __device__ float operator()(const thrust::tuple<float, float>& t) const
    {
        float mn = thrust::get<0>(t);
        float mx = thrust::get<1>(t);
        return (mn < MAX_FLOAT) ? fmaxf(0.0f, mx - mn) : 0.0f;
    }
};

// ==================================================================================
// THE CUTOFF SOLVE, shared by both selection doors.
//
// The exclusive scan, done on host ints because that is cheaper than launching a kernel to scan them
// and copying the answer back. NEITHER SELECTION NEEDS A RANK -- top-X over a bucketed value is an
// order, and a histogram plus a scan gives the exact cutoff in two O(n) passes. `cutoff` is the
// bucket the X-th candidate falls in; `pBoundary` is the fraction of that bucket needed to reach
// exactly X. Everything strictly below the cutoff is admitted whole.
//
// X == 0 RETURNS cutoff 0 / pBoundary 0 AND ADMITS NOTHING, which is what makes explore_frac = 0 and
// cost_frac = 0 real ablation arms rather than special cases the caller has to guard.
//
// If the loop never breaks, the whole candidate pool is inside what X demands: cutoff lands at
// nBuckets, which no bucket index can equal, so every candidate passes `b < cutoff`. That is the
// intended saturation, not an overrun.
// ==================================================================================
static void csSolveCutoff(const int* hist, int nBuckets, float X, int& cutoff, float& pBoundary)
{
    cutoff    = nBuckets;
    pBoundary = 0.0f;
    float acc = 0.0f;
    for(int k = 0; k < nBuckets; k++)
        {
            float h = float(hist[k]);
            if(acc + h >= X)
                {
                    cutoff    = k;
                    pBoundary = (h > 0.0f) ? fminf(1.0f, fmaxf(0.0f, (X - acc) / h)) : 0.0f;
                    return;
                }
            acc += h;
        }
}

void CountingStars::propagateFrontier(float* d_obstacles_ptr, uint h_obstaclesCount)
{
    // --- PER-ITERATION region cost MAX, cleared before the launch that fills it.
    //
    // THE ASYMMETRY WITH minCostsR1 IS DELIBERATE. The min is cumulative -- it is what "this
    // region's best" means and it must never be forgotten -- while the max is only ever read as
    // max_r (maxCostsR1[r] - minCostsR1[r]), the exact upper bound on THIS iteration's cost
    // distances. Letting it accumulate would leave the cost histogram anchored on a spread no
    // current candidate can reach, so its top buckets would empty out and the resolution would
    // drift downward over a run.
    //
    // 0 rather than -MAX_FLOAT: costs are cumulative sums of a non-negative edge cost under either
    // COST_MODE, so 0 is below every real value, and a region no candidate reached this iteration
    // yields a negative difference that the host's transform clamps to 0.
    //
    // IT CANNOT LIVE IN updateFrontier. The distMax reduction runs there, after this launch, so a
    // clear alongside the accept passes' accumulators would destroy the data it reads.
    thrust::fill(d_maxCostsR1_.begin(), d_maxCostsR1_.end(), 0.0f);

    // --- Build spatial hash grid for fast collision detection ---
    updateSpatialHashGrid(d_spatialHashGrid_, d_obstacles_ptr, h_obstaclesCount);
    cudaMemcpy(&h_spatialHashGrid_, d_spatialHashGrid_, sizeof(SpatialHashGrid), cudaMemcpyDeviceToHost);

    // --- Find indices and size of frontier ---
    thrust::exclusive_scan(d_frontier_.begin(), d_frontier_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontier_[MAX_TREE_SIZE - 1]) ? ++h_frontierSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontier_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // ================================================================================
    // FAN-OUT. Every node already carries the block count it was admitted with; this step only
    // totals the frontier's demand, shrinks it if the buffer cannot take it, and writes the counts.
    //
    // WHY IT IS HERE. d_activeFrontierIdxs_[0, h_frontierSize_) is, at this instant, the WHOLE
    // frontier -- optimal admissions, freshness admissions and reactivations alike -- so the total
    // is known before a single block is launched. That makes the ceiling something to SOLVE against
    // rather than clamp after the fact.
    //
    // THIS IS ALSO THE ONLY WRITER of d_activeFrontierRepeatCount_. It runs over exactly the
    // compacted frontier, so rep >= 1 holds for every member BY CONSTRUCTION -- no node can be left
    // blockless (which would strand its frontier bit forever, since kernel1 clears the bit from the
    // expanding block) and no node outside the frontier can hold a count (which would make repeatInd
    // emit a slice no thread writes, fathering phantom-parented nodes at cost 0). A goal node clears
    // its own frontier bit and is therefore absent here, which is why Part A needs no count clearing
    // and no ordering constraint.
    // ================================================================================
    thrust::fill(d_activeFrontierRepeatCount_.begin(), d_activeFrontierRepeatCount_.end(), 0);
    if(h_frontierSize_ > 0)
        {
            // Blocks the frontier is asking for, and the blocks the buffer can give.
            //
            // FLOAT CASTS ARE MANDATORY: h_treeSize_ is uint, so an overshoot wraps to ~4e9. The
            // 0.8 is the margin against the kernel1 condition below; h_activeBlockSize_, never a
            // literal 32, because that condition uses the member and hard-coding the value here
            // would silently desynchronise them.
            CountingStars_BlocksOf blocksOf{d_nodeBlocks_ptr_};
            long long wantBlocks = thrust::transform_reduce(d_activeFrontierIdxs_.begin(),
                                                            d_activeFrontierIdxs_.begin() + h_frontierSize_,
                                                            blocksOf, (long long)0, thrust::plus<long long>());

            float remaining = fmaxf(0.0f, float(MAX_TREE_SIZE) - float(h_treeSize_));
            h_blockCeiling_ = 0.8f * remaining / float(h_activeBlockSize_);

            // Shrink the BOOST, never the floor. Every frontier node keeps its one block whatever
            // happens, so only the excess above F is scalable:
            //
            //   sum(rep) = F + scale * (wantBlocks - F)  <=  blockCeiling
            //
            // scale == 1 means the ceiling did not bind. Below 1 says the BUFFER, not the fan-out
            // rule, is setting how hard nodes expand -- and a scale near 0 means F itself has eaten
            // the budget, which is a goal_frontier_size problem and no other knob will move it.
            float excess  = float(wantBlocks) - float(h_frontierSize_);
            float allowed = h_blockCeiling_ - float(h_frontierSize_);
            h_blockScale_ = (excess > 0.0f) ? fminf(1.0f, fmaxf(0.0f, allowed / excess)) : 1.0f;

            CountingStars_assignFanout_kernel<<<iDivUp(h_frontierSize_, h_blockSize_), h_blockSize_>>>(
              h_frontierSize_, d_activeFrontierIdxs_ptr_, d_nodeBlocks_ptr_, h_blockScale_,
              d_activeFrontierRepeatCount_ptr_);
        }

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

    // Cap the expanded frontier to the tree buffer: h_frontierRepeatSize_ (sum of the per-node
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

            CountingStars_propagateFrontier_kernel2<<<iDivUp(h_propIterations_ * h_frontierRepeatSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              d_frontierNext_ptr_, graph_.d_validCounterArray_ptr_,
              h_propIterations_, d_minCornerCS_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_maxCostsR1_ptr_, d_sumCostsR1_ptr_,
              d_frontierNextXR1s_ptr_, d_candDoor_ptr_,
              d_touchedR2Count_ptr_,
              d_unexploredSampleCosts_ptr_, h_spatialHashGrid_);

            // kernel2 launches h_frontierRepeatSize_ * h_propIterations_ threads, one candidate each.
            h_propAttempted_ = h_frontierRepeatSize_ * (uint)h_propIterations_;
        }
    else
        {
            CountingStars_propagateFrontier_kernel1<<<iDivUp(h_frontierRepeatSize_ * h_activeBlockSize_, h_activeBlockSize_), h_activeBlockSize_>>>(
              d_frontier_ptr_, d_activeFrontierRepeatIdxs_ptr_, d_treeSamples_ptr_, d_unexploredSamples_ptr_, h_frontierRepeatSize_,
              d_randomSeeds_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_obstacles_ptr, h_obstaclesCount, graph_.d_activeSubVertices_ptr_,
              d_frontierNext_ptr_, graph_.d_validCounterArray_ptr_,
              d_minCornerCS_ptr_,
              d_treeSampleCosts_ptr_, d_minCostsR1_ptr_, d_maxCostsR1_ptr_, d_sumCostsR1_ptr_,
              d_frontierNextXR1s_ptr_, d_candDoor_ptr_,
              d_touchedR2Count_ptr_,
              d_unexploredSampleCosts_ptr_, h_spatialHashGrid_);

            // kernel1 launches one block of h_activeBlockSize_ threads per repeat entry.
            h_propAttempted_ = h_frontierRepeatSize_ * h_activeBlockSize_;
        }
}

/***************************/
/* FAN-OUT ASSIGNMENT KERNEL */
/***************************/
// One thread per COMPACTED FRONTIER ENTRY, so the frontier is covered exactly: every member is
// written once, nothing outside it is touched. Indexing by activeFrontierIdxs[tid] and not by tid is
// the whole reason this is a separate launch from repeatInd -- nodeBlocks is tree-indexed.
__global__ void CountingStars_assignFanout_kernel(uint frontierSize, uint* activeFrontierIdxs,
                                                  int* nodeBlocks, float scale,
                                                  uint* activeFrontierRepeatCount)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierSize) return;

    uint treeIdx = activeFrontierIdxs[tid];
    int  want    = nodeBlocks[treeIdx];
    if(want < 1) want = 1;

    // Scale the BOOST, not the floor, and FLOOR the result rather than rounding. Rounding to
    // nearest would overshoot the ceiling by up to half a block per favoured node -- small, but it
    // would turn sum(rep) <= blockCeiling back into an approximation, and approximating this budget
    // is what used to flip propagate onto the slow kernel2 path.
    unsigned int rep = 1u + (unsigned int)floorf(float(want - 1) * scale);
    activeFrontierRepeatCount[treeIdx] = (rep >= 1u) ? rep : 1u;
}

/***************************/
/* R1 MIN-CORNER INITIALISATION */
/***************************/
// The exact inverse of getRegion's encode:
//
//   r1 = wRegion * C_R1_LENGTH^C_DIM * V_R1_LENGTH^V_DIM + aRegion * V_R1_LENGTH^V_DIM + vRegion
//
// so the groups strip off in reverse significance -- velocity first, then attitude, and whatever
// remains is workspace. Graph.cu's version reads them in the opposite order AND uses hardcoded
// exponents (C_R1_LENGTH^2, V_R1_LENGTH^1) where the encode uses C_DIM and V_DIM, which collapses
// NUM_R1_REGIONS regions onto far fewer distinct corners. Written entirely in config macros so it
// stays correct at any discretisation; scripts/check_region_math.py proves it is a bijection.
//
// The WITHIN-group digit order below is the same as Graph.cu's and was never wrong: getRegion builds
// each group with axis 0 as the most significant digit.
__global__ void CountingStars_initializeRegions_kernel(float* minCorner)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= NUM_R1_REGIONS) return;

    int cPow = 1;
    for(int i = 0; i < C_DIM; ++i) cPow *= C_R1_LENGTH;
    int vPow = 1;
    for(int i = 0; i < V_DIM; ++i) vPow *= V_R1_LENGTH;

    int vRegion = tid % vPow;
    int aRegion = (tid / vPow) % cPow;
    int wRegion = tid / (vPow * cPow);

    int temp = wRegion;
    for(int i = W_DIM - 1; i >= 0; --i)
        {
            minCorner[tid * STATE_DIM + i] = W_MIN + (temp % W_R1_LENGTH) * W_R1_SIZE;
            temp /= W_R1_LENGTH;
        }

    temp = aRegion;
    for(int i = C_DIM - 1; i >= 0; --i)
        {
            minCorner[tid * STATE_DIM + W_DIM + i] = C_MIN + (temp % C_R1_LENGTH) * C_R1_SIZE;
            temp /= C_R1_LENGTH;
        }

    temp = vRegion;
    for(int i = V_DIM - 1; i >= 0; --i)
        {
            minCorner[tid * STATE_DIM + W_DIM + C_DIM + i] = V_MIN + (temp % V_R1_LENGTH) * V_R1_SIZE;
            temp /= V_R1_LENGTH;
        }
}

/***************************/
/* PROPAGATE FRONTIER KERNEL 1 */
/***************************/
// One Block Per Frontier Sample — CANDIDATE PRODUCER ONLY. No acceptance decision, no RNG draw:
// every collision-free sample is recorded with its cost and region, and the accept passes decide
// once the region statistics have converged.
__global__ void CountingStars_propagateFrontier_kernel1(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* validVertexCounter, float* minCorner,
                                                   float* treeSampleCosts, float* minCostsR1, float* maxCostsR1, float* sumCostsR1,
                                                   int* frontierNextXR1s, int* candDoor,
                                                   uint* touchedR2Count,
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
    int x1SubVertex                  = getSubRegion(x1, x1Vertex, minCorner);

    // --- Update Graph statistics ---
    //
    // v3: NO PER-REGION COUNT OF *ATTEMPTS*. v2 opened with atomicAdd(&vertexCounter[x1Vertex], 1)
    // on every thread whether it collided or not -- the hottest atomic in the planner, one per
    // attempted propagation onto a per-region address -- and its only consumer was the collision
    // fraction, a diagnostic. The host already knows both of that fraction's terms exactly:
    // h_propAttempted_ from the launch geometry and h_candidatesPreGate_ from the post-propagate
    // scan. So graph_.d_counterArray_ is left at zero here and nothing reads it; CountingStars never
    // calls graph_.updateVertices(), which is the only other consumer in the family.
    if(valid)
        {
            atomicAdd(&validVertexCounter[x1Vertex], 1);

            // Cumulative cost from root
            float cost = s_x0Cost + edgeCost(s_x0, x1);

            // --- Region cost statistics. These are what the accept passes read once the launch has
            // finished; reading them HERE would see them mid-flight.
            //
            // v3: maxCostsR1 joins them, and it is what anchors the cost histogram. The host reduces
            // max_r (maxCostsR1[r] - minCostsR1[r]) into distMax BEFORE accept pass 1 launches, so
            // the bucket map has an exact upper bound without a third pass over the candidates.
            //
            // cntCostsR1 IS GONE: it was incremented right here, in this same branch, exactly as
            // often as validVertexCounter above, and both were cleared only in resetPlanner. The
            // mean's denominator now reads the graph counter. ---
            if(minCostsR1[x1Vertex] > cost) atomicMinFloat(&minCostsR1[x1Vertex], cost);
            if(maxCostsR1[x1Vertex] < cost) atomicMaxFloat(&maxCostsR1[x1Vertex], cost);
            atomicAdd(&sumCostsR1[x1Vertex], cost);

            // --- R2 MARKING, FOR THE COVERAGE METRIC ONLY. No door reads a sub-cell in v2;
            // ordinality replaced novelty. This survives so r2_coverage_pct stays comparable with
            // the KPAX-family baselines, and it stays in THIS form because the CAS's return value
            // is what makes the running total exact: exactly one thread in the whole launch can
            // turn a given cell from 0 to 1, so touchedR2Count gains exactly one per cell, ever.
            //
            // READ-THEN-CAS, not a bare CAS. The overwhelming majority of candidates land in cells
            // that were claimed iterations ago, and a plain load rejects those without touching the
            // atomic unit at all. The two are exactly equivalent: a cell only ever goes 0 -> 1, so a
            // load that sees 1 can never be a stale rejection of a cell that is still free. ---
            if(activeSubVertices[x1SubVertex] == 0 && atomicCAS(&activeSubVertices[x1SubVertex], 0, 1) == 0)
                atomicAdd(touchedR2Count, 1u);

            // --- Record the candidate. No admission decision, no RNG draw. The door is CLEARED
            // rather than left alone: these slots are reused every iteration, and a stale door from
            // an earlier batch would be read by Part A as an admission. ---
            candDoor[tid]              = CS_DOOR_NONE;
            unexploredSampleCosts[tid] = cost;
            frontierNextXR1s[tid]      = x1Vertex;
            frontierNext[tid]          = true;
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* PROPAGATE FRONTIER KERNEL 2 */
/***************************/
// Iterations mode — CANDIDATE PRODUCER ONLY (see kernel 1).
__global__ void CountingStars_propagateFrontier_kernel2(bool* frontier, uint* activeFrontierIdxs, float* treeSamples,
                                                   float* unexploredSamples, uint frontierSize, curandState* randomSeeds,
                                                   int* unexploredSamplesParentIdxs, float* obstacles, int obstaclesCount,
                                                   int* activeSubVertices, bool* frontierNext,
                                                   int* validVertexCounter, int iterations,
                                                   float* minCorner,
                                                   float* treeSampleCosts, float* minCostsR1, float* maxCostsR1, float* sumCostsR1,
                                                   int* frontierNextXR1s, int* candDoor,
                                                   uint* touchedR2Count,
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
    int x1SubVertex                  = getSubRegion(x1, x1Vertex, minCorner);

    // --- Update Graph statistics. No attempt counter, and maxCostsR1 alongside the min (kernel 1). ---
    if(valid)
        {
            atomicAdd(&validVertexCounter[x1Vertex], 1);

            float cost = x0Cost + edgeCost(x0, x1);

            // --- Region cost statistics (see kernel 1). ---
            if(minCostsR1[x1Vertex] > cost) atomicMinFloat(&minCostsR1[x1Vertex], cost);
            if(maxCostsR1[x1Vertex] < cost) atomicMaxFloat(&maxCostsR1[x1Vertex], cost);
            atomicAdd(&sumCostsR1[x1Vertex], cost);

            // --- R2 marking for the coverage metric only, read-then-CAS (see kernel 1). ---
            if(activeSubVertices[x1SubVertex] == 0 && atomicCAS(&activeSubVertices[x1SubVertex], 0, 1) == 0)
                atomicAdd(touchedR2Count, 1u);

            // --- Record the candidate (see kernel 1 for why the door is cleared here). ---
            candDoor[tid]              = CS_DOOR_NONE;
            unexploredSampleCosts[tid] = cost;
            frontierNextXR1s[tid]      = x1Vertex;
            frontierNext[tid]          = true;
        }

    randomSeeds[tid] = randSeed;
}

/***************************/
/* ACCEPT PASS 1 - measure, do not decide */
/***************************/
// Runs after propagate has finished, so minCostsR1 is converged rather than mid-flight -- the one
// invariant CleanCost established that this planner keeps.
//
// It computes the two quantities the budget is spent against, and NOTHING else:
//
//   distance   (cost - minCostsR1[r]) / costScale, and 0 IS THE OPTIMAL MARK. costScale is
//              CleanCost's global scale -- (mean cost over valid samples) - (min over regions) --
//              which is what makes the distance scale-free rather than a raw cost difference.
//   ordinality regionNodeCount[r], the candidate's REGION's population. Per-region, not
//              per-candidate: "freshest" means "from the least-populated region", which is a single
//              read with no per-candidate counter behind it.
//
// v3: A NON-OPTIMAL CANDIDATE VOTES IN TWO HISTOGRAMS, not one -- its ordinality bucket and its cost
// bucket. The two doors are independent selections over the same pool, so the same candidate is
// eligible for both and pass 2 resolves the overlap. Optimal candidates still vote in NEITHER.
//
// THE OPTIMAL TEST IS `cost <= minCostsR1[r]`, NOT `distance == 0.0f`, and the difference matters
// when costScale collapses to 0 (an empty or single-cost tree): the division would be 0/0. The
// comparison is exact, is what v1's cost door was, and distance is DERIVED from it afterwards --
// so the flag written into candDistance is 0 for exactly the set the comparison selects.
__global__ void CountingStars_acceptPass1_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                                 float* minCostsR1, int* frontierNextXR1s, float* unexploredSampleCosts,
                                                 int* regionNodeCount, float costScale, float distMax,
                                                 float* candDistance, int* acceptHistogram)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierNextSize) return;

    // ONE indirection, and every array below is indexed by idx -- the unexplored-sample SLOT --
    // never by tid, the compacted position. Indexing by tid would read another candidate's data.
    int   idx  = activeFrontierNextIdxs[tid];
    float cost = unexploredSampleCosts[idx];
    int   xR1  = frontierNextXR1s[idx];
    float m    = minCostsR1[xR1];

    if(cost <= m)
        {
            // OPTIMAL. Uncapped, and it does not enter the histogram: the freshness door spends
            // what is LEFT of the budget after these, so counting them as fresh too would let one
            // node consume two doors' worth of it.
            candDistance[idx] = 0.0f;
            // Slot CS_HIST_OPT_SLOT is NOT a bucket -- it is the optimal count, riding in the same
            // buffer so the host reads everything back in one synchronising memcpy. Neither cutoff
            // scan touches it.
            atomicAdd(&acceptHistogram[CS_HIST_OPT_SLOT], 1);
            return;
        }

    // Non-optimal, so the distance written here MUST NOT BE ZERO -- pass 2 reads 0 as the optimal
    // mark. Two ways it could be: a collapsed costScale, where the ratio is 0/0 rather than
    // infinite, and an underflow when the spread is enormous next to the gap.
    //
    // The collapsed-scale branch falls back to the RAW difference, which cannot be 0 here (cost > m,
    // and float subtraction of nearby values is exact) and is in the same units as the distMax that
    // branch produces, so the two stay comparable.
    //
    // The underflow falls back to CS_MIN_DISTANCE, not to the raw difference as v2 did. v2 never
    // used the value for anything but the zero test; v3 BUCKETS it, and a raw gap is in cost units,
    // can exceed distMax, and would bucket a candidate sitting at its region's minimum as the WORST
    // in the pool. See CS_MIN_DISTANCE. The !( > 0) form also catches a NaN.
    float d = csNodeDistance(cost, m, costScale);
    candDistance[idx] = d;

    // --- BOTH VOTES. csOrdBucket / csCostBucket are the single definitions of the two bucket maps
    // (see the header); pass 2 calls the very same functions, so the two passes cannot disagree
    // about which bucket a candidate is in -- which they could when each spelled the clamp out. ---
    atomicAdd(&acceptHistogram[CS_HIST_ORD_BASE  + csOrdBucket(regionNodeCount[xR1])], 1);
    atomicAdd(&acceptHistogram[CS_HIST_COST_BASE + csCostBucket(d, distMax)], 1);
}

/***************************/
/* REACTIVATION SCAN - measure the dormant tree (v3.1) */
/***************************/
// The Part B counterpart of accept pass 1: it measures and decides nothing. One thread per tree node.
//
// WHAT COUNTS AS ELIGIBLE, and why the answer is stored rather than recomputed. A node is in the
// reactivation population when it is dormant (`!frontier`), still expandable (`!goalSet`), and not
// its region's best -- region bests come back through the GUARANTEE for free, so letting them into
// this histogram would spend the cost arm's budget on nodes that were never going to need it, and
// they sit at distance ~0 so they would take the budget's cheap end entirely.
//
// bestNodeIdxPerR1 is atomicExch'd by PART A, which runs in the same launch as Part B. So Part B
// cannot re-derive this predicate and get the scan's answer, and a cost arm selecting from a
// population its own histogram never measured would neither hit its budget nor be exact. Writing the
// flag makes the scan the SINGLE writer of the population -- the same discipline that makes accept
// pass 1 the single measurer of the candidate pool.
//
// distMax IS THE CANDIDATE ANCHOR, REUSED. It bounds this iteration's CANDIDATE distances, not tree
// distances, so a dormant node above it clamps into the top bucket. That is harmless for this
// selection and is not a shortcut worth removing: the arm takes the SMALLEST distances,
// csCostBucket stays monotone (so csSolveCutoff's exact min(X, n) still holds), and everything in
// the top bucket is the expensive tail being excluded anyway. h_reactCutoffDist_ and
// h_dormantCount_ are logged so the one case where it would bite -- a cutoff pinned at the top
// bucket, meaning the budget exceeds the population below distMax -- is visible rather than assumed.
__global__ void CountingStars_reactScan_kernel(int treeSize, bool* frontier, bool* goalSet,
                                               int* treeXR1s, float* treeSampleCosts, float* minCostsR1,
                                               int* bestNodeIdxPerR1, float costScale, float distMax,
                                               bool* reactEligible, int* acceptHistogram)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= treeSize) return;

    // Written for EVERY node, not only eligible ones: the array is reused across iterations, and a
    // stale true would put a node into an arm whose histogram never counted it.
    if(goalSet[tid] || frontier[tid])
        {
            reactEligible[tid] = false;
            return;
        }

    int xR1 = treeXR1s[tid];
    if(tid == bestNodeIdxPerR1[xR1])
        {
            reactEligible[tid] = false;
            return;
        }

    reactEligible[tid] = true;
    atomicAdd(&acceptHistogram[CS_HIST_DORMANT_SLOT], 1);
    atomicAdd(&acceptHistogram[CS_HIST_REACT_BASE
                               + csCostBucket(csNodeDistance(treeSampleCosts[tid], minCostsR1[xR1], costScale),
                                              distMax)],
              1);
}

/***************************/
/* ACCEPT PASS 2 - the ONLY admission decision */
/***************************/
// Admits against the cutoffs the host solved from pass 1's two histograms:
//
//   OPTIMAL   distance == 0                                          door = COST      (uncapped)
//   FRESHEST  ordBucket  <  ordCutoff,  or == it at pBoundary        door = EXPLORE
//   CHEAPEST  costBucket <  costCutoff, or == it at pCostBoundary    door = COSTDIST
//   BOTH      cleared both of the two above                          door = BOTH
//
// THE TWO NON-OPTIMAL DOORS ARE A UNION, NOT A PRIORITY CHAIN, and that is the one place v3's
// structure differs from a straight second copy of the freshness door. They select over the same
// candidate pool on independent signals -- region population and cost distance -- so a candidate can
// clear both. It is still ONE tree node, so the second admission is spent as FAN-OUT: CS_DOOR_BOTH
// takes maxBlocks in Part A whatever its region's thinness says.
//
// Ordering the two as a priority chain instead would make the second door's realised count depend on
// the first door's picks, so neither would meet its share.
//
// THE BOUNDARY ROLL IS WHAT MAKES EACH COUNT EXACT. The X-th candidate almost never falls on a
// bucket edge, and admitting the whole boundary bucket would overshoot by up to one bucket's width
// -- which, where thousands of candidates share a bucket, is most of a frontier. The roll spends the
// fractional remainder and nothing more. ONE curandState is loaded and stored even when a candidate
// sits on both boundaries; the two draws come off the same advanced state, which is exactly what two
// sequential draws mean.
//
// BOTH SIGNALS ARE RE-READ HERE RATHER THAN CARRIED FROM PASS 1, and that is safe by construction:
// regionNodeCount is only written by Part A of updateFrontier, which runs after both passes, and
// candDistance was written by pass 1 over this same compacted list. The bucket maps are the shared
// csOrdBucket / csCostBucket, so a candidate cannot be compared against a cutoff derived from a
// histogram it was never counted in.
__global__ void CountingStars_acceptPass2_kernel(uint* activeFrontierNextIdxs, uint frontierNextSize,
                                                 int* frontierNextXR1s, int* regionNodeCount,
                                                 float* candDistance, bool* frontierNext, int* candDoor,
                                                 bool* regionCovered, curandState* randomSeeds,
                                                 int ordCutoff, float pBoundary,
                                                 int costCutoff, float pCostBoundary, float distMax,
                                                 unsigned long long* doorCounts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= frontierNextSize) return;

    int idx = activeFrontierNextIdxs[tid];
    int xR1 = frontierNextXR1s[idx];

    // --- OPTIMAL: first claim on the budget, every iteration. That is a stronger optimality
    // guarantee than v1's region-best reactivation, which only restored a region's best AFTER it
    // had already been passed over. ---
    if(candDistance[idx] == 0.0f)
        {
            candDoor[idx] = CS_DOOR_COST;
            // Part B's guarantee is deduplicated against this: a region whose best came back in
            // through the top door does not also spend a guarantee slot on the node it superseded.
            regionCovered[xR1] = true;
            atomicAdd(&doorCounts[CountingStars::CS_SLOT_COST], 1ULL);
            return;
        }

    // --- FRESHEST: from the least-populated regions. CHEAPEST: from the smallest cost distances.
    // Evaluated together, because they are a union. ---
    int  ob       = csOrdBucket(regionNodeCount[xR1]);
    int  cb       = csCostBucket(candDistance[idx], distMax);
    bool takeOrd  = (ob < ordCutoff);
    bool takeCost = (cb < costCutoff);

    // One state, at most two draws, one store -- even for a candidate sitting on both boundaries.
    bool ordBoundary  = (!takeOrd  && ob == ordCutoff  && pBoundary     > 0.0f);
    bool costBoundary = (!takeCost && cb == costCutoff && pCostBoundary > 0.0f);
    if(ordBoundary || costBoundary)
        {
            curandState seed = randomSeeds[idx];
            if(ordBoundary)  takeOrd  = (curand_uniform(&seed) < pBoundary);
            if(costBoundary) takeCost = (curand_uniform(&seed) < pCostBoundary);
            randomSeeds[idx] = seed;
        }

    if(takeOrd || takeCost)
        {
            // BOTH is its own door value, so nodeDoor answers "which door built this node" without
            // a side array, and Part A reads it for the fan-out boost.
            candDoor[idx] = (takeOrd && takeCost) ? CS_DOOR_BOTH
                                                  : (takeOrd ? CS_DOOR_EXPLORE : CS_DOOR_COSTDIST);
            // THE COUNTERS OVERLAP DELIBERATELY: a BOTH candidate increments all three, so
            // explore + costdist - both is the exact number of nodes these two doors admitted.
            if(takeOrd)              atomicAdd(&doorCounts[CountingStars::CS_SLOT_EXPLORE], 1ULL);
            if(takeCost)             atomicAdd(&doorCounts[CountingStars::CS_SLOT_COSTDIST], 1ULL);
            if(takeOrd && takeCost)  atomicAdd(&doorCounts[CountingStars::CS_SLOT_BOTH], 1ULL);
            return;
        }

    // --- Rejected. Subtractive, like CleanCost's: propagate set the flag, admission leaves it, and
    // only rejection clears it. ---
    candDoor[idx]     = CS_DOOR_NONE;
    frontierNext[idx] = false;
}

/***************************/
/* FRONTIER UPDATE KERNEL */
/***************************/
// Part A inserts this iteration's admitted candidates; Part B fills what the budget has left. The
// two run in one launch over disjoint index ranges -- Part A owns
// [treeSize, treeSize + frontierNextSize) and Part B owns [0, treeSize) -- so they never contend for
// a node.
//
// EVERY BRANCH THAT SETS frontier[i] = true MUST WRITE nodeBlocks[i]. A missed one leaves the node
// carrying whatever block count the previous occupant of its tree slot had, and it fails silently:
// the node simply expands by the wrong amount.
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
                               bool* reactEligible, float costScale, float distMax,
                               int reactCutoff, float pReactBoundary, float reactFloor,
                               int maxBlocks,
                               unsigned long long* doorCounts)
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
            frontierNext[x1UnexploredIdx] = false;

            float* x1  = &unexploredSamples[x1UnexploredIdx * SAMPLE_DIM];
            int x0Idx  = unexploredSamplesParentIdxs[x1UnexploredIdx];
            float cost = unexploredSampleCosts[x1UnexploredIdx];
            int xR1    = frontierNextXR1s[x1UnexploredIdx];
            int door   = candDoor[x1UnexploredIdx];

            // Transfer to tree
            treeSamplesParentIdxs[x1TreeIdx] = x0Idx;
            for(int i = 0; i < SAMPLE_DIM; i++)
                treeSamples[x1TreeIdx * SAMPLE_DIM + i] = x1[i];
            treeSampleCosts[x1TreeIdx] = cost;
            treeXR1s[x1TreeIdx]        = xR1;
            nodeDoor[x1TreeIdx]        = door;

            // Always add to frontier (it survived the gate)
            frontier[x1TreeIdx] = true;

            // --- THE REGION'S POPULATION, which is the ordinality every LATER iteration's freshness
            // cutoff is measured against. Cumulative over the run and never reset mid-run, so it is
            // "how many nodes this region has ever taken". The atomicAdd's return value is not
            // needed any more -- v1 used it to index the geometric fan-out ramp, which is gone with
            // the explore door that indexed it -- but the increment itself is load-bearing. ---
            atomicAdd(&regionNodeCount[xR1], 1);

            // --- FAN-OUT, decided here and read next iteration by propagateFrontier.
            //
            // REGION-KEYED, NOT DOOR-KEYED. A node earns the burst because it landed in ground the
            // search has barely touched, whatever door let it in -- exactly KPAXCap's and
            // CleanCost's `validVertexCounter[region] < 10 ? 15 : 1`.
            //
            // validVertexCounter was written by propagate in a completed launch, so reading it here
            // is safe. It counts PROPAGATIONS and gains ~32 per frontier node per iteration, which
            // is what keeps the burst a one-shot: a region crosses the threshold almost as soon as
            // it is touched. Keying on regionNodeCount instead would leave nearly every region thin
            // for hundreds of iterations and concentrate nothing -- see CS_NOVEL_THRESH. ---
            int blocks = (validVertexCounter[xR1] < CS_NOVEL_THRESH) ? maxBlocks : 1;
            // v3: a candidate that cleared BOTH selection cutoffs earns the burst regardless of its
            // region's thinness. That is where "a node can get added twice" is spent -- a candidate
            // is one tree node however many doors admitted it, so the second admission buys
            // propagation instead of a duplicate.
            if(door == CS_DOOR_BOTH) blocks = maxBlocks;
            nodeBlocks[x1TreeIdx] = (blocks > 1) ? blocks : 1;

            // Update best-node index if this is the new region best. THE GUARANTEE'S TABLE: Part B
            // reads it to put a region's cheapest node back when no optimal admission covered the
            // region this iteration. One atomicExch on a branch that is already taken.
            if(cost <= minCostsR1[xR1])
                atomicExch(&bestNodeIdxPerR1[xR1], x1TreeIdx);

            // Goal criteria check - accumulate goal nodes into goalSet; the min-cost path is
            // reconstructed afterwards by getControlPathToGoal.
            //
            // NO ORDERING CONSTRAINT. This used to have to run last, because it cleared a repeat
            // count the fan-out write could otherwise resurrect -- leaving a node with count > 0 and
            // frontier == false, owning a slice of d_activeFrontierRepeatIdxs_ that no thread
            // writes, so propagate expanded stale tree indices into phantom-parented nodes at cost 0
            // that then won minCost. Counts are not written here at all now: clearing the frontier
            // bit is sufficient, because propagateFrontier zeroes every count and then writes only
            // the compacted frontier, which a goal node is by definition not in.
            if(distance(x1, s_xGoal) < GOAL_THRESH && cost <= *minCost)
                {
                    atomicMinFloat(minCost, cost);
                    goalSet[x1TreeIdx]  = true;
                    frontier[x1TreeIdx] = false;
                    iterations[x1TreeIdx] = iteration;
                }
        }

    // --- Part B: Re-activate existing tree nodes, filling what the budget has left ---
    else if(tid < frontierNextSize + treeSize)
        {
            int treeIdx = tid - frontierNextSize;
            if(goalSet[treeIdx]) return;
            if(frontier[treeIdx]) return;   // already in the frontier; nothing to draw for

            int xR1 = treeXR1s[treeIdx];

            // --- THE REGION-BEST GUARANTEE, DEDUPLICATED. Unconditional for an UNCOVERED region:
            // no roll, no budget, ahead of the draw.
            //
            // This is KinoPaxPlus's invariant and the reason a region's cheapest node keeps getting
            // expanded until it stops being cheapest. Without it a region's best sits in the frontier
            // only by luck, its subtree goes many iterations without improvement, and FINAL COST
            // STALLS.
            //
            // WHAT THE DEDUP BUYS OVER v1. v1 ran this for EVERY active region every iteration, so
            // F >= nActive unconditionally. Here a region whose best was just re-admitted through
            // the optimal door is already covered, so the guarantee only pays for the regions the
            // top door missed. v3 no longer COUNTS those on the host -- the draw's share does not
            // depend on them any more -- but the dedup itself is kept: without it a region pays for
            // its best twice in one iteration, once as a candidate and once as a reactivation.
            //
            // IT IS STILL A FLOOR ON F AT THE UNCOVERED-REGION COUNT, and that is the honest limit
            // on what B can control: this arm is UNCONDITIONAL, so once nActive exceeds B the
            // guarantee alone overruns the budget and the draw contributes nothing. B binds only
            // while B > NUM_R1_REGIONS. Capping it here is the obvious next lever if the sweep says
            // B is not moving F -- KinoPaxPlus's precedent is hysteresis (un-prune a region best
            // only after ~5 idle iterations), which caps re-entry at nActive/5 without giving up
            // the invariant.
            //
            // bestNodeIdxPerR1 is written in Part A under `cost <= minCostsR1[xR1]` with an
            // atomicExch, so ties resolve arbitrarily -- exactly one node per region, which is what
            // this arm wants. A Part A write racing this read can only move the guarantee by one
            // node for one iteration, and the covered case is gated by regionCovered anyway.
            if(!regionCovered[xR1] && treeIdx == bestNodeIdxPerR1[xR1])
                {
                    frontier[treeIdx]   = true;
                    nodeDoor[treeIdx]   = CS_DOOR_BEST;
                    // ONE BLOCK. A reactivated node is being revisited, not discovered, and both
                    // ancestors set exactly 1 on both reactivation arms.
                    nodeBlocks[treeIdx] = 1;
                    atomicAdd(&doorCounts[CountingStars::CS_SLOT_BEST], 1ULL);
                    return;
                }

            // --- THE DRAW. Uniform over what the guarantee did not take, at
            // p = (B - admitted - guaranteed) / treeSize, so the EXPECTED number added is exactly
            // the budget's remainder whatever the tree size.
            //
            // It exists for a different job from the guarantee. The guarantee keeps every region's
            // best expandable, which is an OPTIMALITY mechanism. This keeps a thin random sample of
            // ordinary nodes alive, which is a REACH mechanism -- without it the only nodes ever
            // re-expanded would be region bests, and the tree would only ever deepen along its own
            // cheapest paths.
            //
            // Uniform is the simplest thing that hits the count exactly. Top-K by cost, by recency,
            // or a mix all slot in HERE and nowhere else. ---
            // ============================================================================
            // v3.1: THE REACTIVATION BUDGET IS SPENT ENTIRELY ON COST, and a separate floor keeps
            // the planner complete. v2 and v3 drew UNIFORMLY here, which is the one cost mechanism
            // CleanCost has and this planner did not: its Part B reactivation is weighted by
            // costProbExpGlobal, so cheap dormant nodes come back preferentially. The volumes were
            // already comparable -- this is selectivity, not throughput.
            //
            // WHY IT IS THIS ARM THAT MATTERS FOR OPTIMALITY. A cheaper route to the goal is built
            // by deepening a cheap INTERIOR branch, and Part B is the only thing that re-expands
            // the interior; new candidates are the growing edge.
            //
            // ONE curandState, at most two draws, one store -- the accept-pass-2 pattern.
            // ============================================================================
            curandState seed = randomSeeds[treeIdx];

            // --- ARM 2: CHEAPEST. The whole react_frac * B budget, top-K by cost distance against
            // the cutoff the host solved from the scan's histogram. Restricted to the population
            // the scan measured, because a node outside it was never counted and would be judged
            // against a cutoff it did not contribute to. ---
            if(reactEligible[treeIdx])
                {
                    int rb = csCostBucket(csNodeDistance(treeSampleCosts[treeIdx], minCostsR1[xR1], costScale),
                                          distMax);
                    bool take = (rb < reactCutoff);
                    if(!take && rb == reactCutoff && pReactBoundary > 0.0f)
                        take = (curand_uniform(&seed) < pReactBoundary);

                    if(take)
                        {
                            randomSeeds[treeIdx] = seed;
                            frontier[treeIdx]    = true;
                            nodeDoor[treeIdx]    = CS_DOOR_REACT_COST;
                            nodeBlocks[treeIdx]  = 1;   // revisited, not discovered -- see above
                            atomicAdd(&doorCounts[CountingStars::CS_SLOT_REACT_COST], 1ULL);
                            return;
                        }
                }

            // --- ARM 3: THE COMPLETENESS FLOOR. Deliberately OUTSIDE the eligibility test and
            // outside any budget: every dormant node gets this roll, including a covered region's
            // superseded best. Eligibility is bookkeeping for the budgeted arm; completeness has to
            // cover the whole tree.
            //
            // Without it the cost arm alone is NOT probabilistically complete, and the failure is
            // permanent: a node's distance has a fixed numerator over a non-increasing
            // minCostsR1[r], so it only ever grows, and a node once above the cutoff can never
            // return. See h_reactFloor_. ---
            if(reactFloor > 0.0f && curand_uniform(&seed) < reactFloor)
                {
                    frontier[treeIdx]   = true;
                    nodeDoor[treeIdx]   = CS_DOOR_REACT;
                    nodeBlocks[treeIdx] = 1;
                    atomicAdd(&doorCounts[CountingStars::CS_SLOT_REACT], 1ULL);
                }
            randomSeeds[treeIdx] = seed;
        }
}


void CountingStars::updateFrontier()
{
    // --- Find indices and size of the candidate list ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontierNext_[MAX_TREE_SIZE - 1]) ? ++h_frontierNextSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // Collision-free candidates the accept passes are about to judge. Captured here because the
    // post-gate re-scan below overwrites h_frontierNextSize_ with the survivors.
    //
    // Do NOT reconstruct it as frontierRepeatSize * 32 * nu: h_propAttempted_ is set by two
    // different formulas depending on which propagate path ran (repeatSize * 32 on kernel1,
    // repeatSize * propIterations on kernel2), so that product is a no-op round trip in one branch
    // and overstates by up to 32x in the other.
    h_candidatesPreGate_ = h_frontierNextSize_;

    // Per-iteration accumulators the accept passes fill. regionCovered MUST be cleared here rather
    // than in propagate: it is written by accept pass 2 and read by Part B, both inside this call.
    thrust::fill(d_doorCounts_.begin(), d_doorCounts_.end(), 0ULL);
    // One fill covers all three: both bucket ranges and the optimal count share the buffer.
    thrust::fill(d_acceptHistogram_.begin(), d_acceptHistogram_.end(), 0);
    thrust::fill(d_regionCovered_.begin(), d_regionCovered_.end(), false);
    // NOTE d_maxCostsR1_ IS NOT CLEARED HERE, and must not be. Every other accumulator on this list
    // is written by the accept passes further down this same call, so clearing them at the top is
    // correct. maxCostsR1 is written by the PROPAGATE that has already run, and the distMax
    // reduction a few lines below is its only reader -- a clear here would zero exactly the data it
    // needs. It is cleared at the top of propagateFrontier instead.

    // --- Collision fraction. v3: FREE, and per-iteration. v2 summed graph_.d_counterArray_ and
    // d_validCounterArray_ over NUM_R1_REGIONS to get it, which paid for an atomicAdd on every
    // ATTEMPTED propagation just to feed a diagnostic. Both terms are already exact on the host:
    // h_propAttempted_ is set by whichever propagate path ran, and h_candidatesPreGate_ is the scan
    // above. Two reductions and the planner's hottest atomic go away for the same number. ---
    h_globalCollisionFrac_ = (h_propAttempted_ > 0)
                               ? 1.0f - float(h_candidatesPreGate_) / float(h_propAttempted_)
                               : 0.0f;

    // --- CleanCost's GLOBAL cost scale: (mean cost over all valid samples) - (min over regions).
    // It is the denominator of a candidate's distance, which is what makes "distance 0" a
    // scale-free statement instead of one in raw cost units. Unreached regions contribute sum = 0,
    // cnt = 0, min = MAX_FLOAT, so all three reductions are correct with no masking. Three passes
    // over NUM_R1_REGIONS against the two existing MAX_TREE_SIZE scans -- negligible. Must run
    // after propagate (which fills the arrays) and before accept pass 1. ---
    float sumAll = thrust::reduce(d_sumCostsR1_.begin(), d_sumCostsR1_.end(), 0.0f);
    // v3: the denominator comes from the graph counter, which counts exactly what d_cntCostsR1_ used
    // to -- both were incremented in the same if(valid) branch of propagate and cleared only at
    // reset, so they were equal at every point in a run. 64-BIT: over a long run at a large frontier
    // the total valid-sample count passes what an int accumulator holds.
    long long cntAll = thrust::reduce(graph_.d_validCounterArray_.begin(), graph_.d_validCounterArray_.end(), (long long)0);
    float minAll = thrust::reduce(d_minCostsR1_.begin(), d_minCostsR1_.end(), MAX_FLOAT, thrust::minimum<float>());
    h_costScale_ = (cntAll > 0 && minAll < MAX_FLOAT) ? (sumAll / (float)cntAll - minAll) : 0.0f;

    // --- v3: distMax, the cost histogram's anchor. The largest distance any candidate this
    // iteration can have, computed BEFORE pass 1 launches so the bucket map is fixed for the whole
    // pass and no third measuring pass is needed. Same cost class as the three reductions above. ---
    float spreadMax = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(d_minCostsR1_.begin(), d_maxCostsR1_.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(d_minCostsR1_.end(), d_maxCostsR1_.end())),
      CountingStars_SpreadOf(), 0.0f, thrust::maximum<float>());
    // Divided by costScale exactly as pass 1 divides a candidate's raw gap, INCLUDING the fallback,
    // so distMax and the distances it buckets are always in the same units.
    h_distMax_ = (h_costScale_ > 0.0f) ? (spreadMax / h_costScale_) : spreadMax;

    h_optimalCount_    = 0;
    h_ordCutoff_       = 0;
    h_pBoundary_       = 0.0f;
    h_costCutoff_      = 0;
    h_pCostBoundary_   = 0.0f;
    h_costCutoffDist_  = 0.0f;
    h_reactCutoff_     = 0;
    h_pReactBoundary_  = 0.0f;
    h_reactCutoffDist_ = 0.0f;
    h_dormantCount_    = 0;
    for(int i = 0; i < CS_HIST_SIZE; i++) h_acceptHistogram_[i] = 0;

    // ================================================================================
    // v3.1: MEASURE THE DORMANT TREE, alongside accept pass 1 rather than after it.
    //
    // Every input this needs is settled the moment propagate returned -- frontier[0, treeSize) is
    // not written again until Part A/B, minCostsR1 only by propagate, costScale and distMax are the
    // host scalars just solved above. So it votes into the SAME histogram buffer and its result
    // comes back in the SAME synchronising memcpy the accept passes already pay for. A third
    // selection signal, zero extra round trips.
    //
    // Launched OUTSIDE the `candidates > 0` guard below on purpose: the dormant tree exists and
    // wants reactivating whether or not this iteration produced any candidates.
    // ================================================================================
    if(h_treeSize_ > 0)
        {
            CountingStars_reactScan_kernel<<<iDivUp(h_treeSize_, h_blockSize_), h_blockSize_>>>(
              h_treeSize_, d_frontier_ptr_, d_goalSet_ptr_,
              d_treeXR1s_ptr_, d_treeSampleCosts_ptr_, d_minCostsR1_ptr_,
              d_bestNodeIdxPerR1_ptr_, h_costScale_, h_distMax_,
              d_reactEligible_ptr_, d_acceptHistogram_ptr_);
        }

    // ================================================================================
    // THE ADMISSION DECISION, IN TWO PASSES. Guard the launches: iDivUp(0, block) is 0 blocks,
    // which is cudaErrorInvalidConfiguration.
    // ================================================================================
    if(h_frontierNextSize_ > 0)
        {
            // --- Pass 1: measure. Fills candDistance, the ordinality histogram and the optimal
            // count; stamps no door, because the cutoff is not known yet. ---
            CountingStars_acceptPass1_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
              d_activeFrontierIdxs_ptr_, h_frontierNextSize_,
              d_minCostsR1_ptr_, d_frontierNextXR1s_ptr_, d_unexploredSampleCosts_ptr_,
              d_regionNodeCount_ptr_, h_costScale_, h_distMax_,
              d_candDistance_ptr_, d_acceptHistogram_ptr_);

            // STILL ONE SYNCHRONISING COPY, and that is what the shared buffer buys: both
            // histograms and the optimal count come back together. This stall sits mid-iteration
            // between the accept passes and serialises everything behind it, so v3's second
            // selection signal had to cost zero extra round trips.
            cudaMemcpy(h_acceptHistogram_, d_acceptHistogram_ptr_, CS_HIST_SIZE * sizeof(int),
                       cudaMemcpyDeviceToHost);
            h_optimalCount_ = (uint)h_acceptHistogram_[CS_HIST_OPT_SLOT];

            // --- SOLVE BOTH CUTOFFS. See csSolveCutoff for the scan and why no rank is needed.
            //
            // THE SHARES ARE OF B ITSELF, NOT OF (B - optimalCount). v2 handed freshness a share of
            // what the optimal door had left, which made the freshness door's size depend on a count
            // it has nothing to do with. v3's three fractions are fixed and the uncapped optimal
            // door spends on top of them -- so the frontier is optimalCount + B*(explore + cost) +
            // guarantee + B*react, and the overshoot is deliberate and measurable rather than
            // absorbed silently.
            //
            // At frac = 0 the solve returns cutoff 0 / pBoundary 0 and the door admits nothing,
            // which is what makes the sweep's 0 points real ablation arms. ---
            csSolveCutoff(h_acceptHistogram_ + CS_HIST_ORD_BASE, CS_ORD_BUCKETS,
                          h_exploreFrac_ * float(h_goalFrontierSize_), h_ordCutoff_, h_pBoundary_);
            csSolveCutoff(h_acceptHistogram_ + CS_HIST_COST_BASE, CS_COST_BUCKETS,
                          h_costFrac_ * float(h_goalFrontierSize_), h_costCutoff_, h_pCostBoundary_);

            // The cutoff BUCKET is only meaningful against the distMax that produced it, and distMax
            // moves every iteration. Invert the bucket map here so the CSV carries the distance
            // threshold as well, which is the column that is comparable across a run. Saturation
            // (nothing was cheap enough to bound) reports distMax itself.
            h_costCutoffDist_ = (h_costCutoff_ >= CS_COST_BUCKETS)
                                  ? h_distMax_
                                  : h_distMax_ * exp2f((float(h_costCutoff_) - float(CS_COST_BUCKETS - 1))
                                                       / CS_COST_LOG_SCALE);

            // --- Pass 2: decide. The only door writer, and the only place frontierNext is cleared. ---
            CountingStars_acceptPass2_kernel<<<iDivUp(h_frontierNextSize_, h_blockSize_), h_blockSize_>>>(
              d_activeFrontierIdxs_ptr_, h_frontierNextSize_,
              d_frontierNextXR1s_ptr_, d_regionNodeCount_ptr_,
              d_candDistance_ptr_, d_frontierNext_ptr_, d_candDoor_ptr_,
              d_regionCovered_ptr_, d_randomSeeds_ptr_,
              h_ordCutoff_, h_pBoundary_,
              h_costCutoff_, h_pCostBoundary_, h_distMax_,
              d_doorCounts_ptr_);
        }

    // ================================================================================
    // v3.1: SOLVE THE REACTIVATION CUTOFF. Outside the candidate guard, because the scan above ran
    // unconditionally -- an iteration that produced no candidates still has a dormant tree to
    // reactivate, and skipping this would leave the cutoff at its cleared 0 and switch the cost arm
    // off for that iteration.
    //
    // ITS BUDGET IS THE WHOLE react_frac * B. v2 and v3 spent this share on a UNIFORM draw; the
    // entire share now goes to the cheapest dormant nodes, and the completeness floor
    // (h_reactFloor_) is added on top rather than carved out of it.
    //
    // The histogram was copied back above with the candidate ones when there were candidates. When
    // there were none, nothing has copied it yet -- so do it here, and note this branch costs a
    // synchronising memcpy only on an iteration that had no candidates at all.
    // ================================================================================
    if(h_frontierNextSize_ == 0 && h_treeSize_ > 0)
        cudaMemcpy(h_acceptHistogram_, d_acceptHistogram_ptr_, CS_HIST_SIZE * sizeof(int),
                   cudaMemcpyDeviceToHost);

    h_dormantCount_ = (uint)h_acceptHistogram_[CS_HIST_DORMANT_SLOT];
    csSolveCutoff(h_acceptHistogram_ + CS_HIST_REACT_BASE, CS_COST_BUCKETS,
                  h_reactFrac_ * float(h_goalFrontierSize_), h_reactCutoff_, h_pReactBoundary_);
    // The readable form -- the bucket index only means anything against the distMax that produced
    // it. Saturation (nothing was cheap enough to bound the budget) reports distMax itself.
    h_reactCutoffDist_ = (h_reactCutoff_ >= CS_COST_BUCKETS)
                           ? h_distMax_
                           : h_distMax_ * exp2f((float(h_reactCutoff_) - float(CS_COST_BUCKETS - 1))
                                                / CS_COST_LOG_SCALE);

    // --- Re-scan after the accept passes. The trailing-element correction matters: a candidate
    // landing in the last slot is otherwise dropped from the count. ---
    thrust::exclusive_scan(d_frontierNext_.begin(), d_frontierNext_.end(), d_frontierScanIdx_.begin(), 0, thrust::plus<uint>());
    h_frontierNextSize_ = d_frontierScanIdx_[MAX_TREE_SIZE - 1];
    (d_frontierNext_[MAX_TREE_SIZE - 1]) ? ++h_frontierNextSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_frontierNext_ptr_, d_frontierScanIdx_ptr_, d_activeFrontierIdxs_ptr_);

    // --- Check tree capacity ---
    if(h_treeSize_ + h_frontierNextSize_ >= MAX_TREE_SIZE)
        {
            h_propIterations_ = 0;
            return;
        }

    // ================================================================================
    // NO REACTIVATION PROBABILITY TO SOLVE ANY MORE. v3 computed
    // p = react_frac * B / treeSize for a uniform draw; v3.1 spends that whole share through the
    // cost cutoff solved above, which gets its count from the histogram rather than from a
    // probability, and adds the flat completeness floor h_reactFloor_ on top.
    //
    // The frontier is therefore
    //
    //     optimalCount + B*(explore_frac + cost_frac) + guaranteed + B*react_frac
    //                  + reactFloor*dormantCount
    //
    // in expectation -- over B by the two uncapped doors, deliberately, and by an amount the CSV
    // reports directly as budget_used against goal_frontier_size. The floor's contribution is ~30
    // nodes and is not meant to be visible in that number.
    // ================================================================================

    // NO DESIGN-BUDGET SPLIT ANY MORE. The old rule divided `maxBlocks * B` between the optimal
    // door and everyone else, and in the nominal case the divisor came out at B - optimalCount so
    // EVERY frontier node received maxBlocks -- it concentrated nothing while the ancestors were
    // concentrating 15-to-1. Fan-out is now region-keyed and decided per node in Part A; the only
    // block constraint left on the host is blockCeiling / blockScale in propagateFrontier, which is
    // the buffer bound and a different thing entirely.

    // --- Update Frontier. Part A inserts and stamps blocks; Part B fills the remainder. ---
    CountingStars_updateFrontier_kernel<<<iDivUp(h_frontierNextSize_ + h_treeSize_, h_blockSize_), h_blockSize_>>>(
      d_frontier_ptr_, d_frontierNext_ptr_, d_activeFrontierIdxs_ptr_, h_frontierNextSize_, d_goalSample_ptr_, h_treeSize_,
      d_unexploredSamples_ptr_, d_treeSamples_ptr_, d_unexploredSamplesParentIdxs_ptr_, d_treeSamplesParentIdxs_ptr_,
      d_treeSampleCosts_ptr_, d_randomSeeds_ptr_,
      d_candDoor_ptr_, d_nodeDoor_ptr_, d_nodeBlocks_ptr_,
      d_regionNodeCount_ptr_, d_regionCovered_ptr_, graph_.d_validCounterArray_ptr_,
      d_minCostsR1_ptr_, d_treeXR1s_ptr_, d_frontierNextXR1s_ptr_, d_bestNodeIdxPerR1_ptr_,
      d_minCost_ptr_, d_unexploredSampleCosts_ptr_, d_goalSet_ptr_,
      d_iterations_ptr_, h_itr_,
      d_reactEligible_ptr_, h_costScale_, h_distMax_,
      h_reactCutoff_, h_pReactBoundary_, h_reactFloor_,
      h_maxBlocks_,
      d_doorCounts_ptr_);

    // --- Read back the door counts. One memcpy for the whole "what built this tree" answer. ---
    cudaMemcpy(h_doorCounts_, d_doorCounts_ptr_, CS_NUM_DOOR_SLOTS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    h_admittedExplore_  = (uint)h_doorCounts_[CS_SLOT_EXPLORE];
    h_admittedCost_     = (uint)h_doorCounts_[CS_SLOT_COST];
    h_admittedCostDist_ = (uint)h_doorCounts_[CS_SLOT_COSTDIST];
    h_admittedBoth_     = (uint)h_doorCounts_[CS_SLOT_BOTH];
    h_reactivated_      = (uint)h_doorCounts_[CS_SLOT_REACT];
    h_reactivatedBest_  = (uint)h_doorCounts_[CS_SLOT_BEST];
    h_reactivatedCost_  = (uint)h_doorCounts_[CS_SLOT_REACT_COST];
    // EXPLORE and COSTDIST overlap by construction, so the identity that checks all four counters
    // against the compaction is
    //
    //     h_frontierNextSize_ == optimal + explore + costdist - both
    //
    // and it holds exactly, because pass 2 is the only door writer and every candidate takes exactly
    // one of its branches.
    cudaMemcpy(&h_touchedR2_, d_touchedR2Count_ptr_, sizeof(uint), cudaMemcpyDeviceToHost);

    // What the doors actually committed, from the REALISED counts rather than the plan: admissions
    // plus the guaranteed and drawn reactivations that survived Part B's skips. Read against
    // goal_frontier_size -- this is the claim the whole design rests on, in one column.
    // v3.1: Part B has THREE arms now, so all three count toward what the doors committed. The
    // benchmark's independent thrust::count of frontier bits among the pre-existing tree must equal
    // reactivatedBest + reactivatedCost + reactivated -- a free check on every arm, since one side
    // is a host scan and the other is device atomics.
    h_budgetUsed_ = h_frontierNextSize_ + h_reactivatedBest_ + h_reactivatedCost_ + h_reactivated_;

    // --- Sync goal state ---
    cudaMemcpy(&h_minCost_, d_minCost_ptr_, sizeof(float), cudaMemcpyDeviceToHost);

    // --- Update Tree Size ---
    h_treeSize_ += h_frontierNextSize_;
}

/***************************/
/* GET CONTROL PATH TO GOAL */
/***************************/
void CountingStars::getControlPathToGoal()
{
    thrust::exclusive_scan(d_goalSet_.begin(), d_goalSet_.end(), d_goalSetScanIdx_.begin(), 0, thrust::plus<uint>());
    h_solSetSize_ = d_goalSetScanIdx_[MAX_TREE_SIZE - 1];
    (d_goalSet_[MAX_TREE_SIZE - 1]) ? ++h_solSetSize_ : 0;
    findInd<<<h_gridSize_, h_blockSize_>>>(MAX_TREE_SIZE, d_goalSet_ptr_, d_goalSetScanIdx_ptr_, d_goalSetIdxs_ptr_);

    if(h_solSetSize_ == 0) return;

    CountingStars_getControlPathToGoal_kernel<<<iDivUp(h_solSetSize_, h_blockSize_), h_blockSize_>>>(
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
__global__ void CountingStars_getControlPathToGoal_kernel(float* controlPathsToGoal, float* treeSamples,
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

void CountingStars::writeExecutionTimeToCSV(double time)
{
    std::ostringstream filename;
    std::filesystem::create_directories("Data");
    std::filesystem::create_directories("Data/ExecutionTime");
    filename.str("");
    filename << "Data/ExecutionTime/executionTime.csv";
    writeValueToCSV(time, filename.str());
}
