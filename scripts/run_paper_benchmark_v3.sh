#!/bin/bash
# =============================================================================
# Paper Benchmark v3 -- a duplicate of run_paper_benchmark_v2.sh / paper_benchmark_v2.cu, NOT a
# modification of it (that file is untouched), with three changes:
#
#   1. MAX_TREE_SIZE 3,000,000 -> 1,000,000 (write_config()'s three model arms, below).
#   2. MAX_TIME_MS (the per-run wall-clock cap, in paper_benchmark_v3.cu) 10s -> 3s.
#   3. ALL THREE DISCRETIZATIONS (large/fine/tiny) run per model now, not v2's one -- restored as a
#      full axis (MODEL_DELTA_* below), each written to its OWN top-level output subfolder
#      ("COARSE"/"FINE"/"TINY", see delta_folder_name() below and paper_benchmark_v3.cu's new
#      deltaFolder CLI arg) rather than being distinguished only by filename TAG as before.
#
# CountingStars' CS_RAMP_FILL_ITERS (paper_benchmark_v3.cu) is rescaled accordingly: it was
# empirically calibrated at 700 for MAX_TREE_SIZE=3,000,000 and the 10s cap; v3 uses
# 700 * (3s/10s) = 210 -- see that constant's own comment in paper_benchmark_v3.cu for the full
# derivation and its caveats (a linear estimate, not independently re-benchmarked at v3's own
# tree size/cap).
#
# Everything else -- the four planners, all three vehicle models (Double Integrator/Dubins
# Airplane/Quad), 30 runs/planner, 4 environments, 2 cost metrics, per-planner process isolation
# wrapped in an external `timeout` -- is identical to v2. See run_paper_benchmark_v2.sh's own
# header (unchanged, not duplicated here) for the full MODEL 1->2->3 derivation history, the
# Dubins dimension-breakdown correction, and the CountingStars v2->v3.5 algorithm design history
# (an unrelated "v2/v3" -- that's the CountingStars algorithm's own version numbering inside
# CountingStars.cuh, not this script's).
#
# ALL THREE VEHICLE MODELS x ALL THREE DISCRETIZATIONS THIS PASS (MODEL_DELTA_* below, 9 rows: 3
# models x 3 deltas), 30 RUNS PER PLANNER, and each planner isolated to its OWN PROCESS invocation
# wrapped in an external `timeout` (PLANNER_FLAGS/PLANNER_NAMES below) -- a true CUDA hang has no
# in-process recovery (see the KPAX buffer-overshoot history further below), so this is what
# actually lets a hang be detected, logged as a failure, and skipped past rather than freezing the
# whole sweep. Per (model, delta, environment, cost metric):
#   KPAX                                                    = 1 point  x 30 runs
#   KinoPaxPlus                                             = 1 point  x 30 runs
#   KinoPaxSTARTrue syclopCap 1.0 (no cap), ancestorPrune = 1
#                                                            = 1 point  x 30 runs
#   CountingStars   bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15, cost_frac 0.75,
#                   hopelessGuard PERMANENTLY ON (v3.5)      = 1 point  x 30 runs
#
# BOTH COST METRICS (length + effort), FOUR ENVIRONMENTS (empty, house, narrowPassage, zigzag --
# see ENV_NAMES below), one build per (model, delta, cost metric): 9 rows x 4 series x 30 runs x 4
# environments x 2 cost metrics = 8,640 runs total, from 18 compiled binaries, run as 288 separate
# (model, delta, cost, environment, planner) process invocations, each wrapped in its own timeout.
#
# The original grid-sweep version of this file (bufferSlope/bufferFloor grid, ancestorPrune {0,1})
# is countingstars_sweep.cu / run_countingstars_sweep.sh, unmodified by this file's existence.
#
# ============================================================================================
# WHAT CHANGED FROM v2, AND WHY THIS SWEEP EXISTS
#
# 1. B IS DERIVED, NOT SWEPT. v2's grid was goal_frontier_size {200, 2000, 6000, 10000} with no
#    derivation behind any of them. v3 made the planner compute it instead:
#
#        B = floor(fill_frac * MAX_TREE_SIZE / MAX_ITER)      -- v3, ONE VALUE PER RUN
#
#    -- "the frontier size that fills the tree exactly at MAX_ITER", scaled by fill_frac. B rides
#    into every CSV as the goal_frontier_size column rather than being re-derived by the plot
#    script. SEE ITEM 5 BELOW: v3.2 replaces the single fill_frac with a per-iteration ramp, so
#    this formula is no longer what the planner actually runs -- kept here as the ancestor the
#    ramp's bufferSlope = 0 case reproduces exactly.
#
# 2. THERE IS A NEW DOOR: CHEAPEST. v2 could admit a node for being THE cheapest in its region
#    (distance 0) and for nothing else -- a candidate one part in 1e6 above its region's minimum was
#    treated exactly like one at ten times the minimum. v3 selects the top cost_frac * B smallest
#    cost distances with a HISTOGRAM plus an exclusive scan plus a boundary roll, exactly as the
#    freshness door already selects the top explore_frac * B smallest ordinalities. (An earlier
#    branch did this with a sort over the distances and kept breaking. Nothing here needs a rank.)
#
#    Its buckets are LOG and anchored at an exactly computed dist_max, because a distance is
#    (cost - regionMin)/costScale and therefore piles up near 0 with a long tail; linear buckets let
#    one outlier region compress the whole real distribution into bucket 0, where the boundary roll
#    degrades the door to a uniform draw.
#
# 3. THE BUDGET SPLITS THREE WAYS BY FIXED FRACTION, not "one share plus a remainder":
#    explore_frac to freshness, cost_frac to cheapness (which OPTIMAL candidates now always compete
#    inside too, permanently -- v3.4), and react_frac = 1 - explore - cost to reactivation. Nothing
#    on the candidate side is uncapped any more; the region-best GUARANTEE that used to be uncapped
#    on the dormant-node side is gone permanently too, folded into the reactivation budget -- see
#    CS_DOORBIT_GUAR.
#
# 4. v3.1: REACTIVATION IS COST-SELECTIVE, and that is the change this pass is really testing.
#    v2 and v3 spent react_frac * B on a UNIFORM draw over the tree. CleanCost weights the same arm
#    by costProbExpGlobal, and that is the one cost mechanism this line did not have -- the volumes
#    were already comparable, so it was selectivity rather than throughput. The whole share now goes
#    to the CHEAPEST dormant nodes, chosen by a third histogram (over dormant tree nodes) that rides
#    in the same buffer and the same memcpy as the two candidate ones, so it costs no extra
#    synchronisation.
#
#    WHY THAT ARM AND NOT ANOTHER: a cheaper route to the goal is built by deepening a cheap
#    INTERIOR branch, and Part B is the only thing that re-expands the interior -- new candidates
#    are the growing edge.
#
#    A SEPARATE COMPLETENESS FLOOR (react_floor, 1e-5) is added ON TOP, not carved out of the
#    budget, and it is a correctness constant rather than a knob. A node's cost distance has a fixed
#    numerator over a non-increasing region minimum, so it only ever GROWS: under a pure top-K a node
#    once above the cutoff can never come back and its whole subtree is unreachable. The floor
#    restores "expanded infinitely often in the limit". At 1e-5 over a 3e6-node tree it wakes ~30
#    nodes per iteration -- completeness in the limit, not reach inside one run.
#
# 5. v3.2: B BECOMES A RAMP. Sweep results under v3's constant B showed the standard tradeoff: a
#    small fill_frac found a first solution fast but converged to a worse final cost; a large one
#    was the reverse. Rather than pick one point on that tradeoff, B now varies OVER the run:
#
#        x         = itr / MAX_ITER                              (fraction of the run elapsed)
#        B_frac(x) = bufferSlope * x + bufferFloor
#        B(x)      = floor(B_frac(x) * MAX_TREE_SIZE / MAX_ITER)  -- RECOMPUTED EVERY ITERATION
#
#    bufferSlope = 0 REPRODUCES v3's CONSTANT B EXACTLY (B_frac(x) = bufferFloor for every x), so
#    that subgrid is a free, structural comparison against the old fixed-buffer design rather than
#    a separate baseline that has to be swept again. (Not on this pass's grid -- see below.)
#
#    explore_frac AND cost_frac were both swept in earlier passes that found the fixed point this
#    file runs (bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15, cost_frac 0.75 -- see
#    CS_BUFFER_SLOPE etc. in paper_benchmark_v2.cu) rather than sweeping them again.
#
#    B IS A PURE HOST SCALAR (read only inside updateFrontier(), never by propagateFrontier() or
#    any device kernel directly), so making it dynamic cost no device array, no new kernel, and no
#    new synchronisation -- it is one floating-point formula recomputed once per iteration.
#
# (bufferSlope, bufferFloor) = (0, 0) is NOT on this pass's grid (minimums are 1.0 and 0.3) -- it
# used to be the deepest ablation arm back when OPTIMAL and the region-best GUARANTEE were both
# uncapped regardless of B. Both are gone permanently now, so a B=0 point would mean something
# different (and less useful) than it used to; not worth re-deriving for this grid.
# ============================================================================================
#
# NOTHING ON THE CANDIDATE SIDE IS UNCAPPED ANY MORE (v3.4, permanent) -- the region-best GUARANTEE
# that used to also be uncapped on the dormant-node side is gone too. So B binds whenever the
# candidate pool for an iteration exceeds it, and is a soft target otherwise; the iteration where
# budget_used/goal_frontier_size crosses 1 is where time-to-first-solution gets decided.
#
# Read budget_used/goal_frontier_size as a CURVE over iterations rather than a single number: the
# iteration where it crosses 1 IS the measurement, and it should track a MOVING target now that B
# itself climbs over the run.
#
# READ IN THIS ORDER:
#   1. goal_frontier_size vs iteration, FIRST. Confirms the realized ramp actually matches
#      slope*x + floor before reading anything else that depends on B.
#   2. frontier_repeat_size / frontier_size. The realised mean rep, which should sit near 1 with a
#      small excess from thin regions and the both-doors boost, not near 4.
#   3. budget_used / goal_frontier_size, as a curve. See above -- a moving target on both sides.
#   4. reactivated_cost against reactivated_count. The cost arm should carry essentially ALL of
#      Part B's volume (~ react_frac * B; reactivated_best should read exactly 0 -- its arm is gone
#      permanently); reactivated_count is the completeness FLOOR alone, ~ react_floor *
#      dormant_count, so ~30 nodes. Large there means the floor is doing
#      reach work it was not sized for. And react_cutoff_dist against dist_max says whether the arm
#      is actually selecting: pinned at 1 means the budget exceeds the population below the anchor
#      and it is partly picking at random within the clamped tail.
#   5. optimal_count against admitted_cost. THE OPTIMAL-DOOR STARVATION SIGNAL: optimal_count is
#      pass 1's measured population at distance 0; admitted_cost is what pass 2 actually let
#      through the SAME cutoff CHEAPEST uses. Their gap is how many optimal candidates the budget
#      did not have room for at this (bufferSlope, bufferFloor, explore_frac, cost_frac) point.
#   6. admitted_costdist against admitted_explore and admitted_cost. THE CHEAPEST DOOR'S ACTUAL
#      SHARE. Pinned at 0 means the cutoff solve is degenerate; equal to cost_frac * B every
#      iteration means it is working exactly as designed. The two selection doors OVERLAP, so
#      admitted_both/admitted_opt_fresh_both are what make the counts add back up:
#      admitted == admitted_cost + admitted_explore + admitted_costdist + admitted_floor
#                - admitted_opt_fresh_both - admitted_both.
#   7. cost_cutoff_dist against dist_max. The direct read on whether the log bucket map has the right
#      shape. A collapse toward dist_max / 2^21 means everything is landing in bucket 0 and the door
#      has degraded to a uniform draw among near-optimal candidates -- the signal to switch
#      csCostBucket to a linear map, which is a one-line change in the header.
#   8. ord_cutoff over the run. Rising means regions are filling and freshness is getting scarce,
#      which is expected. Pinned at 0 means explore_frac is doing nothing; pinned at 256 means the
#      whole candidate pool is fresh enough and explore_frac is not binding either.
#   9. First-solution time and cost, final cost, and success rate -- the headline comparison this
#      tool exists to produce reliably across all four planners and environments.
#
# ALL THREE MODELS RUN THE FULL COMPARISON -- NOT --only-kinopaxplus at any of them. Conclusions at
# one model do not automatically hold at the others, so all three need the full comparison, not
# KinoPaxPlus alone.
#
# Runs on all four environments (empty, house, narrowPassage, zigzag -- see ENV_NAMES below), each
# written to its own subfolder under Data/Benchmarks/PaperBenchmarkV3/<COARSE|FINE|TINY>/<env>/ --
# delta is a new top-level split in v3 (see delta_folder_name() above), environment nested inside.
#
# MODEL, NUM_R1_REGIONS, and COST_MODE are all COMPILE-TIME (config.h, and MODEL/a #if inside
# edgeCost), so none can vary within one binary. This script therefore borrows
# run_delta_benchmark.sh's build-cache pattern: write config.h and build once per (model, delta,
# cost metric) row, caching each binary under a suffixed name, then run them in a second pass. All
# three labels ride into every output filename via the argv[1] TAG (m1_large_length /
# m2_fine_effort / m3_tiny_length / ... -- see the model x delta axis further below), which is what
# keeps different (model, delta)'s per-run CSVs from colliding within an environment folder (see
# the BUILD phase's own comment for why that matters) -- on top of that, delta ALSO now splits the
# output at the top level into COARSE/FINE/TINY subfolders (see delta_folder_name() above).
#
# It builds ONLY the PaperBenchmarkV3 target. That still compiles KPAX_lib, which is the
# monolithic library holding every planner in the repo -- so warnings from unrelated sources
# (ReKino and friends) scroll past on every build. They are pre-existing and unavoidable without
# splitting the library.
#
# ALL THREE MODELS x ALL THREE DISCRETIZATIONS RUN THIS PASS -- v2 ran each model at exactly ONE
# discretization ("large"); v3 restores the full large/fine/tiny sweep per model. See the model x
# delta axis (MODEL_DELTA_MODEL_IDS/MODEL_DELTA_LABELS/MODEL_DELTA_W_R1S/MODEL_DELTA_C_R1S/
# MODEL_DELTA_V_R1S) further above for the derivation of every one of the 9 points.
#
# NUM_R1_REGIONS' FORMULA SHAPE DIFFERS BY MODEL (Double Integrator: W^3*V^3, no C term; Dubins
# Airplane: W^3*C^2*V; Quad: W^3*C^3*V^3) -- see the model axis further above for the actual numbers
# and their derivation, which is why all three shapes are handled explicitly (compute_regions()
# above, and write_config()'s three case arms) rather than one formula being re-derived per model.
#
# "large"/"fine"/"tiny" name the CELL, not the count, and the count differs by model -- see the
# model x delta axis further above for the full 9-point table. Watch the larger points ("fine" and
# especially "tiny") for the per-region arrays -- every NUM_R1_REGIONS allocation and every
# full-array fill scales with this, and graph_.updateVertices() runs a kernel over all of them with
# 64 sub-vertex reads each.
#
# Original config.h is backed up and restored on exit/error.
#
# There is no automated cross-check script for this file's triplet (paper_benchmark_v3.cu /
# run_paper_benchmark_v3.sh / a v3-aware plot script, not yet created) the way
# scripts/cross_check_countingstars_grid.py covers the sweep -- there's nothing to sweep-check any
# more (every axis here is a single fixed point per row), but a label typo in any file would still
# silently show up as loadRuns() (in whatever plot script eventually reads this) reporting "0 runs"
# for a series, not an error.
#
# Usage:
#   cd scripts && bash run_paper_benchmark_v3.sh
#   cd scripts && bash run_paper_benchmark_v3.sh --skip-build   # run only (cached binaries)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"

# Model x delta axis (v3: RESTORED to a full 3-point large/fine/tiny sweep per model -- v2's single
# fixed point per model, MODEL_DELTA_LABELS=("large" "large" "large"), is what this replaces).
# Flattened into ONE set of parallel arrays over all 9 (model, delta) rows -- bash has no clean 2D
# array, so this extends the same flat-array idiom MODEL_IDS/MODEL_NAMES already used rather than
# introducing per-model sub-arrays or a lookup function. MODEL_NAMES is still a plain 3-entry bash
# array (0-indexed: DoubleIntegrator/DubinsAirplane/Quad at 0/1/2, NOT 1/2/3), so every lookup
# below uses `MODEL_NAMES[$((MODEL-1))]` -- MODEL_DELTA_MODEL_IDS maps each of the 9 rows to its
# raw model id (1/2/3, for write_config()'s `case "$MODEL"` and that name lookup).
#
# Numbers verified against three independent sources that all agree: run_countingstars_sweep.sh's
# own live large/fine/tiny arrays (Model 1), the "Dubins Airplane deltas" and "OLD three-delta,
# Quad-only sweep" blocks this file used to carry commented out here (Models 2/3, now the ACTIVE
# source instead of dead history), and run_paper_benchmark.sh's own matching Model 1/2 arrays.
# Quad's `tiny` uses the SAFE, REDUCED point (6,2,6 -> 373,248), NOT the old (7,2,6 -> 592,704) one
# -- see the KPAX buffer-overshoot warning immediately below, which is NOW LIVE (not just history)
# since v3 actually runs Quad tiny, unlike v2 which had backed off to `large` only:
#
#   Model 1 (Double Integrator, W^3*V^3):     large W7/V3->9,261   fine W16/V4->262,144  tiny W14/V6->592,704
#   Model 2 (Dubins Airplane,  W^3*C^2*V):    large W7/C3/V3->9,261 fine W16/C4/V4->262,144 tiny W14/C6/V6->592,704
#   Model 3 (Quad,             W^3*C^3*V^3):  large W5/C2/V2->8,000 fine W6/C2/V5->216,000  tiny W6/C2/V6->373,248
#
# KPAX BUFFER-OVERSHOOT WARNING (LIVE FOR v3): KPAX has a confirmed buffer-overshoot bug
# (h_treeSize_ exceeding MAX_TREE_SIZE via propagateFrontier()'s h_propIterations_==0 edge case,
# KPAX.cu:254-271) that hit a hang under Dubins Airplane and a cudaErrorIllegalAddress crash under
# Quad on Jetson, specifically at Quad's OLD tiny (7,2,6 -> 592,704, NOT used here). The REDUCED
# tiny (6,2,6 -> 373,248) used above was chosen to stay far enough from MAX_TREE_SIZE to avoid
# triggering it; paper_benchmark_v3.cu's skipKPAXThisDelta hard-skip is kept commented as the
# fallback if this doesn't hold, and the per-planner `timeout`-wrapped process isolation below would
# catch and log a recurrence regardless, rather than hanging the whole sweep.
#
# Every other model/delta point above sits at or below 592,704 regions too (all previously
# confirmed clean at "large" and "fine" across KPAX/KinoPaxPlus/CountingStars/KinoPaxSTARTrue), so
# this is the one row worth watching on a first v3 run.

MODEL_NAMES=("DoubleIntegrator" "DubinsAirplane" "Quad")   # indexed by raw model id (1/2/3)

MODEL_DELTA_MODEL_IDS=(1     1     1     2     2     2     3     3     3)
MODEL_DELTA_LABELS=(large fine  tiny  large fine  tiny  large fine  tiny)
MODEL_DELTA_W_R1S=(7     16    14    7     16    14    5     6     6)
MODEL_DELTA_C_R1S=(1     1     1     3     4     6     2     2     2)   # inert for Model 1 (C_DIM 0)
MODEL_DELTA_V_R1S=(3     4     6     3     4     6     2     5     6)

# Output-folder name per delta label -- NEW IN v3: each discretization now gets its own top-level
# subfolder (paper_benchmark_v3.cu's new deltaFolder CLI arg), since three deltas now share one
# environment's output space instead of v2's one.
delta_folder_name() {
    case "$1" in
        large) echo "COARSE" ;;
        fine)  echo "FINE" ;;
        tiny)  echo "TINY" ;;
        *) echo "ERROR: delta_folder_name: unknown delta label '$1'" >&2; exit 1 ;;
    esac
}

# Per-model NUM_R1_REGIONS formula -- shape differs by model (see write_config()'s three arms):
#   Model 1 (C_DIM 0): W_R1^3 * V_R1^3               (no C_R1 term)
#   Model 2 (C_DIM 2): W_R1^3 * C_R1^2 * V_R1
#   Model 3 (C_DIM 3): W_R1^3 * C_R1^3 * V_R1^3
compute_regions() {
    local model=$1 w=$2 c=$3 v=$4
    case "$model" in
        1) echo $(( w**3 * v**3 )) ;;
        2) echo $(( w**3 * c**2 * v )) ;;
        3) echo $(( w**3 * c**3 * v**3 )) ;;
        *) echo "ERROR: compute_regions: unknown MODEL '$model'" >&2; exit 1 ;;
    esac
}

# Planner axis (NEW this pass): one process invocation per planner now, instead of one process
# running all four sequentially -- bounds a hang's blast radius to just the one planner that hung
# (see paper_benchmark_v2.cu's main() for the matching --only-* flags), and lets the RUN phase
# below wrap each invocation in an external `timeout`. Order matches today's dispatch order.
PLANNER_FLAGS=("--only-kpax" "--only-kinopaxplus" "--only-kinopaxstartrue" "--only-countingstars")
PLANNER_NAMES=("KPAX"        "KinoPaxPlus"        "KinoPaxSTARTrue"        "CountingStars")

# Cost metric axis: label + COST_MODE (0 = workspace distance, 1 = control effort). BOTH THIS
# PASS -- doubles the build count (one binary per model x cost metric) and the run count.
COST_LABELS=("length" "effort")
COST_MODES=(0 1)
# COST_LABELS=("length")
# COST_MODES=(0)

# Environments -- obstacle CSVs are still authored in [0,1]^3, but Quad's W_SIZE is 100, not 1
# (see the MODEL 3 switch above). paper_benchmark_v2.cu scales every obstacle coordinate and the
# start/goal points by W_SIZE at load time now (a no-op for every other model, whose W_SIZE is
# 1.0), so this still points at the same CSVs unchanged. Each environment gets its own output
# subfolder.
# SCOPE: empty only this pass -- CHANGED FROM zigzag. paper_benchmark.cu's tiny/empty run froze,
# `empty` was pulled out of that suite as a diagnostic, and the freeze PERSISTED on the remaining
# environments (house/narrowPassage/zigzag) -- so `empty` alone is not the (sole) trigger there.
# But this harness itself has NEVER run `empty` -- every "confirmed clean at tiny" claim so far
# (KPAX, KinoPaxPlus, CountingStars, KinoPaxSTARTrue, and now the hopeless guard) was measured on
# zigzag only, under Model 1. Pointing this harness at `empty` instead tests the ONE combination
# still never isolated this way: `empty` itself, on Model 1, in this simpler one-planner-at-a-time
# harness (which paper_benchmark.cu's own five/six/eight-series-at-once run is not). If tiny stays
# clean here, that argues against `empty` (under Model 1) as a factor at all, pointing harder at
# Model 2 (Dubins Airplane) -- which this harness still does not run -- as the real new variable.
# ALL FOUR ENVIRONMENTS THIS PASS -- the one deliberate axis change from countingstars_sweep.cu (see
# file header). narrowPassage's wall sits at x in [0.3, 0.5] spanning all z, split by a gap at y in
# [0.49, 0.51] -- 0.02 wide against an agent diameter of 0.01 (AGENT_RADIUS 0.005). The benchmark's
# start (0.1, 0.08, 0.05) and goal (0.8, 0.95, 0.9) are clear of every environment's obstacles and on
# opposite sides of narrowPassage's wall, so no endpoint change is needed -- but expect low success
# rates there, and read the success-rate subplot alongside the cost bars (unsolved runs are dropped
# from the cost mean, so a config that solved once cheaply can look best).
ENV_NAMES=("empty" "house" "narrowPassage" "zigzag")
ENV_OBSTACLES=(
    "../include/config/obstacles/empty/obstacles.csv"
    "../include/config/obstacles/house/obstacles.csv"
    "../include/config/obstacles/narrowPassage/obstacles.csv"
    "../include/config/obstacles/zigzag/obstacles.csv"
)

# --- Parse arguments ---
SKIP_BUILD=false
for arg in "$@"; do
    if [ "$arg" = "--skip-build" ]; then
        SKIP_BUILD=true
    fi
done

# --- Auto-detect compilers (cluster has gcc-12, Jetson uses default gcc) ---
CMAKE_COMPILER_FLAGS=""
if command -v gcc-12 &> /dev/null; then
    echo "Detected gcc-12 (cluster environment)"
    CMAKE_COMPILER_FLAGS="-DCMAKE_C_COMPILER=$(which gcc-12) -DCMAKE_CXX_COMPILER=$(which g++-12) -DCMAKE_CUDA_HOST_COMPILER=$(which g++-12)"
else
    echo "Using default system compilers (Jetson/local environment)"
fi

# Restore config.h on exit
cleanup() {
    echo ""
    echo "Restoring original config.h..."
    if [ -f "$CONFIG_BACKUP" ]; then
        cp "$CONFIG_BACKUP" "$CONFIG_FILE"
        rm -f "$CONFIG_BACKUP"
        echo "Config restored."
    fi
}
trap cleanup EXIT ERR INT TERM

# --- Back up original config.h ---
echo "Backing up config.h..."
cp "$CONFIG_FILE" "$CONFIG_BACKUP"

# --- Ensure build directory exists ---
mkdir -p "$BUILD_DIR"

# Function to write config.h for the given MODEL (1=Double Integrator, 2=Dubins Airplane,
# 3=Quad) at the given discretization. One `case` arm per model -- kept as three FULL blocks
# (matching this repo's "every model's constants stay physically present, active or not" convention
# seen in config.h itself) rather than one DRY/parameterized block, so each arm can be diffed
# directly against its own proven source:
#   Model 1 -- copied from scripts/run_countingstars_sweep.sh's own write_config() (that sweep is
#              this harness's proven-reliable ancestor, run clean at exactly this discretization).
#   Model 2 -- recovered from THIS FILE's own git history (commit 5a09775, "model 2" -- the real
#              config this script actually compiled and ran before switching to Quad), not
#              reconstructed from config.h's separately-maintained historical comment block (which,
#              per its own header, was never actually built from by this script).
#   Model 3 -- unchanged from today's active (Quad) content.
write_config() {
    local MODEL=$1
    local W_R1=$2
    local C_R1=$3
    local V_R1=$4
    local COST_MODE=$5
    case "$MODEL" in
    1)
        cat > "$CONFIG_FILE" << CONFIGEOF
#pragma once
/***************************/
/* 6D DOUBLE INTEGRATOR    */
/***************************/
#define MODEL 1
#define COST_MODE ${COST_MODE}  // path cost: 1 = control effort ((ax^2+ay^2+az^2)*dt), 0 = workspace distance
#define MAX_TREE_SIZE 1000000
#define MAX_FLOAT 1e38f
#define MAX_SOL_SET_SIZE 500
#define MAX_ITER 1000
#define MAX_ITER_REKINO 20000
#define STEP_SIZE 0.1f
#define MAX_PROPAGATION_DURATION 10
#define ACCEPT 0.99f
#define AGENT_RADIUS 0.005f
#define GOAL_THRESH 0.05f
#define STATE_DIM 6
#define CONTROL_DIM 3
#define SAMPLE_DIM (STATE_DIM + CONTROL_DIM + 1)
#define W_DIM 3
#define C_DIM 0
#define V_DIM 3
#define W_MIN 0.0f
#define W_MAX 1.0f
#define W_SIZE 1.0f
#define C_MIN -M_PI
#define C_MAX M_PI
#define V_MIN -0.3f
#define V_MAX 0.3f
#define A_MIN -0.2f
#define A_MAX 0.2f
#define W_R1_LENGTH ${W_R1}
#define C_R1_LENGTH ${C_R1}
#define V_R1_LENGTH ${V_R1}
#define W_R2_LENGTH 2
#define C_R2_LENGTH 1
#define V_R2_LENGTH 2
#define W_R1_SIZE ((W_MAX - W_MIN) / W_R1_LENGTH)
#define C_R1_SIZE ((C_MAX - C_MIN) / C_R1_LENGTH)
#define V_R1_SIZE ((V_MAX - V_MIN) / V_R1_LENGTH)
#define W_R1_VOL (W_R1_SIZE * W_R1_SIZE * W_R1_SIZE)
#define NUM_R1_REGIONS (W_R1_LENGTH * W_R1_LENGTH * W_R1_LENGTH * V_R1_LENGTH * V_R1_LENGTH * V_R1_LENGTH)
#define NUM_R2_REGIONS (NUM_R1_REGIONS * W_R2_LENGTH * W_R2_LENGTH * W_R2_LENGTH * V_R2_LENGTH * V_R2_LENGTH * V_R2_LENGTH)
#define NUM_R2_PER_R1 W_R2_LENGTH *W_R2_LENGTH *W_R2_LENGTH *V_R2_LENGTH *V_R2_LENGTH *V_R2_LENGTH
#define NUM_R1_REGIONS_KERNEL1 1024
#define NUM_PARTIAL_SUMS 1024
#define EPSILON 1e-2f
#define VERBOSE 1
// Without this the #if in KinoPaxPlus.cu sees an undefined macro and takes the 0 branch, so
// the baseline would run NODE-ONLY pruning instead of the full parent chain that the
// checked-in config.h selects. KinoPaxPlus is a headline series here (at two
// discretizations), so it must be the real one.
#define KINOPAXPLUS_PARENT_CHAIN_PRUNING 1
// --- UNICYCLE MODEL: MODEL 0 ---
#define UNI_MIN_STEERING -M_PI / 2
#define UNI_MAX_STEERING M_PI / 2
#define UNI_MIN_DT 0.1f
#define UNI_MAX_DT 2.0f
#define UNI_LENGTH 1.0f
// --- DUBINS AIRPLANE: MODEL 2 ---
#define DUBINS_AIRPLANE_MIN_PR (-M_PI / 4)
#define DUBINS_AIRPLANE_MAX_PR (M_PI / 4)
#define DUBINS_AIRPLANE_MIN_YR (-M_PI / 4)
#define DUBINS_AIRPLANE_MAX_YR (M_PI / 4)
#define DUBINS_AIRPLANE_MIN_YAW -M_PI
#define DUBINS_AIRPLANE_MAX_YAW M_PI
#define DUBINS_AIRPLANE_MIN_PITCH -M_PI / 3
#define DUBINS_AIRPLANE_MAX_PITCH M_PI / 3
// --- NON LINEAR QUAD: MODEL 3 ---
#define QUAD_MIN_Zc 0.0f
#define QUAD_MAX_Zc 30.0f
#define QUAD_MIN_Lc -M_PI
#define QUAD_MAX_Lc M_PI
#define QUAD_MIN_Mc -M_PI
#define QUAD_MAX_Mc M_PI
#define QUAD_MIN_Nc -M_PI
#define QUAD_MAX_Nc M_PI
#define QUAD_MIN_YAW -M_PI
#define QUAD_MAX_YAW M_PI
#define QUAD_MIN_PITCH -M_PI
#define QUAD_MAX_PITCH M_PI
#define QUAD_MIN_ROLL -M_PI
#define QUAD_MAX_ROLL M_PI
#define QUAD_MIN_ANGLE_RATE -30.0f
#define QUAD_MAX_ANGLE_RATE 30.0f
#define NU 10e-3f
#define MU 2e-6f
#define KM 0.03f
#define IX 1.0f
#define IY 1.0f
#define IZ 2.0f
#define GRAVITY -9.81f
#define MASS 1.0f
#define MASS_INV 1.0f / MASS
CONFIGEOF
        ;;
    2)
        cat > "$CONFIG_FILE" << CONFIGEOF
#pragma once
/***************************/
/* 6D DUBINS AIRPLANE      */
/***************************/
#define MODEL 2
#define COST_MODE ${COST_MODE}  // path cost: 1 = control effort ((yawRate^2+pitchRate^2+a^2)*dt), 0 = workspace distance
#define MAX_TREE_SIZE 1000000
#define MAX_FLOAT 1e38f
#define MAX_SOL_SET_SIZE 500
#define MAX_ITER 1000
#define MAX_ITER_REKINO 20000
#define STEP_SIZE 0.1f
#define MAX_PROPAGATION_DURATION 10
#define ACCEPT 0.99f
#define AGENT_RADIUS 0.005f
#define GOAL_THRESH 0.05f
#define STATE_DIM 6
#define CONTROL_DIM 3
#define SAMPLE_DIM (STATE_DIM + CONTROL_DIM + 1)
#define W_DIM 3
#define C_DIM 2
#define V_DIM 1
#define W_MIN 0.0f
#define W_MAX 1.0f
#define W_SIZE 1.0f
#define C_MIN -M_PI
#define C_MAX M_PI
#define V_MIN 0.0f
#define V_MAX 0.3f
#define A_MIN -0.3f
#define A_MAX 0.3f
#define W_R1_LENGTH ${W_R1}
#define C_R1_LENGTH ${C_R1}
#define V_R1_LENGTH ${V_R1}
#define W_R2_LENGTH 2
#define C_R2_LENGTH 2
#define V_R2_LENGTH 2
#define W_R1_SIZE ((W_MAX - W_MIN) / W_R1_LENGTH)
#define C_R1_SIZE ((C_MAX - C_MIN) / C_R1_LENGTH)
#define V_R1_SIZE ((V_MAX - V_MIN) / V_R1_LENGTH)
#define W_R1_VOL (W_R1_SIZE * W_R1_SIZE * W_R1_SIZE)
#define NUM_R1_REGIONS (W_R1_LENGTH * W_R1_LENGTH * W_R1_LENGTH * C_R1_LENGTH * C_R1_LENGTH * V_R1_LENGTH)
#define NUM_R2_REGIONS (NUM_R1_REGIONS * W_R2_LENGTH * W_R2_LENGTH * W_R2_LENGTH * C_R2_LENGTH * C_R2_LENGTH * V_R2_LENGTH)
#define NUM_R2_PER_R1 W_R2_LENGTH *W_R2_LENGTH *W_R2_LENGTH *C_R2_LENGTH *C_R2_LENGTH *V_R2_LENGTH
#define NUM_R1_REGIONS_KERNEL1 1024
#define NUM_PARTIAL_SUMS 1024
#define EPSILON 1e-2f
#define VERBOSE 1
// Without this the #if in KinoPaxPlus.cu sees an undefined macro and takes the 0 branch, so
// the baseline would run NODE-ONLY pruning instead of the full parent chain that the
// checked-in config.h selects. KinoPaxPlus is a headline series here (at two
// discretizations), so it must be the real one.
#define KINOPAXPLUS_PARENT_CHAIN_PRUNING 1
// --- UNICYCLE MODEL: MODEL 0 ---
#define UNI_MIN_STEERING -M_PI / 2
#define UNI_MAX_STEERING M_PI / 2
#define UNI_MIN_DT 0.1f
#define UNI_MAX_DT 2.0f
#define UNI_LENGTH 1.0f
// --- DUBINS AIRPLANE: MODEL 2 ---
#define DUBINS_AIRPLANE_MIN_PR (-M_PI / 4)
#define DUBINS_AIRPLANE_MAX_PR (M_PI / 4)
#define DUBINS_AIRPLANE_MIN_YR (-M_PI / 4)
#define DUBINS_AIRPLANE_MAX_YR (M_PI / 4)
#define DUBINS_AIRPLANE_MIN_YAW -M_PI
#define DUBINS_AIRPLANE_MAX_YAW M_PI
#define DUBINS_AIRPLANE_MIN_PITCH -M_PI / 3
#define DUBINS_AIRPLANE_MAX_PITCH M_PI / 3
// --- NON LINEAR QUAD: MODEL 3 ---
#define QUAD_MIN_Zc 0.0f
#define QUAD_MAX_Zc 30.0f
#define QUAD_MIN_Lc -M_PI
#define QUAD_MAX_Lc M_PI
#define QUAD_MIN_Mc -M_PI
#define QUAD_MAX_Mc M_PI
#define QUAD_MIN_Nc -M_PI
#define QUAD_MAX_Nc M_PI
#define QUAD_MIN_YAW -M_PI
#define QUAD_MAX_YAW M_PI
#define QUAD_MIN_PITCH -M_PI
#define QUAD_MAX_PITCH M_PI
#define QUAD_MIN_ROLL -M_PI
#define QUAD_MAX_ROLL M_PI
#define QUAD_MIN_ANGLE_RATE -30.0f
#define QUAD_MAX_ANGLE_RATE 30.0f
#define NU 10e-3f
#define MU 2e-6f
#define KM 0.03f
#define IX 1.0f
#define IY 1.0f
#define IZ 2.0f
#define GRAVITY -9.81f
#define MASS 1.0f
#define MASS_INV 1.0f / MASS
CONFIGEOF
        ;;
    3)
        cat > "$CONFIG_FILE" << CONFIGEOF
#pragma once
/***************************/
/* 12D NON-LINEAR QUAD     */
/***************************/
// --- Previous models this file has used (Double Integrator, Dubins Airplane) -- see this
// function's case 1)/2) arms above for their full blocks, not deleted. Quad is STATE_DIM=12 (not
// 6), has a THIRD cubed discretization axis (NUM_R1_REGIONS = W^3*C^3*V^3, not W^3*C^2*V or
// W^3*V^3), and its own values use a [0,100] workspace/[-30,30] velocity scale (not [0,1]/
// [-0.3,0.3]) -- kept as-is here rather than shrunk to match, so gravity/mass/thrust stay
// physically consistent with how Quad was tuned; obstacles/start/goal are scaled x100 at load
// time in paper_benchmark_v2.cu instead (a no-op for every other model, since their W_SIZE is 1.0). ---
#define MODEL 3
#define COST_MODE ${COST_MODE}  // path cost: 1 = control effort, 0 = workspace distance (see edgeCost() -- Quad has no COST_MODE==1 branch of its own, so effort silently falls back to distance)
#define MAX_TREE_SIZE 1000000
#define MAX_FLOAT 1e38f
#define MAX_SOL_SET_SIZE 500
#define MAX_ITER 1000
#define MAX_ITER_REKINO 20000
#define STEP_SIZE 0.1f
#define MAX_PROPAGATION_DURATION 10
#define ACCEPT 0.99f
#define AGENT_RADIUS 0.5f        // x100, matching Quad's workspace scale (still unused by collision checking today, same as before)
#define GOAL_THRESH 5.0f         // x100 -- matches Quad's own checked-in config.h default exactly
#define STATE_DIM 12
#define CONTROL_DIM 4
#define SAMPLE_DIM (STATE_DIM + CONTROL_DIM + 1)
#define W_DIM 3
#define C_DIM 3   // roll, pitch, yaw (not validity-checked by the propagator, region-binning only)
#define V_DIM 3   // body-frame u, v, w (the propagator DOES validity-check these against V_MIN/V_MAX)
#define W_MIN 0.0f
#define W_MAX 100.0f
#define W_SIZE 100.0f
#define C_MIN -M_PI
#define C_MAX M_PI
#define V_MIN -30.0f
#define V_MAX 30.0f
#define A_MIN -0.3f
#define A_MAX 0.3f
#define W_R1_LENGTH ${W_R1}
#define C_R1_LENGTH ${C_R1}
#define V_R1_LENGTH ${V_R1}
#define W_R2_LENGTH 2
#define C_R2_LENGTH 1
#define V_R2_LENGTH 2
#define W_R1_SIZE ((W_MAX - W_MIN) / W_R1_LENGTH)
#define C_R1_SIZE ((C_MAX - C_MIN) / C_R1_LENGTH)
#define V_R1_SIZE ((V_MAX - V_MIN) / V_R1_LENGTH)
#define W_R1_VOL (W_R1_SIZE * W_R1_SIZE * W_R1_SIZE)
#define NUM_R1_REGIONS (W_R1_LENGTH * W_R1_LENGTH * W_R1_LENGTH * C_R1_LENGTH * C_R1_LENGTH * C_R1_LENGTH * V_R1_LENGTH * V_R1_LENGTH * V_R1_LENGTH)
#define NUM_R2_REGIONS (NUM_R1_REGIONS * W_R2_LENGTH * W_R2_LENGTH * W_R2_LENGTH * C_R2_LENGTH * C_R2_LENGTH * C_R2_LENGTH * V_R2_LENGTH * V_R2_LENGTH * V_R2_LENGTH)
#define NUM_R2_PER_R1 W_R2_LENGTH *W_R2_LENGTH *W_R2_LENGTH *C_R2_LENGTH *C_R2_LENGTH *C_R2_LENGTH *V_R2_LENGTH *V_R2_LENGTH *V_R2_LENGTH
#define NUM_R1_REGIONS_KERNEL1 1024
#define NUM_PARTIAL_SUMS 1024
#define EPSILON 1e-2f
#define VERBOSE 1
// Without this the #if in KinoPaxPlus.cu sees an undefined macro and takes the 0 branch, so
// the baseline would run NODE-ONLY pruning instead of the full parent chain that the
// checked-in config.h selects. KinoPaxPlus is a headline series here (at two
// discretizations), so it must be the real one.
#define KINOPAXPLUS_PARENT_CHAIN_PRUNING 1
// --- UNICYCLE MODEL: MODEL 0 ---
#define UNI_MIN_STEERING -M_PI / 2
#define UNI_MAX_STEERING M_PI / 2
#define UNI_MIN_DT 0.1f
#define UNI_MAX_DT 2.0f
#define UNI_LENGTH 1.0f
// --- DUBINS AIRPLANE: MODEL 2 ---
#define DUBINS_AIRPLANE_MIN_PR (-M_PI / 4)
#define DUBINS_AIRPLANE_MAX_PR (M_PI / 4)
#define DUBINS_AIRPLANE_MIN_YR (-M_PI / 4)
#define DUBINS_AIRPLANE_MAX_YR (M_PI / 4)
#define DUBINS_AIRPLANE_MIN_YAW -M_PI
#define DUBINS_AIRPLANE_MAX_YAW M_PI
#define DUBINS_AIRPLANE_MIN_PITCH -M_PI / 3
#define DUBINS_AIRPLANE_MAX_PITCH M_PI / 3
// --- NON LINEAR QUAD: MODEL 3 ---
#define QUAD_MIN_Zc 0.0f
#define QUAD_MAX_Zc 30.0f
#define QUAD_MIN_Lc -M_PI
#define QUAD_MAX_Lc M_PI
#define QUAD_MIN_Mc -M_PI
#define QUAD_MAX_Mc M_PI
#define QUAD_MIN_Nc -M_PI
#define QUAD_MAX_Nc M_PI
#define QUAD_MIN_YAW -M_PI
#define QUAD_MAX_YAW M_PI
#define QUAD_MIN_PITCH -M_PI
#define QUAD_MAX_PITCH M_PI
#define QUAD_MIN_ROLL -M_PI
#define QUAD_MAX_ROLL M_PI
#define QUAD_MIN_ANGLE_RATE -30.0f
#define QUAD_MAX_ANGLE_RATE 30.0f
#define NU 10e-3f
#define MU 2e-6f
#define KM 0.03f
#define IX 1.0f
#define IY 1.0f
#define IZ 2.0f
#define GRAVITY -9.81f
#define MASS 1.0f
#define MASS_INV 1.0f / MASS
CONFIGEOF
        ;;
    *)
        echo "ERROR: write_config: unknown MODEL '$MODEL' (expected 1, 2, or 3)" >&2
        exit 1
        ;;
    esac
}

echo ""
echo "======================================================="
echo "  PAPER BENCHMARK V3 -- fixed 4-planner comparison, on countingstars_sweep.cu's harness"
echo "  ALL THREE MODELS x ALL THREE DISCRETIZATIONS this pass: ${MODEL_NAMES[*]}"
echo "  MAX_TREE_SIZE=1,000,000 (was 3,000,000 in v2); per-run wall-clock cap 3s (was 10s)"
echo "  (Quad's obstacles/start/goal are scaled x100 from their [0,1]-authored CSVs -- see file header)"
echo "  Environments: ${ENV_NAMES[*]}  (nested inside each COARSE/FINE/TINY output subfolder)"
for row in "${!MODEL_DELTA_LABELS[@]}"; do
    MODEL="${MODEL_DELTA_MODEL_IDS[$row]}"
    DL="${MODEL_DELTA_LABELS[$row]}"
    DF=$(delta_folder_name "$DL")
    R=$(compute_regions "$MODEL" "${MODEL_DELTA_W_R1S[$row]}" "${MODEL_DELTA_C_R1S[$row]}" "${MODEL_DELTA_V_R1S[$row]}")
    echo "  Model ${MODEL} (${MODEL_NAMES[$((MODEL-1))]}): delta=${DL} [${DF}] | W_R1=${MODEL_DELTA_W_R1S[$row]} C_R1=${MODEL_DELTA_C_R1S[$row]} V_R1=${MODEL_DELTA_V_R1S[$row]} | Regions=${R} | full comparison"
done
echo "  Cost metrics: ${COST_LABELS[*]}  (one build each, per model x delta row)"
echo "  Planners run as SEPARATE PROCESSES this pass (${PLANNER_NAMES[*]}), each wrapped in an"
echo "  external timeout -- a true CUDA hang has no in-process recovery, so this bounds a hang's"
echo "  blast radius to the one planner that hung and logs it as a failure instead of freezing the"
echo "  whole sweep (see RUN_TIMEOUT_S / FAILURE_LOG below). Series per (model, delta, env, cost),"
echo "  30 runs each -- every axis below is a SINGLE FIXED POINT, not a grid:"
echo "    KPAX             baseline -- Quad's tiny row uses the SAFE, reduced region count, not the"
echo "                     old one that triggered a confirmed KPAX buffer-overshoot bug on Quad"
echo "                     (see paper_benchmark_v3.cu's skipKPAXThisDelta comment); the per-planner"
echo "                     timeout below will still catch and log any hang regardless."
echo "    KinoPaxPlus"
echo "    KinoPaxSTARTrue  syclopCap 1.0 (no cap), ancestorPrune = 1 (guarded stale-best prune on"
echo "                     top of the naive KPAX/KinoPaxPlus fusion)."
echo "    CountingStars    bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15, cost_frac 0.75,"
echo "                     hopelessGuard PERMANENTLY ON (v3.5): excludes any candidate/dormant node"
echo "                     whose own cost already forecloses beating the best solution found so far"
echo "                     from every door (FRESHEST/CHEAPEST/OPTIMAL/both floors), not just"
echo "                     cost-based ones -- see h_hopelessGuard_ in CountingStars.cuh."
TOTAL_INVOCATIONS=$(( ${#MODEL_DELTA_LABELS[@]} * ${#COST_LABELS[@]} * ${#ENV_NAMES[@]} * ${#PLANNER_NAMES[@]} ))
TOTAL_RUNS=$(( TOTAL_INVOCATIONS * 30 ))
echo "  = ${#MODEL_DELTA_LABELS[@]} (model,delta) rows x ${#PLANNER_NAMES[@]} planners x 30 runs x ${#ENV_NAMES[@]} environments x ${#COST_LABELS[@]} cost"
echo "    metrics = ${TOTAL_INVOCATIONS} process invocations, ${TOTAL_RUNS} runs total."
echo "  Filenames: CountingStars_bs120_bf40_ef150_cf750_hg1, KinoPaxSTARTrue_cap100_anc1."
echo "  Earlier CSVs from the retired tuning grid (_bs120_bf30_..., _bs180_bf50_..., anc0, etc.) do"
echo "  not collide with these names, so they simply stop loading if left in the output folder."
echo "  B IS STILL A RAMP, RECOMPUTED EVERY ITERATION:"
echo "    x = itr/fill_iters, B(x) = floor((slope*x + floor) * MAX_TREE_SIZE/fill_iters)"
echo "  THREE DOORS PLUS A FLAT ADMISSION FLOOR, ALL BUDGETED (v3.4, permanent):"
echo "    FRESHEST   explore_frac * B, from the least-populated regions"
echo "    CHEAPEST   cost_frac * B, from the smallest cost distances -- OPTIMAL candidates (distance"
echo "               0, i.e. cost <= minCostsR1[r]) compete here too, permanently, always at bucket 0"
echo "    ADMIT FLOOR every candidate at accept_floor = 1e-4, only when nothing else admitted it"
echo "    REACT FLOOR every dormant node at react_floor = 1e-5, ON TOP of the budget"
echo "  Score floor:    COUNTINGSTARS HAS NO SCORE FLOOR AND USES NO EPSILON: it never reads"
echo "                  vertexScores, h_scoreFloor_, h_nActive_ or regionCoverage in any decision."
echo "======================================================="

# =============================================================================
# BUILD — compile once per (model, delta, cost metric) row, caching each binary. TAG carries both
# the model AND the delta, both because they're now independent build axes and because per-run CSV
# filenames are keyed only by (environment, delta label, run number) -- NEVER by model -- so
# without a model-qualified label, two models' per-run CSVs for the same env/cost/delta would land
# at IDENTICAL paths and silently overwrite each other. TAG is threaded into argv[1] in the RUN
# phase below for exactly this reason, not just into the binary's own filename.
# =============================================================================
if [ "$SKIP_BUILD" = false ]; then
    for row in "${!MODEL_DELTA_LABELS[@]}"; do
        MODEL="${MODEL_DELTA_MODEL_IDS[$row]}"
        DL="${MODEL_DELTA_LABELS[$row]}"
        W_R1="${MODEL_DELTA_W_R1S[$row]}"; C_R1="${MODEL_DELTA_C_R1S[$row]}"; V_R1="${MODEL_DELTA_V_R1S[$row]}"
        TAG="m${MODEL}_${DL}"
        for i in "${!COST_LABELS[@]}"; do
            CL="${COST_LABELS[$i]}"
            CM="${COST_MODES[$i]}"
            REGIONS=$(compute_regions "$MODEL" "$W_R1" "$C_R1" "$V_R1")

            echo ""
            echo "=== BUILDING (model=${MODEL_NAMES[$((MODEL-1))]}, delta=${DL}, cost=${CL}, COST_MODE=${CM}, Regions=${REGIONS}) ==="

            write_config "$MODEL" "$W_R1" "$C_R1" "$V_R1" "$CM"

            cd "$BUILD_DIR"
            # shellcheck disable=SC2086
            cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_COMPILER_FLAGS 2>&1 | tail -5
            make PaperBenchmarkV3 -j"$(nproc)" 2>&1 | tail -20
            # Cache under a (model, delta, metric)-suffixed name so the run phase needs no rebuild
            cp PaperBenchmarkV3 "PaperBenchmarkV3_${TAG}_${CL}"
            cd "$PROJECT_DIR"
        done
    done
else
    echo ""
    echo "=== SKIPPING BUILD PHASE (using cached binaries) ==="
    cd "$BUILD_DIR"
    for row in "${!MODEL_DELTA_LABELS[@]}"; do
        MODEL="${MODEL_DELTA_MODEL_IDS[$row]}"
        DL="${MODEL_DELTA_LABELS[$row]}"
        TAG="m${MODEL}_${DL}"
        for CL in "${COST_LABELS[@]}"; do
            if [ ! -f "PaperBenchmarkV3_${TAG}_${CL}" ]; then
                echo "ERROR: Cached binary not found: PaperBenchmarkV3_${TAG}_${CL}"
                echo "Run without --skip-build first to create cached binaries."
                exit 1
            fi
        done
    done
    echo "  All cached binaries found."
    cd "$PROJECT_DIR"
fi

# =============================================================================
# RUN — model x delta x cost x environment x planner, each its own process invocation, wrapped in
# an external timeout. NOTHING inside the .cu file's own host loop can recover from a true hang:
# the same host thread that would check plannerMs >= maxTimeMs is the one parked inside
# cudaEventSynchronize if a single kernel launch genuinely wedges (see paper_benchmark_v3.cu's
# skipKPAXThisDelta comment for the historical hang this guards against). OS-level termination from
# outside the process is the only mechanism that can recover.
# =============================================================================
# --dump-viz writes run-0's full tree per variant (+ meta.csv) for the tree-growth /
# R1-density visualization. OFF by default here: every variant dumps a full tree of up to
# MAX_TREE_SIZE nodes, and the count multiplies by builds and environments.
# Enable with DUMP_VIZ=1 bash run_countingstars_sweep.sh
VIZ_FLAG=""
if [ "${DUMP_VIZ:-0}" != "0" ]; then
    VIZ_FLAG="--dump-viz"
fi

# RUN_TIMEOUT_S: RECOMPUTED FOR v3's 3s MAX_TIME_MS (was 900s at v2's 10s cap, same formula):
# 30 runs x up to 3s MAX_TIME_MS + 29 x 0.5s inter-run sleeps = 104.5s pure planner-loop ceiling
# (the expected case, not a pessimistic one) -- DOUBLED for margin, ~209s, rounded to 210s, so a
# legitimately slow run is never misclassified as a failure. A real hang now costs a few extra
# minutes of detection across 288 invocations (up from v2's 96), same as before.
RUN_TIMEOUT_S=210
KILL_AFTER_S=30   # grace period after SIGTERM before timeout escalates to SIGKILL

OUTPUT_ROOT="$BUILD_DIR/Data/Benchmarks/PaperBenchmarkV3"
FAILURE_LOG="$OUTPUT_ROOT/failures.log"
mkdir -p "$OUTPUT_ROOT"
: > "$FAILURE_LOG"   # fresh log every sweep

cd "$BUILD_DIR"
for row in "${!MODEL_DELTA_LABELS[@]}"; do
    MODEL="${MODEL_DELTA_MODEL_IDS[$row]}"
    DL="${MODEL_DELTA_LABELS[$row]}"
    DF=$(delta_folder_name "$DL")
    TAG="m${MODEL}_${DL}"
    for i in "${!COST_LABELS[@]}"; do
        CL="${COST_LABELS[$i]}"
        BIN="PaperBenchmarkV3_${TAG}_${CL}"
        for e in "${!ENV_NAMES[@]}"; do
            EN="${ENV_NAMES[$e]}"
            EO="${ENV_OBSTACLES[$e]}"
            for p in "${!PLANNER_NAMES[@]}"; do
                PN="${PLANNER_NAMES[$p]}"
                PF="${PLANNER_FLAGS[$p]}"

                echo ""
                echo "=== RUNNING (model=${MODEL_NAMES[$((MODEL-1))]}, delta=${DL} [${DF}], planner=${PN}, cost=${CL}, env=${EN}) ==="

                # `|| ec=$?` (NOT `if ! CMD; then ec=$?`): `!` negation collapses $? to a plain 0/1
                # boolean and would destroy the exit-code distinction (124 timeout vs 128+N killed
                # vs a plain crash) the log below depends on. This form also needs no set +e/-e
                # toggling: a command on the left of `||` is already exempt from errexit.
                ec=0
                # argv[1] carries the model-qualified TAG + cost metric, so it lands in every
                # output filename as _delta${TAG}_${CL} -- this is what keeps different models'
                # per-run CSVs from colliding (see the BUILD phase's TAG comment above). argv[4] is
                # NEW IN v3: the discretization output folder (DF), read by paper_benchmark_v3.cu
                # before the --only-* flags -- see its own argv-parsing comment.
                # shellcheck disable=SC2086
                timeout --kill-after="${KILL_AFTER_S}s" "${RUN_TIMEOUT_S}s" \
                    "./${BIN}" "${TAG}_${CL}" "$EO" "$EN" "$DF" "$PF" $VIZ_FLAG || ec=$?

                if [ "$ec" -ne 0 ]; then
                    # Without --preserve-status (not used here, deliberately), `timeout` itself
                    # reports 124 for ANY timeout-triggered kill -- whether the initial TERM was
                    # enough or --kill-after had to escalate to KILL -- so 124 alone already means
                    # "this hung" unambiguously. ec>=128 is therefore something else entirely: the
                    # process died from a signal on its OWN (e.g. a genuine CUDA crash), unrelated
                    # to this timeout wrapper.
                    if   [ "$ec" -eq 124 ]; then STATUS="TIMEOUT"                   # hit RUN_TIMEOUT_S
                    elif [ "$ec" -ge 128 ]; then STATUS="CRASHED_SIG$((ec - 128))"  # 139=SEGV, 134=ABRT, ...
                    else                          STATUS="ERROR"
                    fi
                    echo "  *** FAILURE: exit=${ec} status=${STATUS} -- logged to ${FAILURE_LOG} ***"
                    printf '%s model=%s planner=%s env=%s cost=%s exit=%s status=%s\n' \
                        "$(date -Iseconds)" "$MODEL" "$PN" "$EN" "$CL" "$ec" "$STATUS" >> "$FAILURE_LOG"
                fi
                # Falls through to the next (model, cost, env, planner) combination either way --
                # nothing above calls exit/return/break on a nonzero $ec.
            done
        done
    done
done
cd "$PROJECT_DIR"

echo ""
echo "======================================================="
echo "  PAPER BENCHMARK V3 COMPLETE"
echo "======================================================="
for DFOLDER in COARSE FINE TINY; do
    for EN in "${ENV_NAMES[@]}"; do
        echo "Results in: $BUILD_DIR/Data/Benchmarks/PaperBenchmarkV3/${DFOLDER}/${EN}/"
    done
done
echo "No v3-aware plot script exists yet (process_paper_benchmark_v2_and_plot.m reads v2's flat"
echo "<env>/ layout, not v3's <COARSE|FINE|TINY>/<env>/ one) -- read the per-run CSVs directly, or"
echo "adapt that script's dataDir/environments handling to the new nesting first."
if [ "${DUMP_VIZ:-0}" != "0" ]; then
    echo "Viz dumps:  $BUILD_DIR/Data/Benchmarks/PaperBenchmarkV3/<COARSE|FINE|TINY>/<env>/viz/  (visualize with scripts/visualize_tree_growth.m)"
fi
FAILURE_COUNT=$(wc -l < "$FAILURE_LOG" | tr -d ' ')
echo "Failures logged: ${FAILURE_COUNT} (see $FAILURE_LOG)"
if [ "$FAILURE_COUNT" != "0" ]; then
    cat "$FAILURE_LOG"
fi
echo "Config.h will be restored to original on exit."
