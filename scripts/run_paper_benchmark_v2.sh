#!/bin/bash
# =============================================================================
# Paper Benchmark v2 -- countingstars_sweep.cu's harness, minimally adapted to reproduce
# paper_benchmark.cu's fixed 4-planner headline comparison reliably.
#
# paper_benchmark.cu (MODEL 2 / Dubins Airplane) hangs at the `tiny` discretization; the root cause
# was not fully resolved despite a deep investigation. countingstars_sweep.cu (MODEL 1 / Double
# Integrator), built from the same four planners at the same three discretizations, has completed
# every delta including `tiny` without hanging. This file is that harness with three changes from
# it: (1) all four environments run, not just one; (2) the sweep's two exploratory grids
# (CountingStars bufferSlope/bufferFloor, KinoPaxSTARTrue ancestorPrune) are collapsed to the single
# fixed operating points paper_benchmark.cu was always meant to report; (3) MODEL switched from 1
# (Double Integrator) to 2 (Dubins Airplane) -- reproducing paper_benchmark.cu's original target
# model on a harness that has actually completed every delta without hanging.
#
# THE DIMENSION BREAKDOWN IS THE CORRECTED ONE, NOT PAPER_BENCHMARK.CU'S: paper_benchmark.cu treats
# Dubins' state as if it were shaped like the Double Integrator (C_DIM=0, V_DIM=3, so yaw/pitch get
# crammed into the velocity-shaped V_DIM=3 slot bounded to [-0.3,0.3]) -- a real region-density-skew
# bug flagged earlier this session. This file uses config.h's own correctly-shaped Dubins reference
# block instead: C_DIM=2 (yaw+pitch, properly bounded [-pi,pi]), V_DIM=1 (airspeed only, bounded
# [0,0.3] -- also fixing a separate flagged bug: the old V_MIN=-0.3 let the "airplane" fly backward).
# Consequence: this tool's Model 2 results are NOT directly comparable to paper_benchmark.cu's own
# Model 2 numbers (different, better-calibrated region discretization), though the same COST_MODE
# and MAX_ITERATIONS/MAX_TIME_MS apply to both.
#
# THE HOPELESS GUARD (v3.5, see h_hopelessGuard_ in include/planners/CountingStars.cuh) runs ON,
# unconditionally: a candidate/dormant node whose own cost already forecloses beating the best
# full-solution cost found so far is excluded from every door (FRESHEST, CHEAPEST, OPTIMAL, both
# completeness floors), not just the cost-based ones. The background WHY (v2->v3.5 design history)
# further below still applies in full -- only the "sweep a grid" framing changed to "one fixed
# point per axis".
#
# Per environment, AT EACH OF THREE DISCRETIZATIONS (`large`/`fine`/`tiny`, unchanged from the
# sweep -- see DELTA_LABELS below; all three run the full comparison):
#   KPAX                                                    = 1 point  x 5 runs
#   KinoPaxPlus                                             = 1 point  x 5 runs
#   KinoPaxSTARTrue syclopCap 1.0 (no cap), ancestorPrune = 1
#                                                            = 1 point  x 5 runs
#   CountingStars   bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15, cost_frac 0.75,
#                   hopelessGuard PERMANENTLY ON (v3.5)      = 1 point  x 5 runs
#
# ONE COST METRIC (length/COST_MODE=0 only), FOUR ENVIRONMENTS (empty, house, narrowPassage,
# zigzag -- see ENV_NAMES below), one full build per delta: 4 series x 5 runs x 3 deltas x 4
# environments = 240 runs total, from 3 compiled binaries.
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
# ALL THREE DELTAS RUN THE FULL COMPARISON -- NOT --only-kinopaxplus at any of them. Conclusions at
# one delta do not automatically hold at the others, so all three need the full comparison, not
# KinoPaxPlus alone at the finer ones.
#
# Runs on all four environments (empty, house, narrowPassage, zigzag -- see ENV_NAMES below), each
# written to its own subfolder under Data/Benchmarks/PaperBenchmarkV2/<env>/.
#
# NUM_R1_REGIONS and COST_MODE are both COMPILE-TIME (config.h, and a #if inside edgeCost), so
# neither can vary within one binary. This script therefore borrows run_delta_benchmark.sh's
# build-cache pattern: write config.h and build once per (delta, cost metric), caching each binary
# under a suffixed name, then run them in a second pass. Both labels ride into every output filename
# as the argv[1] delta label (large_length / large_effort / fine_length / ...).
#
# It builds ONLY the PaperBenchmarkV2 target. That still compiles KPAX_lib, which is the
# monolithic library holding every planner in the repo -- so warnings from unrelated sources
# (ReKino and friends) scroll past on every build. They are pre-existing and unavoidable without
# splitting the library.
#
# THREE DELTAS RUN THIS PASS -- "large" and "fine" are paper_benchmark.cu's own current large/fine
# (copied by hand from run_paper_benchmark.sh; there is no cross-check enforcing they stay equal,
# so re-check both files if either one's deltas change again), plus this sweep's own "tiny",
# unchanged from earlier passes, all full comparison, per the header above.
#
# C_R1 IS REAL HERE, UNLIKE THE DOUBLE-INTEGRATOR SWEEP'S C_R1=1. NUM_R1_REGIONS = W_R1^3 *
# C_R1^2 * V_R1 (C_DIM=2: yaw+pitch; V_DIM=1: airspeed only), so C_R1 genuinely discriminates
# attitude at the R1 level, unlike the C_DIM=0 sweep this file started from.
#
# Deltas (Dubins Airplane: W_DIM=3, C_DIM=2, V_DIM=1):
#   large   W_R1=7   C_R1=3  V_R1=3  ->  7^3  * 3^2 * 3 =   9,261   (full comparison)
#   fine    W_R1=16  C_R1=4  V_R1=4  -> 16^3  * 4^2 * 4 = 262,144   (full comparison)
#   tiny    W_R1=14  C_R1=6  V_R1=6  -> 14^3  * 6^2 * 6 = 592,704   (full comparison)
#
# "tiny" names the CELL, not the count: it is the finest delta this pass runs, at 592,704 regions.
# Watch it for the per-region arrays -- every NUM_R1_REGIONS allocation and every full-array fill
# scales with this, and graph_.updateVertices() runs a kernel over all of them with 64 sub-vertex
# reads each.
#
# Original config.h is backed up and restored on exit/error.
#
# There is no automated cross-check script for this file's triplet (paper_benchmark_v2.cu /
# run_paper_benchmark_v2.sh / process_paper_benchmark_v2_and_plot.m) the way
# scripts/cross_check_countingstars_grid.py covers the sweep -- there's nothing to sweep-check any
# more (every axis here is a single fixed point), but a label typo in any of the three files would
# still silently show up as loadRuns() reporting "0 runs" for a series in MATLAB, not an error.
#
# Usage:
#   cd scripts && bash run_paper_benchmark_v2.sh
#   cd scripts && bash run_paper_benchmark_v2.sh --skip-build   # run only (cached binaries)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"

# Deltas: parallel arrays of label / W_R1 / C_R1 / V_R1. ALL THREE run the FULL comparison this
# pass (see DELTA_EXTRA_ARGS) -- a tuning conclusion at one delta is not assumed to hold at the
# others, so there is no "--only-kinopaxplus" arm to skip it with here. One build per
# (delta, cost metric), cached, so restoring or trimming the list changes only the loop bounds.
#
# ALL THREE DELTAS NOW MATCH run_paper_benchmark.sh's OWN DELTAS EXACTLY (kept in step by hand --
# there is no cross-check between the two sweep tools). "tiny" here USED TO be this sweep's own,
# independently-chosen delta while paper_benchmark.cu ran a different, riskier tiny (W_R1=17,
# V_R1=5) that hit a cudaErrorIllegalAddress / hang in the `empty` environment (a buffer-overflow
# bug in KPAX.cu/KinoPaxPlus.cu's goal-path reconstruction, fixed at three sites -- but the hang
# persisted afterward, root cause still open). This sweep's own tiny (W_R1=14, V_R1=6) ran clean
# for KPAX/KinoPaxPlus/CountingStars/KinoPaxSTARTrue at all three deltas here, so
# run_paper_benchmark.sh was updated to reuse it -- see that script's own header for the same
# history from its side.
# C_R1/V_R1 are NOT inert for Dubins Airplane (C_DIM=2: yaw+pitch; V_DIM=1: airspeed only -- see
# the MODEL 2 switch below). NUM_R1_REGIONS = W_R1^3 * C_R1^2 * V_R1 for this model, a different
# shape from Double Integrator's W_R1^3 * V_R1^3, so these are NOT the same numbers reused
# unchanged -- C_R1/V_R1 were re-derived to land on the same 9,261 / 262,144 / 592,704 region
# counts (all three exact) under this model's own formula: large 7^3*3^2*3=9261,
# fine 16^3*4^2*4=262144, tiny 14^3*6^2*6=592704.
DELTA_LABELS=("large" "fine" "tiny")
DELTA_W_R1S=(7  16 14)
DELTA_C_R1S=(3  4  6)
DELTA_V_R1S=(3  4  6)
DELTA_EXTRA_ARGS=("" "" "")

# --- Coarse delta only (uncomment to restore; comment out the four lines above) ---
# DELTA_LABELS=("large")
# DELTA_W_R1S=(7)
# DELTA_C_R1S=(3)
# DELTA_V_R1S=(3)
# DELTA_EXTRA_ARGS=("")

# Cost metric axis: label + COST_MODE (0 = workspace distance, 1 = control effort). BOTH THIS
# PASS -- doubles the build count (one binary per delta x cost metric) and the run count.
COST_LABELS=("length" "effort")
COST_MODES=(0 1)
# COST_LABELS=("length")
# COST_MODES=(0)

# Environments (obstacles already in [0,1]^3; W_MIN/W_MAX/W_SIZE unchanged from the Double
# Integrator sweep, so no rescaling needed). Each gets its own output subfolder.
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

# Function to write complete Model 2 (Dubins Airplane) config.h
write_config() {
    local W_R1=$1
    local C_R1=$2
    local V_R1=$3
    local COST_MODE=$4
    cat > "$CONFIG_FILE" << CONFIGEOF
#pragma once
/***************************/
/* 6D DUBINS AIRPLANE      */
/***************************/
#define MODEL 2
#define COST_MODE ${COST_MODE}  // path cost: 1 = control effort ((yawRate^2+pitchRate^2+a^2)*dt), 0 = workspace distance
#define MAX_TREE_SIZE 3000000
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
}

echo ""
echo "======================================================="
echo "  PAPER BENCHMARK V2 -- fixed 4-planner comparison, on countingstars_sweep.cu's harness"
echo "  Model: 2 (6D Dubins Airplane) -- corrected C_DIM=2/V_DIM=1 breakdown, not paper_benchmark.cu's"
echo "         own C_DIM=0/V_DIM=3 (yaw/pitch crammed into the velocity slot -- see file header)"
echo "  Environments: ${ENV_NAMES[*]}  (separate output subfolders)"
for i in "${!DELTA_LABELS[@]}"; do
    R=$(( DELTA_W_R1S[i]**3 * DELTA_V_R1S[i]**3 ))
    if [ -z "${DELTA_EXTRA_ARGS[$i]}" ]; then
        WHAT="full comparison"
    else
        WHAT="KinoPaxPlus only"
    fi
    echo "  Delta: ${DELTA_LABELS[$i]} | W_R1=${DELTA_W_R1S[$i]} C_R1=${DELTA_C_R1S[$i]} V_R1=${DELTA_V_R1S[$i]} | Regions=${R} | ${WHAT}"
done
echo "  Cost metrics: ${COST_LABELS[*]}  (one build each)"
echo "  Series this pass, ALL AT ALL THREE DELTAS x ALL FOUR ENVIRONMENTS x BOTH COST METRICS"
echo "  (5 runs each) -- every axis below is a SINGLE FIXED POINT, not a grid:"
echo "    KPAX             baseline"
echo "    KinoPaxPlus       "
echo "    KinoPaxSTARTrue  syclopCap 1.0 (no cap), ancestorPrune = 1 (guarded stale-best prune on"
echo "                     top of the naive KPAX/KinoPaxPlus fusion)."
echo "    CountingStars    bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15, cost_frac 0.75,"
echo "                     hopelessGuard PERMANENTLY ON (v3.5): excludes any candidate/dormant node"
echo "                     whose own cost already forecloses beating the best solution found so far"
echo "                     from every door (FRESHEST/CHEAPEST/OPTIMAL/both floors), not just"
echo "                     cost-based ones -- see h_hopelessGuard_ in CountingStars.cuh."
TOTAL_RUNS=$(( 4 * 5 * ${#DELTA_LABELS[@]} * ${#ENV_NAMES[@]} * ${#COST_LABELS[@]} ))
echo "  = 4 series x 5 runs x ${#DELTA_LABELS[@]} deltas x ${#ENV_NAMES[@]} environments x ${#COST_LABELS[@]} cost"
echo "    metrics = ${TOTAL_RUNS} runs total."
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
echo "  Baselines: KPAX, KinoPaxPlus -- BOTH AT ALL THREE DELTAS this pass, not KinoPaxPlus-only at"
echo "             the finer ones."
echo "======================================================="

# =============================================================================
# BUILD — compile the Large delta config once per cost metric, caching each binary
# =============================================================================
if [ "$SKIP_BUILD" = false ]; then
    for d in "${!DELTA_LABELS[@]}"; do
        DL="${DELTA_LABELS[$d]}"
        for i in "${!COST_LABELS[@]}"; do
            CL="${COST_LABELS[$i]}"
            CM="${COST_MODES[$i]}"
            REGIONS=$(( DELTA_W_R1S[d]**3 * DELTA_V_R1S[d]**3 ))

            echo ""
            echo "=== BUILDING (delta=${DL}, cost=${CL}, COST_MODE=${CM}, Regions=${REGIONS}) ==="

            write_config "${DELTA_W_R1S[$d]}" "${DELTA_C_R1S[$d]}" "${DELTA_V_R1S[$d]}" "$CM"

            cd "$BUILD_DIR"
            # shellcheck disable=SC2086
            cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_COMPILER_FLAGS 2>&1 | tail -5
            make PaperBenchmarkV2 -j"$(nproc)" 2>&1 | tail -20
            # Cache under a (delta, metric)-suffixed name so the run phase needs no rebuild
            cp PaperBenchmarkV2 "PaperBenchmarkV2_${DL}_${CL}"
            cd "$PROJECT_DIR"
        done
    done
else
    echo ""
    echo "=== SKIPPING BUILD PHASE (using cached binaries) ==="
    cd "$BUILD_DIR"
    for DL in "${DELTA_LABELS[@]}"; do
        for CL in "${COST_LABELS[@]}"; do
            if [ ! -f "PaperBenchmarkV2_${DL}_${CL}" ]; then
                echo "ERROR: Cached binary not found: PaperBenchmarkV2_${DL}_${CL}"
                echo "Run without --skip-build first to create cached binaries."
                exit 1
            fi
        done
    done
    echo "  All cached binaries found."
    cd "$PROJECT_DIR"
fi

# =============================================================================
# RUN — one pass per cost metric, using the cached binaries
# =============================================================================
# --dump-viz writes run-0's full tree per variant (+ meta.csv) for the tree-growth /
# R1-density visualization. OFF by default here: every variant dumps a full tree of up to
# MAX_TREE_SIZE nodes, and the count multiplies by builds and environments.
# Enable with DUMP_VIZ=1 bash run_countingstars_sweep.sh
VIZ_FLAG=""
if [ "${DUMP_VIZ:-0}" != "0" ]; then
    VIZ_FLAG="--dump-viz"
fi

cd "$BUILD_DIR"
for CL in "${COST_LABELS[@]}"; do
    for i in "${!ENV_NAMES[@]}"; do
        EN="${ENV_NAMES[$i]}"
        EO="${ENV_OBSTACLES[$i]}"
        for d in "${!DELTA_LABELS[@]}"; do
            DL="${DELTA_LABELS[$d]}"
            EXTRA="${DELTA_EXTRA_ARGS[$d]}"
            # All three deltas are full-sweep this pass (DELTA_EXTRA_ARGS all empty), so all three
            # dump viz when enabled; the branch below still matters if DELTA_EXTRA_ARGS is ever
            # restored to a KinoPaxPlus-only entry, which has nothing extra to show.
            if [ -z "$EXTRA" ]; then
                PASS_FLAGS="$VIZ_FLAG"
            else
                PASS_FLAGS="$EXTRA"
            fi
            echo ""
            echo "=== RUNNING (delta=${DL}, cost=${CL}, Env=${EN}) ${EXTRA} ==="
            # argv[1] carries the discretization and the cost metric, so it lands in every output
            # filename as _delta${DL}_${CL}; argv[3] selects the per-environment subfolder.
            # shellcheck disable=SC2086
            "./PaperBenchmarkV2_${DL}_${CL}" "${DL}_${CL}" "$EO" "$EN" $PASS_FLAGS
        done
    done
done
cd "$PROJECT_DIR"

echo ""
echo "======================================================="
echo "  PAPER BENCHMARK V2 COMPLETE"
echo "======================================================="
for EN in "${ENV_NAMES[@]}"; do
    echo "Results in: $BUILD_DIR/Data/Benchmarks/PaperBenchmarkV2/${EN}/"
done
echo "Plot each environment separately: cd into its folder, set envName at the top of"
echo "scripts/process_paper_benchmark_v2_and_plot.m to match, then run it by name."
echo "Plot with:  scripts/process_paper_benchmark_v2_and_plot.m (run it from that directory)"
if [ "${DUMP_VIZ:-0}" != "0" ]; then
    echo "Viz dumps:  $BUILD_DIR/Data/Benchmarks/PaperBenchmarkV2/viz/  (visualize with scripts/visualize_tree_growth.m)"
fi
echo "Config.h will be restored to original on exit."
