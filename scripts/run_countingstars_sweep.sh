#!/bin/bash
# =============================================================================
# CountingStars v3.2 Sweep Runner
#
# CountingStars v3.2 (A PER-ITERATION BUDGET RAMP, THREE FIXED SHARES), against the baselines it has
# to beat: KPAX, KPAXCap, KinoPaxPlus, and one tuned KinoPaxSTARCleanCost point. COMBO and TrueStar
# are deliberately NOT in this sweep -- COMBO is the thing being replaced and its own sweep still
# exists; TrueStar answers a cap question this planner does not ask.
#
# THIS PASS'S AXIS IS ANCESTOR BIAS, NOT THE RAMP AND NOT THE REACTIVATION GUARANTEE.
# bufferSlope/bufferFloor are FIXED at one point (1.2, 0.05) rather than swept -- an earlier pass
# already characterized the ramp on its own. The region-best GUARANTEE arm that a LATER pass made
# an ablatable toggle (h_reactGuaranteeBudgeted_) is now GONE PERMANENTLY: it is folded into the
# SAME cost-distance histogram/budget the CHEAPEST reactivation arm uses, with no toggle left for
# it -- that fold measurably curbed the over-reactivation of "optimal" nodes this sweep exists to
# characterize. What varies THIS pass is h_ancestorWeight_/h_ancestorAlpha_: whether CHEAPEST
# admission/reactivation additionally biases its cost-distance vote toward a node's LINEAGE (an EMA
# over its ancestors' own distances at their insertion time) rather than purely the node's own live
# distance. See csBlendDistance/h_ancestorWeight_ in CountingStars.cuh.
#
# Per (environment, cost metric), AT EACH OF TWO DISCRETIZATIONS (coarse and fine -- see
# DELTA_LABELS below; BOTH run the full comparison, unlike earlier passes where only the coarse
# delta did):
#   CountingStars   bufferSlope 1.2 (fixed), bufferFloor 0.05 (fixed),
#                   explore_frac and cost_frac FIXED at 0.3 each (not swept this pass),
#                   ancestorBias {false, true} = 2 points x 5 runs = 10 runs
#   CleanCost       r2 OFF, w 0.9, k 1, cap 0.03            = 1 point  x 5 runs
#   KPAXCap         cap {0.03}                              = 1 point  x 5 runs
#   KPAX                                                    = 1 point  x 5 runs
#   KinoPaxPlus                                             = 1 point  x 5 runs
#
# TWO COST METRICS THIS PASS (length, effort), each its own full build -- see COST_LABELS/
# COST_MODES below -- so every run count above is doubled in practice, AND doubled again across
# the two deltas: 40 CountingStars runs total, one environment (zigzag).
#
# ============================================================================================
# WHAT CHANGED FROM v2, AND WHY THIS SWEEP EXISTS
#
# 1. B IS DERIVED, NOT SWEPT. v2's grid was goal_frontier_size {200, 2000, 6000, 10000} with no
#    derivation behind any of them. v3 made the planner compute it instead:
#
#        B = floor(fill_frac * MAX_TREE_SIZE / MAX_ITER)      -- v3, ONE VALUE PER RUN
#
#    -- "the frontier size that fills the tree exactly at MAX_ITER", scaled by fill_frac, so the
#    remaining (1 - fill_frac) of the buffer is what the uncapped OPTIMAL door is left to spend.
#    B rides into every CSV as the goal_frontier_size column rather than being re-derived by the
#    plot script. SEE ITEM 5 BELOW: v3.2 replaces the single fill_frac with a per-iteration ramp,
#    so this formula is no longer what the planner actually runs -- kept here as the ancestor the
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
#    explore_frac to freshness, cost_frac to cheapness, and react_frac = 1 - explore - cost to
#    reactivation. All three are fractions of B ITSELF; only OPTIMAL stays uncapped and spends on
#    top (the region-best GUARANTEE that used to also be uncapped is gone -- folded permanently
#    into the reactivation budget, see CS_DOORBIT_GUAR in CountingStars.cuh).
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
#    a separate baseline that has to be swept again.
#
#    explore_frac AND cost_frac ARE FIXED AT 0.3 EACH this pass (not swept) -- v3's grid varied
#    them against fill_frac; this pass isolates the slope/floor grid's own effect by holding them
#    still. (They were briefly swept, 2 values each, in an intermediate pass -- not this one.)
#
#    B IS A PURE HOST SCALAR (read only inside updateFrontier(), never by propagateFrontier() or
#    any device kernel directly), so making it dynamic cost no device array, no new kernel, and no
#    new synchronisation -- it is one floating-point formula recomputed once per iteration.
#
# bufferSlope/bufferFloor ARE FIXED THIS PASS (1.2, 0.05), NOT SWEPT -- an earlier pass already
# characterized the ramp; this grid isolates ancestor bias's own effect by holding it still.
# ============================================================================================
#
# THE GRID SITS ENTIRELY BELOW NUM_R1_REGIONS (27,000 at the coarse delta), and that is the honest
# limit on what it can measure. OPTIMAL is the only uncapped door and is bounded by the region
# count rather than by B (at most one region best per region per iteration) -- the region-best
# GUARANTEE that used to also be uncapped and bounded the same way is gone, folded permanently into
# the CHEAPEST reactivation budget (see CS_DOORBIT_GUAR in CountingStars.cuh). So B binds EARLY in
# a run and then stops, at an iteration that moves with the WHOLE RAMP shape (bufferSlope and
# bufferFloor together) rather than a single fill_frac -- and early is exactly where
# time-to-first-solution is decided.
#
# That is what "tree growth is less controlled once min cost is always accepted" amounts to, and it
# is a measurement rather than a defect. Read budget_used/goal_frontier_size as a CURVE over
# iterations: the iteration where it crosses 1 IS the measurement, and a late-run overshoot is
# expected at every point -- MORE SO NOW, since B itself is climbing over the run rather than
# holding still, so the ratio has two moving parts instead of one.
#
# READ IN THIS ORDER:
#   1. goal_frontier_size vs iteration, FIRST. Confirms the realized ramp actually matches
#      slope*x + floor before reading anything else that depends on B. A flat line at every
#      bufferSlope = 0 series is the direct sanity check that the mechanism is wired correctly.
#   2. frontier_repeat_size / frontier_size. The realised mean rep, which should sit near 1 with a
#      small excess from thin regions and the both-doors boost, not near 4.
#   3. budget_used / goal_frontier_size, as a curve. See above -- now a moving target on both sides.
#   4. reactivated_cost against reactivated_count. The cost arm should carry essentially ALL of
#      Part B's non-guarantee volume (~ react_frac * B); reactivated_count is now the completeness
#      FLOOR alone, ~ react_floor * dormant_count, so ~30 nodes. Large there means the floor is doing
#      reach work it was not sized for. And react_cutoff_dist against dist_max says whether the arm
#      is actually selecting: pinned at 1 means the budget exceeds the population below the anchor
#      and it is partly picking at random within the clamped tail.
#   5. admitted_costdist against admitted_explore and optimal_count. THE CHEAPEST DOOR'S ACTUAL
#      SHARE. Pinned at 0 means the cutoff solve is degenerate; equal to cost_frac * B every
#      iteration means it is working exactly as designed. The two selection doors OVERLAP, so
#      admitted_both is what makes the four counts add back up:
#      admitted == optimal_count + admitted_explore + admitted_costdist - admitted_both.
#   6. cost_cutoff_dist against dist_max. The direct read on whether the log bucket map has the right
#      shape. A collapse toward dist_max / 2^21 means everything is landing in bucket 0 and the door
#      has degraded to a uniform draw among near-optimal candidates -- the signal to switch
#      csCostBucket to a linear map, which is a one-line change in the header.
#   7. ord_cutoff over the run. Rising means regions are filling and freshness is getting scarce,
#      which is expected. Pinned at 0 means explore_frac is doing nothing; pinned at 256 means the
#      whole candidate pool is fresh enough and explore_frac is not binding either.
#   8. First-solution time and cost, final cost (figures 11/12). THE ACTUAL QUESTION: does a ramp
#      beat the best bufferSlope = 0 point on time-to-first-solution AND close the final-cost gap
#      against CleanCost. bufferSlope = 0 points are the direct, structural control.
#
# BOTH DELTAS RUN THE FULL COMPARISON THIS PASS -- NOT --only-kinopaxplus at the finer one, unlike
# earlier passes through this file. The concern ancestor bias is meant to address is specifically a
# fine-discretization one: even with the guarantee folded into the budget, a finer delta's larger
# region count means the budget still only reactivates a SUBSET of near-optimal nodes each
# iteration, and biasing that subset toward good lineages may or may not help more at fine than at
# coarse. Answering that needs CountingStars (both toggle settings) AND KinoPaxPlus BOTH measured
# at the fine delta, not KinoPaxPlus alone. The grid stays small (2 CountingStars points)
# specifically so running it twice (once per delta) is still cheap.
#
# Runs on BOTH environments, each written to its own subfolder under
# Data/Benchmarks/CountingStars/<env>/ so they can be plotted independently.
#
# NUM_R1_REGIONS and COST_MODE are both COMPILE-TIME (config.h, and a #if inside edgeCost), so
# neither can vary within one binary. This script therefore borrows run_delta_benchmark.sh's
# build-cache pattern: write config.h and build once per (delta, cost metric), caching each binary
# under a suffixed name, then run them in a second pass. Both labels ride into every output filename
# as the argv[1] delta label (large_length / large_effort / fine_length / ...).
#
# It builds ONLY the CountingStarsSweep target. That still compiles KPAX_lib, which is the
# monolithic library holding every planner in the repo -- so warnings from unrelated sources
# (ReKino and friends) scroll past on every build. They are pre-existing and unavoidable without
# splitting the library.
#
# "fine" and "fine_control" are a CONTROLLED PAIR: identical 216,000 region count, refined in
# different subspaces -- workspace (W_R1 10 -> 20) vs velocity (V_R1 3 -> 6).
#
# C_R1 STAYS AT 1 EVERYWHERE. NUM_R1_REGIONS = W_R1^3 * V_R1^3 has no C term, and this config sets
# C_DIM 0, so getRegion / getSubRegion skip the C dimension entirely -- raising C_R1 would change
# nothing at all. The control-side refinement rides on V_R1.
#
# Deltas (Model 1: W_DIM=3, C_DIM=0, V_DIM=3), TWO THIS PASS -- "tiny" dropped, see above:
#   large (coarse)  W_R1=10  C_R1=1  V_R1=3  ->  10^3 * 3^3 =  27,000   (full comparison)
#   fine            W_R1=20  C_R1=1  V_R1=3  ->  20^3 * 3^3 = 216,000   (full comparison)
#
# Original config.h is backed up and restored on exit/error.
#
# RUN scripts/cross_check_countingstars_grid.py BEFORE A SWEEP. When the grid in this file, the .cu
# and the .m drift apart, MATLAB does not error -- loadRuns() silently finds no files and reports
# "0 runs" for the orphaned series, so the plot just looks sparse. That has cost whole sweeps.
#
# Usage:
#   cd scripts && bash run_countingstars_sweep.sh
#   cd scripts && bash run_countingstars_sweep.sh --skip-build   # run only (cached binaries)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"

# Deltas: parallel arrays of label / W_R1 / C_R1 / V_R1. BOTH run the FULL comparison this pass
# (see DELTA_EXTRA_ARGS) -- ancestor bias's expected effect is specifically about the fine delta,
# so there is no "--only-kinopaxplus" arm to skip it with here. One build per (delta, cost metric),
# cached, so restoring or trimming the list changes only the loop bounds.
DELTA_LABELS=("large" "fine")
DELTA_W_R1S=(10 20)
DELTA_C_R1S=(1  1)   # inert for Model 1 (C_DIM 0); control refinement rides on V_R1
DELTA_V_R1S=(3  3)
DELTA_EXTRA_ARGS=("" "")

# --- Coarse delta only (uncomment to restore; comment out the four lines above) ---
# DELTA_LABELS=("large")
# DELTA_W_R1S=(10)
# DELTA_C_R1S=(1)
# DELTA_V_R1S=(3)
# DELTA_EXTRA_ARGS=("")

# Cost metric axis: label + COST_MODE  (0 = workspace distance, 1 = control effort). BOTH this
# pass -- one build per entry, so two full builds of the whole grid.
COST_LABELS=("length" "effort")
COST_MODES=(0 1)

# Environments (obstacles already in [0,1]^3 for Model 1). Each gets its own output subfolder.
# SCOPE: zigzag only this pass. Other environments preserved below, commented out, for later runs.
ENV_NAMES=("zigzag")
ENV_OBSTACLES=("../include/config/obstacles/zigzag/obstacles.csv")

# --- narrowPassage --- a wall at x in [0.3, 0.5] spanning all z, split by a gap at y in
# [0.49, 0.51] -- 0.02 wide against an agent diameter of 0.01 (AGENT_RADIUS 0.005). The
# benchmark's start (0.1, 0.08, 0.05) and goal (0.8, 0.95, 0.9) are clear of both boxes and on
# opposite sides of the wall, so no endpoint change is needed -- but expect low success rates
# there, and read the success-rate subplot alongside the cost bars (unsolved runs are dropped from
# the cost mean, so a config that solved once cheaply can look best).
# ENV_NAMES=("narrowPassage")
# ENV_OBSTACLES=("../include/config/obstacles/narrowPassage/obstacles.csv")

# --- house + narrowPassage together (uncomment to restore; comment out the ENV_NAMES/ENV_OBSTACLES
# pair above) ---
# ENV_NAMES=("house" "narrowPassage")
# ENV_OBSTACLES=("../include/config/obstacles/house/obstacles.csv"
#                "../include/config/obstacles/narrowPassage/obstacles.csv")

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

# Function to write complete Model 1 config.h
write_config() {
    local W_R1=$1
    local C_R1=$2
    local V_R1=$3
    local COST_MODE=$4
    cat > "$CONFIG_FILE" << CONFIGEOF
#pragma once
/***************************/
/* 6D DOUBLE INTEGRATOR    */
/***************************/
#define MODEL 1
#define COST_MODE ${COST_MODE}  // path cost: 1 = control effort ((ax^2+ay^2+az^2)*dt), 0 = workspace distance
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
}

echo ""
echo "======================================================="
echo "  CountingStars v3 Sweep"
echo "  Model: 1 (6D Double Integrator)"
echo "  Environments: ${ENV_NAMES[*]}  (separate output subfolders)"
for i in "${!DELTA_LABELS[@]}"; do
    R=$(( DELTA_W_R1S[i]**3 * DELTA_V_R1S[i]**3 ))
    if [ -z "${DELTA_EXTRA_ARGS[$i]}" ]; then
        WHAT="full sweep"
    else
        WHAT="KinoPaxPlus only"
    fi
    echo "  Delta: ${DELTA_LABELS[$i]} | W_R1=${DELTA_W_R1S[$i]} C_R1=${DELTA_C_R1S[$i]} V_R1=${DELTA_V_R1S[$i]} | Regions=${R} | ${WHAT}"
done
echo "  Cost metrics: ${COST_LABELS[*]}  (one build each)"
echo "  CountingStars:  bufferSlope 1.2 (FIXED), bufferFloor 0.05 (FIXED)"
echo "                  explore_frac=0.3, cost_frac=0.3 (FIXED, not swept this pass)"
echo "                  ancestorBias {false,true} -- THIS PASS'S AXIS, at one fixed"
echo "                  (alpha,weight)=(0.5,0.5) on-point, not a 2D sweep"
echo "                  = 2 points x BOTH deltas"
echo "                  Filenames: _bs120_bf5_ef300_cf300_ab<on|off>,"
echo "                  e.g. CountingStars_bs120_bf5_ef300_cf300_aboff."
echo "                  Earlier CSVs (_rgon/_rgoff token, no ab token) cannot collide with this"
echo "                  shape, so they simply stop loading -- intended for a retired axis, not a loss."
echo "                  B IS STILL A RAMP, RECOMPUTED EVERY ITERATION (unchanged this pass):"
echo "                    x = itr/fill_iters, B(x) = floor((slope*x + floor) * MAX_TREE_SIZE/fill_iters)"
echo "                  FOUR DOORS PLUS A FLAT ADMISSION FLOOR, two of the four on a fixed share of B."
echo "                  THE REGION-BEST GUARANTEE DOOR IS GONE PERMANENTLY -- folded into CHEAPEST's"
echo "                  budget, no toggle left for it (see CS_DOORBIT_GUAR in CountingStars.cuh):"
echo "                    OPTIMAL    distance 0, i.e. cost <= minCostsR1[r].  UNCAPPED, first claim."
echo "                               v3.3: also competes for FRESHEST rather than returning early."
echo "                    FRESHEST   explore_frac * B, from the least-populated regions"
echo "                    CHEAPEST   cost_frac * B, from the smallest (ancestor-blended when"
echo "                               ancestorBias) cost distances -- region bests included"
echo "                    REACTIVATE react_frac * B, to the CHEAPEST DORMANT NODES (same blend)"
echo "                    ADMIT FLOOR (v3.3) every candidate at accept_floor = 1e-4, only when nothing"
echo "                               else admitted it -- a completeness guarantee, not a reach tool"
echo "                    REACT FLOOR every dormant node at react_floor = 1e-5, ON TOP of the budget"
echo "                  REACTIVATION IS COST-SELECTIVE (v3.1): CleanCost weights its own reactivation"
echo "                  arm by cost, and that was the one cost mechanism this line lacked -- the"
echo "                  volumes already matched, so it was selectivity not throughput. Part B is the"
echo "                  only thing that re-expands the tree INTERIOR, which is where cost refinement"
echo "                  happens."
echo "                  ANCESTOR BIAS (THIS PASS'S AXIS): blends a node's/candidate's own LIVE cost"
echo "                  distance with a FROZEN per-node lineage EMA (ancestors' own distances at"
echo "                  their insertion time) before CHEAPEST admission/reactivation bucket it --"
echo "                  see csBlendDistance/h_ancestorWeight_ in CountingStars.cuh. weight=0 (false)"
echo "                  is exact identity. The concern this tests: the budget only ever reactivates a"
echo "                  SUBSET of near-optimal nodes at a fine delta (more regions, same budget), and"
echo "                  biasing that subset toward good-lineage nodes may spend it more usefully."
echo "                  THE REACTIVATION FLOOR IS A CORRECTNESS CONSTANT, not a knob: a node's cost"
echo "                  distance only ever grows (fixed cost over a non-increasing region min), so"
echo "                  under a pure top-K a node above the cutoff is dead permanently and its"
echo "                  subtree unreachable. 1e-5 wakes ~30 nodes/iter -- completeness in the limit."
echo "                  THE ADMISSION FLOOR (v3.3) makes the same guarantee for CANDIDATES: every"
echo "                  collision-free candidate keeps a nonzero admission chance whatever its"
echo "                  region's state, at 1e-4 -- an order of magnitude above the reactivation"
echo "                  floor, since its pool is per-iteration and far smaller than the whole tree."
echo "                  CLEAR THE OUTPUT FOLDER FIRST IF A PREVIOUS PASS RAN -- the label shape"
echo "                  changed again, so old and new CSVs would otherwise coexist under different"
echo "                  names rather than colliding, which is fine but confusing to plot together."
echo "                  FRESHEST, CHEAPEST AND (v3.3) OPTIMAL select over the SAME candidate pool on"
echo "                  independent signals -- a candidate can clear more than one, and it is still"
echo "                  ONE tree node: every door that admits it buys ONE propagation block"
echo "                  (nodeBlocks = popcount(door) in Part A), not a duplicate node."
echo "                  ancestorBias=true IS THIS PASS'S ABLATION ARM: does biasing the CHEAPEST"
echo "                  budget toward good-lineage nodes spend an already-scarce reactivation budget"
echo "                  more usefully at a fine discretization. reactivated_best should read exactly"
echo "                  0 on every iteration of every run now -- its arm cannot fire any more."
echo "                  RNG NOTE: ancestor blending can shift which nodes land exactly on a cutoff"
echo "                  bucket boundary, so a node whose boundary roll fires under one setting and"
echo "                  not the other has its OWN curandState advance differently between the two --"
echo "                  compare AGGREGATE columns across repeated runs, not per-iteration CSVs"
echo "                  node-for-node."
echo "                  FAN-OUT IS DOOR-COUNT (v3.3), FULL STOP: nodeBlocks = popcount(door), no"
echo "                  region-thinness signal and no swept boost size left -- the region-keyed rule"
echo "                  KPAXCap and CleanCost use is gone from this planner."
echo "                  READ FIRST: goal_frontier_size vs iteration (does the realized ramp match"
echo "                  the intended shape), then the tradeoff scatter across ancestorBias at each"
echo "                  delta -- does the toggle move final cost or time-to-first-solution, and does"
echo "                  that answer change between coarse and fine."
echo "  CleanCost:      r2 OFF, w 0.9, k 1, cap 0.03 = 1 point (baseline)"
echo "  KPAXCap:        cap {0.03} = 1 point"
echo "  Score floor:    dynamic 1/N_active for KPAXCap/CleanCost; legacy EPSILON for KPAX."
echo "                  COUNTINGSTARS HAS NO SCORE FLOOR AND USES NO EPSILON: it never reads"
echo "                  vertexScores, h_scoreFloor_, h_nActive_ or regionCoverage in any decision."
echo "  Baselines: KPAX, CleanCost, KPAXCap, KinoPaxPlus -- ALL FIVE SERIES AT BOTH DELTAS this pass,"
echo "             not KinoPaxPlus-only at the finer one (see the header for why)."
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
            make CountingStarsSweep -j"$(nproc)" 2>&1 | tail -20
            # Cache under a (delta, metric)-suffixed name so the run phase needs no rebuild
            cp CountingStarsSweep "CountingStarsSweep_${DL}_${CL}"
            cd "$PROJECT_DIR"
        done
    done
else
    echo ""
    echo "=== SKIPPING BUILD PHASE (using cached binaries) ==="
    cd "$BUILD_DIR"
    for DL in "${DELTA_LABELS[@]}"; do
        for CL in "${COST_LABELS[@]}"; do
            if [ ! -f "CountingStarsSweep_${DL}_${CL}" ]; then
                echo "ERROR: Cached binary not found: CountingStarsSweep_${DL}_${CL}"
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
            # Both deltas run the full comparison this pass, so both dump viz when enabled.
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
            "./CountingStarsSweep_${DL}_${CL}" "${DL}_${CL}" "$EO" "$EN" $PASS_FLAGS
        done
    done
done
cd "$PROJECT_DIR"

echo ""
echo "======================================================="
echo "  COUNTINGSTARS SWEEP COMPLETE"
echo "======================================================="
for EN in "${ENV_NAMES[@]}"; do
    echo "Results in: $BUILD_DIR/Data/Benchmarks/CountingStars/${EN}/"
done
echo "Plot each environment separately: cd into its folder, set envName at the top of"
echo "scripts/process_countingstars_and_plot.m to match, then run it by name."
echo "Plot with:  scripts/process_countingstars_and_plot.m (run it from that directory)"
if [ "${DUMP_VIZ:-0}" != "0" ]; then
    echo "Viz dumps:  $BUILD_DIR/Data/Benchmarks/CountingStars/viz/  (visualize with scripts/visualize_tree_growth.m)"
fi
echo "Config.h will be restored to original on exit."
