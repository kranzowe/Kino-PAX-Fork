#!/bin/bash
# =============================================================================
# CountingStars v3.4 Sweep Runner -- tuning around permanently-budgeted OPTIMAL admission
#
# CountingStars v3.4, against KPAX and KinoPaxPlus. KPAXCap and KinoPaxSTARCleanCost are gone from
# this sweep entirely: with the optimal-accept-budget question settled (see below), this tool's
# job is tuning CountingStars' own remaining axes, not re-running comparisons against tuned STAR
# variants on every pass. Both are still runnable on their own via kinopaxstar_cost_tuning_sweep.cu
# / kinopaxstar_combo_tuning_sweep.cu. COMBO and TrueStar are, as before, deliberately NOT in this
# sweep -- COMBO is the thing CountingStars replaced, and TrueStar answers a cap question this
# planner does not ask.
#
# THE OPTIMAL-ACCEPT-BUDGET TOGGLE IS GONE. It ran as an on/off axis (h_optimalAcceptBudgeted_) in
# the previous pass through this file: the OPTIMAL door (a candidate at distance 0 from its
# region's minimum) used to be UNCAPPED, admitted unconditionally outside any budget. That sweep
# confirmed folding it into the SAME cost_frac * B histogram/cutoff CHEAPEST already spends against
# -- it always votes into bucket 0 and has to clear that cutoff/roll like anything else, "the
# cheapest of the cheap" rather than free -- improves final cost / time-to-first-solution. It is
# now the planner's ONLY behavior; there is no toggle, no h_optimalAcceptBudgeted_ field, and no
# `_ob` label token left to sweep.
#
# THIS PASS TUNES THE FOUR REMAINING AXES around that permanent behavior:
# Per (environment, cost metric), AT EACH OF THREE DISCRETIZATIONS (`large`/`fine` copied from
# paper_benchmark.cu's own current coarse/fine deltas, plus this sweep's own pre-existing `tiny` --
# see DELTA_LABELS below; ALL THREE run the full comparison, not KinoPaxPlus-only at any of them):
#   CountingStars   bufferSlope {1.0, 1.5} x bufferFloor {0.3, 0.5}
#                   explore_frac {0.1, 0.2}, cost_frac {0.4, 0.6, 0.8}
#                   optimalAcceptBudgeted PERMANENTLY ON (not an axis any more)
#                   = 24 points (FULL FACTORIAL) x 5 runs = 120 runs, per delta
#   KPAX                                                    = 1 point  x 5 runs
#   KinoPaxPlus                                             = 1 point  x 5 runs
#
# ONE COST METRIC THIS PASS (effort/COST_MODE=1 only -- length is dropped), one environment
# (zigzag), one full build per delta: 360 CountingStars runs total across the three deltas.
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
#    explore_frac AND cost_frac ARE BOTH SWEPT THIS PASS, now that the toggle axis is gone and there
#    is room to actually tune both shares alongside bufferSlope/bufferFloor.
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
#   9. First-solution time and cost, final cost. THE ACTUAL QUESTION THIS PASS ASKS: which
#      (bufferSlope, bufferFloor, explore_frac, cost_frac) point wins on time-to-first-solution AND
#      final cost, now that budgeting OPTIMAL is a settled, permanent part of the design.
#
# ALL THREE DELTAS RUN THE FULL COMPARISON THIS PASS -- NOT --only-kinopaxplus at any of them,
# unchanged from the pass that settled the toggle question. Tuning conclusions at one delta do not
# automatically hold at the others, so all three need the full grid, not KinoPaxPlus alone at the
# finer ones.
#
# Runs on zigzag only this pass, written to its own subfolder under
# Data/Benchmarks/CountingStars/zigzag/.
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
# THREE DELTAS RUN THIS PASS -- "large" and "fine" are paper_benchmark.cu's own current large/fine
# (copied by hand from run_paper_benchmark.sh; there is no cross-check enforcing they stay equal,
# so re-check both files if either one's deltas change again), plus this sweep's own "tiny",
# unchanged from earlier passes, all full comparison, per the header above.
#
# C_R1 STAYS AT 1 EVERYWHERE. NUM_R1_REGIONS = W_R1^3 * V_R1^3 has no C term, and this config sets
# C_DIM 0, so getRegion / getSubRegion skip the C dimension entirely -- raising C_R1 would change
# nothing at all. The control-side refinement rides on V_R1.
#
# Deltas (Model 1: W_DIM=3, C_DIM=0, V_DIM=3):
#   large   W_R1=7   C_R1=1  V_R1=3  ->   7^3 * 3^3 =   9,261   (full comparison)
#   fine    W_R1=16  C_R1=1  V_R1=4  ->  16^3 * 4^3 = 262,144   (full comparison)
#   tiny    W_R1=14  C_R1=1  V_R1=6  ->  14^3 * 6^3 = 592,704   (full comparison)
#
# "tiny" names the CELL, not the count: it is the finest delta this pass runs, at 592,704 regions.
# Watch it for the per-region arrays -- every NUM_R1_REGIONS allocation and every full-array fill
# scales with this, and graph_.updateVertices() runs a kernel over all of them with 64 sub-vertex
# reads each.
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

# Deltas: parallel arrays of label / W_R1 / C_R1 / V_R1. ALL THREE run the FULL comparison this
# pass (see DELTA_EXTRA_ARGS) -- a tuning conclusion at one delta is not assumed to hold at the
# others, so there is no "--only-kinopaxplus" arm to skip it with here. One build per
# (delta, cost metric), cached, so restoring or trimming the list changes only the loop bounds.
#
# "large" and "fine" are copied verbatim from run_paper_benchmark.sh's own current large/fine
# deltas (kept in step by hand -- there is no cross-check between the two sweep tools) -- this
# sweep's tuning conclusions are only useful for the paper comparison if they're measured at the
# same discretizations paper_benchmark.cu actually runs. "tiny" is NOT paper's own tiny -- it is
# this sweep's own, pre-existing tiny delta, confirmed to run cleanly for CountingStars/KPAX/
# KinoPaxPlus; paper_benchmark.cu's own tiny (W_R1=17, V_R1=5) currently hangs on KPAX in an
# open environment (a buffer-overflow bug in KPAX.cu/KinoPaxPlus.cu's goal-path reconstruction,
# only partially fixed so far), so it is deliberately NOT reused here.
DELTA_LABELS=("large" "fine" "tiny")
DELTA_W_R1S=(7 16 14)
DELTA_C_R1S=(1  1  1)   # inert for Model 1 (C_DIM 0); control refinement rides on V_R1
DELTA_V_R1S=(3  4  6)
DELTA_EXTRA_ARGS=("" "" "")

# --- Coarse delta only (uncomment to restore; comment out the four lines above) ---
# DELTA_LABELS=("large")
# DELTA_W_R1S=(7)
# DELTA_C_R1S=(1)
# DELTA_V_R1S=(3)
# DELTA_EXTRA_ARGS=("")

# Cost metric axis: label + COST_MODE (0 = workspace distance, 1 = control effort). EFFORT ONLY
# this pass -- length disabled, not removed; uncomment the line below to restore it.
COST_LABELS=("effort")
COST_MODES=(1)
# COST_LABELS=("length" "effort")
# COST_MODES=(0 1)

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
echo "  CountingStars v3.4 Sweep -- optimal-accept budgeting toggle"
echo "  Model: 1 (6D Double Integrator)"
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
echo "  CountingStars:  bufferSlope {1.0,1.5} x bufferFloor {0.3,0.5}"
echo "                  explore_frac {0.1,0.2}, cost_frac {0.4,0.6,0.8}"
echo "                  optimalAcceptBudgeted PERMANENTLY ON -- not an axis any more"
echo "                  = 24 points (full factorial) x ALL THREE deltas"
echo "                  Filenames: _bs<round(100*slope)>_bf<round(100*floor)>_ef<..>_cf<..>,"
echo "                  e.g. CountingStars_bs100_bf30_ef200_cf600."
echo "                  Earlier CSVs (_rgon/_rgoff, _abon/_aboff, _obon/_oboff tokens) cannot collide"
echo "                  with this shape, so they simply stop loading -- intended for three retired"
echo "                  axes, not a loss."
echo "                  B IS STILL A RAMP, RECOMPUTED EVERY ITERATION:"
echo "                    x = itr/fill_iters, B(x) = floor((slope*x + floor) * MAX_TREE_SIZE/fill_iters)"
echo "                  THREE DOORS PLUS A FLAT ADMISSION FLOOR, ALL BUDGETED. THE REGION-BEST"
echo "                  GUARANTEE DOOR IS GONE PERMANENTLY -- folded into CHEAPEST's reactivation"
echo "                  budget, no toggle left for it (see CS_DOORBIT_GUAR in CountingStars.cuh):"
echo "                    FRESHEST   explore_frac * B, from the least-populated regions"
echo "                    CHEAPEST   cost_frac * B, from the smallest cost distances -- OPTIMAL"
echo "                               candidates (distance 0, i.e. cost <= minCostsR1[r]) compete here"
echo "                               too, permanently (v3.4, see below), always at bucket 0. v3.3:"
echo "                               also competes for FRESHEST rather than returning early."
echo "                    ADMIT FLOOR (v3.3) every candidate at accept_floor = 1e-4, only when nothing"
echo "                               else admitted it -- a completeness guarantee, not a reach tool"
echo "                    REACT FLOOR every dormant node at react_floor = 1e-5, ON TOP of the budget"
echo "                  OPTIMAL-ACCEPT BUDGETING IS PERMANENT (v3.4). It ran as an on/off toggle in"
echo "                  the previous pass; that sweep confirmed folding OPTIMAL into the SAME"
echo "                  cost-distance histogram/cutoff CHEAPEST already spends against -- it always"
echo "                  lands in bucket 0 (csCostBucket(0.0f,...) == 0 for any distMax), so it has to"
echo "                  clear that cutoff/boundary-roll like anything else -- improves final cost /"
echo "                  time-to-first-solution. There is no unconditional admission left to compare"
echo "                  against. Read optimal_count (measured at distance 0) against admitted_cost"
echo "                  (actually admitted via CS_DOORBIT_OPTIMAL) -- their gap is how many optimal"
echo "                  candidates the budget did not have room for at a given grid point."
echo "                  REACTIVATION IS COST-SELECTIVE (v3.1, unchanged this pass): CleanCost weights"
echo "                  its own reactivation arm by cost, and that was the one cost mechanism this"
echo "                  line lacked -- the volumes already matched, so it was selectivity not"
echo "                  throughput. Part B is the only thing that re-expands the tree INTERIOR, which"
echo "                  is where cost refinement happens."
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
echo "                  FAN-OUT IS DOOR-COUNT (v3.3), FULL STOP: nodeBlocks = popcount(door), no"
echo "                  region-thinness signal and no swept boost size left -- the region-keyed rule"
echo "                  KPAXCap and CleanCost used is gone from this planner (and both baselines are"
echo "                  gone from this sweep -- see the file header)."
echo "                  READ FIRST: goal_frontier_size vs iteration (does the realized ramp match"
echo "                  the intended shape), then optimal_count against admitted_cost, then"
echo "                  budget_used/goal_frontier_size as a CURVE against a MOVING target, then"
echo "                  admitted_costdist against admitted_explore, then cost_cutoff_dist against"
echo "                  dist_max."
echo "  Score floor:    COUNTINGSTARS HAS NO SCORE FLOOR AND USES NO EPSILON: it never reads"
echo "                  vertexScores, h_scoreFloor_, h_nActive_ or regionCoverage in any decision."
echo "  Baselines: KPAX, KinoPaxPlus -- BOTH AT ALL THREE DELTAS this pass, not KinoPaxPlus-only at"
echo "             the finer ones (see the header for why)."
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
