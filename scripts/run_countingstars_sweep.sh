#!/bin/bash
# =============================================================================
# CountingStars v2 Sweep Runner
#
# CountingStars v2 (ONE GLOBAL NODE BUDGET, filled in priority order), against the baselines it has
# to beat: KPAX, KPAXCap, KinoPaxPlus, and one tuned KinoPaxSTARCleanCost point. COMBO and TrueStar
# are deliberately NOT in this sweep -- COMBO is the thing being replaced and its own sweep still
# exists; TrueStar answers a cap question this planner does not ask.
#
# Per (environment, cost metric) at the COARSE delta:
#   CountingStars   selection_key {ord, dist} x goal_frontier_size {200, 2000, 6000, 10000}
#                   x explore_frac {0.01, 0.5, 0.9} x maxBlocks {1, 4}
#                   = 48 points (FULL FACTORIAL, coarse delta only) x 3 runs = 144 runs
#
#                   maxBlocks is FIXED AT 15 and deliberately NOT an axis. In v1 it was the height
#                   of a geometric fan-out ramp and had to be swept against the ramp's width; v2 has
#                   no ramp -- blockBudget = maxBlocks * B, optimal nodes take maxBlocks each and
#                   everyone else splits the rest -- so maxBlocks and B are the same knob twice.
#   CleanCost       r2 OFF, w 0.9, k 1, cap 0.03            = 1 point  x 3 runs
#   KPAXCap         cap {0.03}                              = 1 point  x 5 runs
#   KPAX                                                    = 1 point  x 5 runs
#   KinoPaxPlus                                             = 1 point  x 5 runs
#
# SELECTION KEY IS THE HEADLINE AXIS THIS PASS. The second door -- the one that spends explore_frac
# of what the optimal door left -- can rank its candidates two ways:
#
#   ord   regionNodeCount[r]: prefers the LEAST-POPULATED regions. v2's behaviour, the control arm.
#   dist  (cost - minCostsR1[r]) / costScale: prefers candidates CLOSEST TO THEIR OWN region's best,
#         ranked by an exact sort rather than a histogram.
#
# BOTH KEYS SPEND EXACTLY THE SAME BUDGET by construction, so any difference in outcome is the
# choice of WHICH candidates. Check admitted_explore ord vs dist at matched (B, frac, mb) FIRST: if
# it differs, the arms are not spending the same budget and nothing downstream is interpretable.
#
# dist is NOT greedy best-first -- distance is per-region normalised, so every region that received
# candidates contributes its best at distance 0 and spatial coverage survives structurally. The risk
# case is that it still concentrates on already-good regions, coverage suffers, and the goal is
# found LATER despite a better final cost. r2_coverage_pct is the corroborating column.
#
# COST ACCEPTANCE IS PERMANENT. An earlier pass carried a cost_accept toggle that removed both
# cost-driven doors -- OPTIMAL (accept pass 2) and the Part B region-best GUARANTEE -- to test
# whether they were costing time to first solution. That experiment is over and the doors stay:
# they are what makes the search converge on cost at all, and without them nothing preferentially
# expands cheap nodes. CSVs carrying a `_ca` token are from that pass and no longer load.
#
# goal_frontier_size is the design's primitive: B is the TARGET the doors fill in priority order,
# so F is an INPUT and propagations-per-node is the output. maxBlocks is the OTHER half of that --
# B sets frontier size, maxBlocks sets propagations per node. They are independent knobs.
#
# FAN-OUT IS REGION-KEYED THIS PASS, and that is the change under test. Blocks now follow REGION
# THINNESS (validVertexCounter[r] < 10 ? maxBlocks : 1, reactivations always 1) -- KPAXCap's and
# CleanCost's exact rule. The previous flat rule gave every frontier node the same count, so at
# B = 10000 / maxBlocks = 16 the planner produced ~224 candidates per node admitted against the
# ancestors' ~32. That was the bulk of the runtime gap, and it was the propagation kernel rather
# than bookkeeping.
#
# maxBlocks = 1 reproduces the ancestors' steady state exactly: their counter is cumulative and
# gains ~32 per frontier node per iteration, so regions cross the threshold almost at once and
# their x15 is a one-shot burst on new ground. The old {16, 32} axis never sampled that regime --
# and the buffer ceiling clamped both to roughly the same realised rep, so it was close to inert.
#
# KinoPaxPlus divides the whole propagation budget over a frontier its parent-chain pruning keeps
# tiny -- bf = MAX_TREE_SIZE/(F*32), so 40,000 propagations per node at F = 10. That is the number
# prop_attempted/frontier_size is read against.
#
# THE GRID STRADDLES NUM_R1_REGIONS (27,000 at the coarse delta), and that is the point of its
# shape. TWO doors are uncapped and BOTH are bounded by the region count rather than by B: the
# OPTIMAL door (at most one region best per region per iteration) and the GUARANTEE (at most one
# node per uncovered region). So B binds only ABOVE that count.
#
# EVERY B ON THIS GRID IS BELOW THAT COUNT (200 / 2000 / 6000 / 10000 against 27,000), so B stops
# binding once nActive passes it. THAT IS THE POINT: B binds EARLY and then stops, at an iteration
# that moves with B -- almost at once at 200, much later at 10000 -- and early is exactly where
# time-to-first-solution is decided.
#
# Read budget_used/B as a CURVE over iterations, not a single number: the iteration where it
# crosses 1 IS the measurement, and a late-run overshoot is expected at every point. If the four
# curves are indistinguishable even early, capping the guarantee (KinoPaxPlus's hysteresis is the
# precedent) is the next lever, not a different B.
#
# READ IN THIS ORDER:
#   1. budget_used vs goal_frontier_size. THE CLAIM THE WHOLE DESIGN RESTS ON. A persistent
#      shortfall means a door is not filling its share; an overshoot means the two uncapped doors
#      already exceeded B on their own -- expected at the two low points, not at the two high ones.
#   2. prop_attempted / frontier_size against KinoPaxPlus's bf. Should move with maxBlocks
#      INDEPENDENTLY of B: B sets frontier size, maxBlocks sets propagations per node.
#   3. ord_cutoff over the run. Rising means regions are filling and freshness is getting scarce,
#      which is expected. Pinned at 0 means explore_frac is doing nothing; pinned at 256 means the
#      whole candidate pool is fresh enough and explore_frac is not binding either.
#
# THE TWO FINER DELTAS RUN KINOPAXPLUS ONLY (--only-kinopaxplus), and that is the point of having
# them: KinoPaxPlus is the planner whose whole advantage is a tiny frontier at a fine
# discretisation, so it is the one baseline that must be measured at all three. Re-running the
# CountingStars grid there would triple the sweep to answer a question the coarse delta already
# answers.
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
# Deltas (Model 1: W_DIM=3, C_DIM=0, V_DIM=3):
#   large   W_R1=10  C_R1=1  V_R1=3  ->  10^3 * 3^3 =  27,000   (full sweep)
#   fine    W_R1=20  C_R1=1  V_R1=3  ->  20^3 * 3^3 = 216,000   (KinoPaxPlus only)
#   tiny    W_R1=14  C_R1=1  V_R1=6  ->  14^3 * 6^3 = 592,704   (KinoPaxPlus only)
#
# "tiny" names the CELL, not the count: it is the FINEST of the three at 592,704 regions. Watch it
# for the per-region arrays -- every NUM_R1_REGIONS allocation and every full-array fill scales with
# this, and graph_.updateVertices() runs a kernel over all of them with 64 sub-vertex reads each.
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

# Deltas: parallel arrays of label / W_R1 / C_R1 / V_R1.
# Index 0 runs the full sweep; every later index runs KinoPaxPlus only. One build per (delta, cost
# metric), cached, so restoring or trimming the list changes only the loop bounds.
DELTA_LABELS=("large" "fine" "tiny")
DELTA_W_R1S=(10 20 14)
DELTA_C_R1S=(1  1  1)   # inert for Model 1 (C_DIM 0); control refinement rides on V_R1
DELTA_V_R1S=(3  3  6)
# Index 0 runs the FULL sweep -- CountingStars grid, KPAX, KPAXCap, KinoPaxPlus, CleanCost.
# Indices 1 and 2 run KINOPAXPLUS ONLY, which is the point of having them: KinoPaxPlus is the
# planner whose whole advantage is a tiny frontier at a fine discretisation, so it is the one
# baseline that has to be measured at all three. Re-running the CountingStars grid there would
# triple the sweep to answer a question the coarse delta already answers.
DELTA_EXTRA_ARGS=("" "--only-kinopaxplus" "--only-kinopaxplus")

# --- Coarse delta only (uncomment to restore; comment out the six lines above) ---
# DELTA_LABELS=("large")
# DELTA_W_R1S=(10)
# DELTA_C_R1S=(1)
# DELTA_V_R1S=(3)
# DELTA_EXTRA_ARGS=("")

# Cost metric axis: label + COST_MODE  (0 = workspace distance, 1 = control effort)
COST_LABELS=("length")
COST_MODES=(0)

# Environments (obstacles already in [0,1]^3 for Model 1). Each gets its own output subfolder.
# narrowPassage is a wall at x in [0.3, 0.5] spanning all z, split by a gap at y in [0.49, 0.51] --
# 0.02 wide against an agent diameter of 0.01 (AGENT_RADIUS 0.005). The benchmark's start
# (0.1, 0.08, 0.05) and goal (0.8, 0.95, 0.9) are clear of both boxes and on opposite sides of the
# wall, so no endpoint change is needed -- but expect low success rates there, and read the
# success-rate subplot alongside the cost bars (unsolved runs are dropped from the cost mean).
# SCOPE: zigzag and narrowPassage this pass. Full set preserved below -- uncomment to restore.
#
# narrowPassage is a wall at x in [0.3, 0.5] spanning all z, split by a gap at y in [0.49, 0.51] --
# 0.02 wide against an agent diameter of 0.01 (AGENT_RADIUS 0.005). Expect low success rates there,
# and read the success-rate subplot alongside the cost bars: unsolved runs are dropped from the cost
# mean rather than penalised, so a config that solved once cheaply can look best.
ENV_NAMES=("house")
ENV_OBSTACLES=("../include/config/obstacles/house/obstacles.csv")

# --- Full environment set (uncomment to restore; comment out the block above) ---
# ENV_NAMES=("house" "zigzag" "narrowPassage")
# ENV_OBSTACLES=("../include/config/obstacles/house/obstacles.csv"
#                "../include/config/obstacles/zigzag/obstacles.csv"
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
#define MAX_ITER 300
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
echo "  CountingStars Sweep"
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
echo "  CountingStars:  selection_key {ord,dist} x goal_frontier_size {200,2000,6000,10000}"
echo "                  x explore_frac {0.01,0.5,0.9} x maxBlocks {1,4}"
echo "                  = 48 points (full factorial, coarse delta only)"
echo "                  ord  = least-populated regions (v2, control arm)"
echo "                  dist = closest to their own region's best, exact sort"
echo "                  Both spend the SAME budget -- check admitted_explore matches"
echo "                  across the two arms before reading anything else."
echo "                  Filenames gain _kord / _kdist."
echo "                  Filenames: _B<budget>_f<round(1000*frac)>_mb<maxBlocks>,"
echo "                  e.g. CountingStars_B10000_f100_mb16."
echo "                  THE _ca TOKEN IS GONE with the cost_accept toggle. CSVs carrying it"
echo "                  are from that experiment, will not match, and are simply skipped --"
echo "                  clear or move them out of the output folder to keep it readable."
echo "                  maxBlocks FIXED at 15 -- with the geometric ramp gone it is the same"
echo "                  knob as B seen twice, so sweeping it would duplicate the axis."
echo "                  ONE GLOBAL NODE BUDGET, filled in priority order. Four doors:"
echo "                    OPTIMAL    distance 0, i.e. cost <= minCostsR1[r].  UNCAPPED, first claim"
echo "                    FRESHEST   explore_frac of what is LEFT, to the least-populated regions"
echo "                    GUARANTEE  each active region's best, if OPTIMAL did not already cover it"
echo "                    DRAW       uniform over the rest, filling whatever the budget has left"
echo "                  COST ACCEPTANCE IS PERMANENT -- the OPTIMAL door and the Part B"
echo "                  region-best GUARANTEE always run. Those two are the ONLY uncapped"
echo "                  doors and both are bounded by NUM_R1_REGIONS (27,000 at the coarse"
echo "                  delta) rather than by B, so B binds only ABOVE that count; below it"
echo "                  budget_used runs over B. 2000 and 10000 are SOFT points where the"
echo "                  two should converge; 50000 is where B genuinely binds."
echo "                  FAN-OUT IS REGION-KEYED: maxBlocks only for a node landing in a region"
echo "                  with < 10 valid propagations, else 1; reactivations always 1. That is"
echo "                  KPAXCap's and CleanCost's rule, ported so the runtime comparison is"
echo "                  like-for-like. maxBlocks = 1 reproduces their steady state exactly."
echo "                  READ FIRST: frontier_repeat_size/frontier_size (realised mean rep --"
echo "                  must be 1 at maxBlocks=1), then prop_attempted/budget_used (candidates"
echo "                  per admitted node; was ~224, target ~32 like the ancestors), then time"
echo "                  to first solution against KPAXCap and CleanCost."
echo "  CleanCost:      r2 OFF, w 0.9, k 1, cap 0.03 = 1 point (baseline)"
echo "  KPAXCap:        cap {0.03} = 1 point"
echo "  Score floor:    dynamic 1/N_active for KPAXCap/CleanCost; legacy EPSILON for KPAX."
echo "                  COUNTINGSTARS HAS NO SCORE FLOOR AND USES NO EPSILON: it never reads"
echo "                  vertexScores, h_scoreFloor_, h_nActive_ or regionCoverage in any decision."
echo "  Baselines: KPAX (coarse delta), KinoPaxPlus (ALL THREE deltas -- the point of having them)"
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
            # Only the full-sweep delta dumps viz; a KinoPaxPlus-only pass has nothing extra to show.
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
