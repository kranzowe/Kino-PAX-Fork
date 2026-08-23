#!/bin/bash
# =============================================================================
# KinoPaxSTAR Cost Tuning Sweep Runner
#
# Tuning sweep for KinoPaxSTARCleanCost, plus KinoPaxSTARTrue / KPAXCap cap sweeps and the
# KPAX / KinoPaxPlus baselines, on both environments:
#
#
# Per (environment, cost metric, delta):
#   KinoPaxSTARCleanCost   r2 {on, off} x w {0.1, 0.5, 0.9, 1.0} x k {1, 16} x cap {0.03, 0.1, 1.0}
#                          = 42 points x 3 runs = 126 runs
#                          (42, not 2*4*2*3 = 48: at w = 1 the cost term vanishes from
#                           weightedAccept, so only k = 1 runs there -- the other six points would
#                           be the same rule differing only by RNG stream)
#   KinoPaxSTARTrue        cap {0.03, 0.1, 0.3, 1.0} = 4 points x 3 runs = 12 runs
#   KPAXCap                cap {0.03, 0.1, 0.3, 1.0} = 4 points x 5 runs = 20 runs
#   KPAX                   baseline, 5 runs  -- KEEPS THE LEGACY EPSILON SCORE FLOOR
#   KinoPaxPlus            baseline, 5 runs
#   => 168 runs per (environment, cost metric)
#
# r2 IS THE R2 SUB-REGION SEEDING FREE PASS, now a swept arm. On (KPAX's behaviour): a candidate
# claiming a virgin R2 sub-region is admitted unconditionally, bypassing the weighted roll. Off: it
# takes the same roll as everything else -- the KinoPaxSTARnoseed condition (pSeed = 0). Propagate
# marks activeSubVertices either way, so r2_coverage_pct stays comparable across both arms and the
# pair measures how much of the exploration is seeding rather than the Syclop score.
#
# TWO NORMALIZATION FIXES land in this pass, which is why k and cap are both re-opened:
#   * Graph's Syclop floor becomes 1/N_active (the mean share) instead of a fixed EPSILON = 1e-2,
#     which exceeded the score it floored by ~270x and capped the number of discriminated regions
#     at 1/EPSILON = 100 at ANY grid size. OPT-IN: KPAXCap / TrueStar / CleanCost take it, KPAX
#     deliberately does not, so KPAX remains an unmodified baseline.
#   * CleanCost drops P_floor and switches to costProbExpGlobal -- the region's own minimum stays
#     the reference, but the SCALE is global, so a cost excess means the same thing everywhere
#     instead of being pinned at x ~ 1 in every region by construction.
# Both scales are now logged per iteration as score_floor / cost_scale.
#
# All three capped planners sweep the SAME cap values, so a cap reads across CleanCost / TrueStar /
# KPAXCap directly.
#
# cap = 0.03 IS THE DERIVED OPERATING POINT. After the acceptance fold, each frontier node offers
# repeat * h_activeBlockSize_ candidates to one rule, so the per-node branching factor carries a
# blockSize term; holding it near 1 gives cap ~ 1/blockSize = 1/32 = 0.03125, of which 0.03 is the
# exact-label neighbour. The finer deltas run ONLY that point (--single-cap): the cap sweep proper
# happens at "large", and the finer ones measure the derived value so the deltas overlay at a
# matched cap.
#
# CleanCost makes exactly ONE acceptance decision, in the accept kernel:
#
#     P = cap * min(1, w*(vertexScore + fAccept) + (1-w)*costProbExp(k) + P_floor)
#
# with region-best and fresh-R2-sub-region candidates exempt. Its predecessor
# KinoPaxSTARWeightedCost also ran a propagate-time filter capped at 0.1 that sat silently upstream
# of w; folding that away is what this sweep is retuning for. cap is the explicit replacement
# throttle and is applied at BOTH the accept kernel and Part-B reactivation.
#
# TrueStar keeps the plain KPAX Syclop roll but scales the region score by cap at both acceptance
# points (fAccept unscaled), with the guarded stale-best cost prune fixed on.
#
# KPAXCap is stock KPAX with that SAME cap and nothing else -- the control arm for the cap. CleanCost
# at w = 1 is NOT KPAX: it applies the cap AND decides after graph_.updateVertices(), reading scores
# already penalised for the batch being judged (computeVertexScores_kernel divides by
# 1 + counterArray^2, cumulative over the run). KPAXCap still decides inside propagate on pre-jump
# scores, so KPAX / KPAXCap / CleanCost-at-w=1 separates the cap's effect from the boundary's.
# Its caps are matched to TrueStar's so the two sweeps line up point-for-point.
#
# Algorithm-vs-algorithm comparison lives in run_comparison_benchmark.sh; this script is the
# tuning surface.
#
# Runs on BOTH environments (house and zigzag), each written to its own subfolder under
# Data/Benchmarks/KinoPaxStarCostTuning/<env>/ so they can be plotted independently.
#
# This pass: 168 runs per (environment, cost metric); 1 env x 2 metrics = 336 runs total.
# At the 6 s per-run cap that is ~17 min per metric, ~34 min overall, plus 2 builds.
#
# THREE DISCRETIZATIONS, ALL RUNNING THE FULL PLANNER SET. NUM_R1_REGIONS is compile-time
# (config.h), so each needs its own binary: this script builds the delta x cost-metric matrix
# (6 binaries) and runs the two finer deltas with --single-cap.
#
# "fine" and "fine_control" are a CONTROLLED PAIR: identical 216,000 region count, refined in
# different subspaces -- workspace (W_R1 10 -> 20) vs velocity (V_R1 3 -> 6).
#
# C_R1 STAYS AT 1 EVERYWHERE. NUM_R1_REGIONS = W_R1^3 * V_R1^3 has no C term, and this config sets
# C_DIM 0, so getRegion / getSubRegion skip the C dimension entirely -- raising C_R1 would change
# nothing at all. The control-side refinement rides on V_R1.
#
# COST_MODE is a compile-time #if inside edgeCost (include/helper/helper.cuh), so
# the cost metric cannot vary within one binary. This script therefore borrows
# run_delta_benchmark.sh's build-cache pattern: it writes config.h and builds once
# per (delta, cost metric), caching each binary under a suffixed name, then runs
# them in a second pass. Both labels ride into every output filename as the
# argv[1] delta label (large_length / large_effort / fine_length / fine_effort).
#
# Deltas (Model 1: W_DIM=3, C_DIM=0, V_DIM=3):
#   large         W_R1=10  C_R1=1  V_R1=3  ->  10^3 * 3^3 =  27,000
#   fine          W_R1=20  C_R1=1  V_R1=3  ->  20^3 * 3^3 = 216,000  (workspace-refined)
#   fine_control  W_R1=10  C_R1=1  V_R1=6  ->  10^3 * 6^3 = 216,000  (velocity-refined)
#
# Original config.h is backed up and restored on exit/error.
#
# Usage:
#   cd scripts && bash run_cost_tuning_sweep.sh
#   cd scripts && bash run_cost_tuning_sweep.sh --skip-build   # run only (cached binaries)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"

# Deltas: parallel arrays of label / W_R1 / C_R1 / V_R1.
# Index 0 ("large") runs the full sweep. Every later index runs KinoPaxPlus ONLY, so the baseline
# can be measured at a finer discretization without re-running the whole grid there.
# SCOPE: this pass runs the coarse delta only. The full three-delta set is preserved below --
# uncomment the second block and comment the first to restore it. Nothing else in the script needs
# to change: the build/run matrix, the per-delta binary caching and --single-cap all still loop.
DELTA_LABELS=("large")
DELTA_W_R1S=(10)
DELTA_C_R1S=(1)         # inert for Model 1 (C_DIM 0); control refinement rides on V_R1
DELTA_V_R1S=(3)
DELTA_EXTRA_ARGS=("")

# --- Full delta set (uncomment to restore; comment out the four lines above) ---
# DELTA_LABELS=("large" "fine" "fine_control")
# DELTA_W_R1S=(10 20 10)
# DELTA_C_R1S=(1  1  1)
# DELTA_V_R1S=(3  3  6)
# Index 0 (the coarse delta) sweeps the cap axis and gets the viz flag appended below; the finer
# deltas run only the derived cap so the three overlay at a matched point.
# DELTA_EXTRA_ARGS=("" "--single-cap" "--single-cap")

# Cost metric axis: label + COST_MODE  (0 = workspace distance, 1 = control effort)
COST_LABELS=("length" "effort")
COST_MODES=(0 1)

# Environments (obstacles already in [0,1]^3 for Model 1). Each gets its own output subfolder.
# narrowPassage is a wall at x in [0.3, 0.5] spanning all z, split by a gap at y in [0.49, 0.51] --
# 0.02 wide against an agent diameter of 0.01 (AGENT_RADIUS 0.005). The benchmark's start
# (0.1, 0.08, 0.05) and goal (0.8, 0.95, 0.9) are clear of both boxes and on opposite sides of the
# wall, so no endpoint change is needed -- but expect low success rates there, and read the
# success-rate subplot alongside the cost bars (unsolved runs are dropped from the cost mean).
# SCOPE: house only this pass. Full set preserved below -- uncomment to restore.
ENV_NAMES=("house")
ENV_OBSTACLES=("../include/config/obstacles/house/obstacles.csv")

# --- Full environment set (uncomment to restore; comment out the two lines above) ---
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
echo "  KinoPaxSTAR Cost Tuning Sweep"
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
echo "  CleanCost:      r2 {on,off} x w {0.1,0.5,0.9,1.0} x k {1,16} x cap {0.03,0.1,1.0} = 42 points"
echo "  TrueStar:       cap {0.03, 0.1, 0.3, 1.0} = 4 points"
echo "  KPAXCap:        cap {0.03, 0.1, 0.3, 1.0} = 4 points"
echo "  Score floor:    dynamic 1/N_active for KPAXCap/TrueStar/CleanCost; legacy EPSILON for KPAX"
echo "  Any finer delta added back runs only cap = 0.1 (CAP_DERIVED) via --single-cap"
echo "  Baselines: KPAX, KinoPaxPlus (large + fine)"
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
            make KinoPaxStarCostTuningSweep -j"$(nproc)" 2>&1 | tail -20
            # Cache under a (delta, metric)-suffixed name so the run phase needs no rebuild
            cp KinoPaxStarCostTuningSweep "KinoPaxStarCostTuningSweep_${DL}_${CL}"
            cd "$PROJECT_DIR"
        done
    done
else
    echo ""
    echo "=== SKIPPING BUILD PHASE (using cached binaries) ==="
    cd "$BUILD_DIR"
    for DL in "${DELTA_LABELS[@]}"; do
        for CL in "${COST_LABELS[@]}"; do
            if [ ! -f "KinoPaxStarCostTuningSweep_${DL}_${CL}" ]; then
                echo "ERROR: Cached binary not found: KinoPaxStarCostTuningSweep_${DL}_${CL}"
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
# R1-density visualization. OFF by default here: 25 variants x 2 builds x 2 envs would dump 100 full
# trees of up to MAX_TREE_SIZE nodes each. Enable with DUMP_VIZ=1 bash run_cost_tuning_sweep.sh
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
            "./KinoPaxStarCostTuningSweep_${DL}_${CL}" "${DL}_${CL}" "$EO" "$EN" $PASS_FLAGS
        done
    done
done
cd "$PROJECT_DIR"

echo ""
echo "======================================================="
echo "  COST TUNING SWEEP COMPLETE"
echo "======================================================="
for EN in "${ENV_NAMES[@]}"; do
    echo "Results in: $BUILD_DIR/Data/Benchmarks/KinoPaxStarCostTuning/${EN}/"
done
echo "Plot each environment separately: cd into its folder, set envName at the top of"
echo "scripts/process_cost_tuning_and_plot.m to match, then run it by name."
echo "Plot with:  scripts/process_cost_tuning_and_plot.m (run it from that directory)"
if [ "${DUMP_VIZ:-0}" != "0" ]; then
    echo "Viz dumps:  $BUILD_DIR/Data/Benchmarks/KinoPaxStarCostTuning/viz/  (visualize with scripts/visualize_tree_growth.m)"
fi
echo "Config.h will be restored to original on exit."
