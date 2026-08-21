#!/bin/bash
# =============================================================================
# KinoPaxSTAR Cost Tuning Sweep Runner
#
# Tuning sweep for KinoPaxSTARCleanCost, plus KinoPaxSTARTrue / KPAXCap cap sweeps and the
# KPAX / KinoPaxPlus baselines, on both environments:
#
#   KinoPaxSTARCleanCost   w {0.8, 0.85, 0.9, 0.95} x k {1, 2, 4} x cap {0.1, 0.2, 0.3, 0.4} = 48
#                          plus w = 1.0 x k = 1 x cap {0.1, 0.2, 0.3, 0.4}                   =  4
#                          = 52 points x 3 runs = 156 runs
#   KinoPaxSTARTrue        cap {0.25, 0.5} = 2 points x 3 runs =  6 runs
#   KPAXCap                cap {0.25, 0.5} = 2 points x 5 runs = 10 runs
#   KPAX                   baseline, 5 runs
#   KinoPaxPlus            baseline, 5 runs at the "large" delta
#   KinoPaxPlus (fine)     the SAME baseline at a finer discretization, 5 runs
#
# w = 1.0 IS RUN AT k = 1 ONLY: at w = 1 the rule is min(1, 1*P_syclop + 0*P_cost + floor), so the
# cost term drops out and the three k rungs would be one rule differing only by RNG stream.
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
# = 182 runs per (environment, cost metric); 4 such passes = 728 runs total.
# At the 6 s per-run cap that is ~18 min per pass, ~1.2 h worst case overall.
#
# TWO DISCRETIZATIONS. NUM_R1_REGIONS is compile-time (config.h), so the extra finer-grid
# KinoPaxPlus series needs its own binary. This script therefore builds the delta x cost-metric
# matrix (4 binaries) and runs the "fine" ones with --only-kinopaxplus, which skips KPAX, the
# CleanCost grid and TrueStar. "fine" matches the med_large rung of run_delta_benchmark.sh.
#
# COST_MODE is a compile-time #if inside edgeCost (include/helper/helper.cuh), so
# the cost metric cannot vary within one binary. This script therefore borrows
# run_delta_benchmark.sh's build-cache pattern: it writes config.h and builds once
# per (delta, cost metric), caching each binary under a suffixed name, then runs
# them in a second pass. Both labels ride into every output filename as the
# argv[1] delta label (large_length / large_effort / fine_length / fine_effort).
#
# Deltas (Model 1: W_DIM=3, C_DIM=1, V_DIM=3):
#   large  W_R1=10  C_R1=1  V_R1=3  ->  NUM_R1_REGIONS = 10^3 * 3^3 =  27,000
#   fine   W_R1=20  C_R1=1  V_R1=3  ->  NUM_R1_REGIONS = 20^3 * 3^3 = 216,000
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
DELTA_LABELS=("large" "fine")
DELTA_W_R1S=(10 20)
DELTA_C_R1S=(1 1)
DELTA_V_R1S=(3 3)
# Extra argv passed to each delta's binary. Index 0 gets the viz flag (appended below); the rest
# are KinoPaxPlus-only passes.
DELTA_EXTRA_ARGS=("" "--only-kinopaxplus")

# Cost metric axis: label + COST_MODE  (0 = workspace distance, 1 = control effort)
COST_LABELS=("length" "effort")
COST_MODES=(0 1)

# Environments (obstacles already in [0,1]^3 for Model 1). Each gets its own output subfolder.
ENV_NAMES=("house" "zigzag")
ENV_OBSTACLES=("../include/config/obstacles/house/obstacles.csv"
               "../include/config/obstacles/zigzag/obstacles.csv")

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
echo "  CleanCost grid: w {0.8, 0.85, 0.9, 0.95, 1.0} x k {1, 2, 4} x cap {0.1, 0.2, 0.3, 0.4}"
echo "                  = 52 points (w=1.0 at k=1 only)"
echo "  TrueStar:       cap {0.25, 0.5} = 2 points"
echo "  KPAXCap:        cap {0.25, 0.5} = 2 points"
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
