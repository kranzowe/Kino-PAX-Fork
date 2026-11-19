#!/bin/bash
# =============================================================================
# KinoPaxSTAR Algorithm Comparison Runner
#
# Head-to-head of the STAR variants against the KPAX and KinoPaxPlus baselines, at fixed
# configurations (no sweeps), on both environments:
#
#   KPAX, KinoPaxPlus                 baselines
#   KinoPaxSTARNoGoalBias             no cost pruning
#   KinoPaxSTARTrue                   guarded stale-best pruning
#   KinoPaxSTARTrueAnc                guarded stale-best + ancestor chain
#   KinoPaxSTARWeightedCost           w = 0.9, k = 1, no cost pruning
#   KinoPaxSTARTrueWeightedCost       w = 0.9, k = 1, guarded stale-best
#   KinoPaxSTARTrueWeightedCostAnc    w = 0.9, k = 1, guarded chain
#
# "Guarded" means cost pruning only ever touches nodes admitted BECAUSE they were their region's
# minimum. Syclop-admitted explorers are never pruned -- without that guard (the retired
# KinoPaxSTARNoPruneAncestor) the entire exploration population froze on the first pruning pass.
#
# None of these carry goal-bias acceptance; that lives only in KinoPaxSTAR.
#
# 8 series x 10 runs x (2 envs x 2 cost metrics) = 320 runs, ~15 min per pass.
#
# COST_MODE is a compile-time #if inside edgeCost (include/helper/helper.cuh), so
# the cost metric cannot vary within one binary. This script therefore borrows
# run_delta_benchmark.sh's build-cache pattern: it writes config.h and builds once
# per cost metric, caching each binary under a metric-suffixed name, then runs
# them in a second pass. The metric rides into every output filename as the
# argv[1] delta label (large_length / large_effort).
#
# Large delta (Model 1: W_DIM=3, C_DIM=1, V_DIM=3):
#   W_R1=10  C_R1=1  V_R1=3  ->  NUM_R1_REGIONS = 10^3 * 3^3 = 27,000
#
# Original config.h is backed up and restored on exit/error.
#
# Usage:
#   cd scripts && bash run_comparison_benchmark.sh
#   cd scripts && bash run_comparison_benchmark.sh --skip-build   # run only (cached binaries)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"

# Single delta: label W_R1 C_R1 V_R1  (Large)
DELTA_LABEL="large"
DELTA_W_R1=10
DELTA_C_R1=1
DELTA_V_R1=3

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

REGIONS=$(( DELTA_W_R1**3 * DELTA_V_R1**3 ))

echo ""
echo "======================================================="
echo "  KinoPaxSTAR Algorithm Comparison"
echo "  Model: 1 (6D Double Integrator)"
echo "  Environments: ${ENV_NAMES[*]}  (separate output subfolders)"
echo "  Delta: ${DELTA_LABEL} | W_R1=${DELTA_W_R1} C_R1=${DELTA_C_R1} V_R1=${DELTA_V_R1} | Regions=${REGIONS}"
echo "  Cost metrics: ${COST_LABELS[*]}  (one build each)"
echo "  Variants:  NoGoalBias, True, TrueAnc, WeightedCost, TrueWeightedCost, TrueWeightedCostAnc"
echo "  Weighted point: w = 0.9, k = 1"
echo "  Baselines: KPAX, KinoPaxPlus"
echo "  Baselines: KPAX, KinoPaxPlus"
echo "======================================================="

# =============================================================================
# BUILD — compile the Large delta config once per cost metric, caching each binary
# =============================================================================
if [ "$SKIP_BUILD" = false ]; then
    for i in "${!COST_LABELS[@]}"; do
        CL="${COST_LABELS[$i]}"
        CM="${COST_MODES[$i]}"

        echo ""
        echo "=== BUILDING (delta=${DELTA_LABEL}, cost=${CL}, COST_MODE=${CM}, Regions=${REGIONS}) ==="

        write_config "$DELTA_W_R1" "$DELTA_C_R1" "$DELTA_V_R1" "$CM"

        cd "$BUILD_DIR"
        # shellcheck disable=SC2086
        cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_COMPILER_FLAGS 2>&1 | tail -5
        make KinoPaxStarComparison -j"$(nproc)" 2>&1 | tail -20
        # Cache the binary under a metric-suffixed name so the run phase needs no rebuild
        cp KinoPaxStarComparison "KinoPaxStarComparison_${CL}"
        cd "$PROJECT_DIR"
    done
else
    echo ""
    echo "=== SKIPPING BUILD PHASE (using cached binaries) ==="
    cd "$BUILD_DIR"
    for CL in "${COST_LABELS[@]}"; do
        if [ ! -f "KinoPaxStarComparison_${CL}" ]; then
            echo "ERROR: Cached binary not found: KinoPaxStarComparison_${CL}"
            echo "Run without --skip-build first to create cached binaries."
            exit 1
        fi
    done
    echo "  All cached binaries found."
    cd "$PROJECT_DIR"
fi

# =============================================================================
# RUN — one pass per cost metric, using the cached binaries
# =============================================================================
# --dump-viz writes run-0's full tree per variant (+ meta.csv) for the tree-growth /
# R1-density visualization. OFF by default here: 25 variants x 2 builds x 2 envs would dump 100 full
# trees of up to MAX_TREE_SIZE nodes each. Enable with DUMP_VIZ=1 bash run_comparison_benchmark.sh
VIZ_FLAG=""
if [ "${DUMP_VIZ:-0}" != "0" ]; then
    VIZ_FLAG="--dump-viz"
fi

cd "$BUILD_DIR"
for CL in "${COST_LABELS[@]}"; do
    for i in "${!ENV_NAMES[@]}"; do
        EN="${ENV_NAMES[$i]}"
        EO="${ENV_OBSTACLES[$i]}"
        echo ""
        echo "=== RUNNING (delta=${DELTA_LABEL}, cost=${CL}, Env=${EN}) ==="
        # argv[1] carries the discretization and the cost metric, so it lands in every output
        # filename as _delta${DELTA_LABEL}_${CL}; argv[3] selects the per-environment subfolder.
        "./KinoPaxStarComparison_${CL}" "${DELTA_LABEL}_${CL}" "$EO" "$EN" $VIZ_FLAG
    done
done
cd "$PROJECT_DIR"

echo ""
echo "======================================================="
echo "  COST TUNING SWEEP COMPLETE"
echo "======================================================="
for EN in "${ENV_NAMES[@]}"; do
    echo "Results in: $BUILD_DIR/Data/Benchmarks/KinoPaxStarComparison/${EN}/"
done
echo "Plot each environment separately: cd into its folder, set envName at the top of"
echo "scripts/plot_comparison_scatter.m to match, then run it by name."
echo "Plot with:  scripts/process_cost_tuning_and_plot.m (run it from that directory)"
if [ "${DUMP_VIZ:-0}" != "0" ]; then
    echo "Viz dumps:  $BUILD_DIR/Data/Benchmarks/KinoPaxStarComparison/viz/  (visualize with scripts/visualize_tree_growth.m)"
fi
echo "Config.h will be restored to original on exit."
