#!/bin/bash
# =============================================================================
# Tree Growth Dump Runner
#
# Builds and runs TreeGrowthDump, which dumps the planner's full tree after each
# of the first 8 iterations for four configurations on the zigzag corridor:
#
#   ancestor_off  KinoPaxSTARNoPruneAncestor, h_ancestorPrune_ = 0  (== stock NoPrune)
#   ancestor_on   KinoPaxSTARNoPruneAncestor, h_ancestorPrune_ = 2  (memoized ancestor chain)
#   KPAX          pure explorer reference
#   KinoPaxPlus   pure optimizer reference (source of the original ancestor pruning)
#
# 32 CSVs + meta.csv, rendered by scripts/plot_tree_growth_iters.m.
#
# The config.h rewrite is NOT optional: the checked-in config.h is a MODEL 3 (12D
# quad, W_MAX 100) config, while the zigzag obstacles and every MATLAB script
# assume MODEL 1 in [0,1]^3. Original config.h is backed up and restored on exit.
#
# Built with COST_MODE 1 (control effort) -- the metric where KinoPaxPlus's edge
# over the STAR variants showed up. Change COST_MODE below for path length.
#
# Usage:
#   cd scripts && bash run_tree_growth_dump.sh
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

# Cost metric for this build: 1 = control effort, 0 = workspace path length
COST_MODE_VAL=1

# Single environment (obstacles already in [0,1]^3 for Model 1)
ENV_NAME="zigzag"
ENV_OBSTACLES="../include/config/obstacles/zigzag/obstacles.csv"

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
echo "  Tree Growth Dump"
echo "  Model: 1 (6D Double Integrator)"
echo "  Environment: ${ENV_NAME}"
echo "  Delta: ${DELTA_LABEL} | W_R1=${DELTA_W_R1} C_R1=${DELTA_C_R1} V_R1=${DELTA_V_R1} | Regions=${REGIONS}"
echo "  Cost metric: control effort (COST_MODE=${COST_MODE_VAL})"
echo "  Configs: ancestor_off, ancestor_on, KPAX, KinoPaxPlus"
echo "  Iterations: 8 each"
echo "======================================================="

# =============================================================================
# BUILD
# =============================================================================
echo ""
echo "=== BUILDING (delta=${DELTA_LABEL}, COST_MODE=${COST_MODE_VAL}, Regions=${REGIONS}) ==="

write_config "$DELTA_W_R1" "$DELTA_C_R1" "$DELTA_V_R1" "$COST_MODE_VAL"

cd "$BUILD_DIR"
# shellcheck disable=SC2086
cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_COMPILER_FLAGS 2>&1 | tail -5
make TreeGrowthDump -j"$(nproc)" 2>&1 | tail -20
cd "$PROJECT_DIR"

# =============================================================================
# RUN
# =============================================================================
echo ""
echo "=== RUNNING (Env=${ENV_NAME}) ==="

cd "$BUILD_DIR"
./TreeGrowthDump "$ENV_OBSTACLES" "$ENV_NAME"
cd "$PROJECT_DIR"

echo ""
echo "======================================================="
echo "  TREE GROWTH DUMP COMPLETE"
echo "======================================================="
echo "Trees in:  $BUILD_DIR/Data/Viz/TreeGrowth/"
echo "Plot with: scripts/plot_tree_growth_iters.m"
echo "           cd $BUILD_DIR/Data/Viz/TreeGrowth"
echo "           matlab -batch \"addpath('$PROJECT_DIR/scripts'); plot_tree_growth_iters\""
echo "Config.h will be restored to original on exit."
