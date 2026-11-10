#!/bin/bash
# =============================================================================
# Tree Growth Animation Dump Runner
#
# Builds and runs TreeGrowthAnimationDump -- the same tool as
# run_tree_checkpoint_dump.sh/tree_checkpoint_dump.cu (same three planners,
# same fixed tunables, same discretization, same cost metric, same tree
# sparsification), just with a much finer checkpoint schedule: every 30ms out
# to 3000ms (100 checkpoints) instead of 6 hand-picked points, so
# plots/tree_growth_animation.py can render an actual animation of the tree
# filling in over time.
#
#   KPAX              pure explorer reference
#   KinoPaxPlus        pure optimizer reference
#   CountingStars      bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15,
#                      cost_frac 0.75, hopelessGuard ON (v3.5) -- the exact
#                      canonical point run_paper_benchmark_v2.sh/
#                      paper_benchmark_v2.cu use for this planner
#
# 3 planners x 100 checkpoints x 2 files (tree + trajectory) = 600 CSVs +
# meta.csv + a checkpoint_stats.csv (one row per (planner, checkpoint)
# actually reached, with the REAL, non-sparsified tree_size and its % of
# MAX_TREE_SIZE), rendered by plots/tree_growth_animation.py. Tree CSVs keep
# only a small fraction of each tree's leaf-rooted trajectories (1% here vs.
# run_tree_checkpoint_dump.sh's 4% -- lower because this tool renders 100
# checkpoints' worth of frames instead of 6) -- full trees can
# reach millions of nodes, so this keeps the CSVs and the render tractable
# while still showing complete, connected paths.
#
# A (planner, checkpoint) pair can be missing entirely if that planner's tree
# fills MAX_TREE_SIZE (or otherwise runs out of room to propagate) before that
# checkpoint's wall-clock threshold is reached -- see
# tree_growth_animation_dump.cu's own header comment for the two exact
# conditions and their console messages.
#
# The config.h rewrite is NOT optional: the checked-in config.h is a MODEL 3
# (12D quad, W_MAX 100) config, while the zigzag obstacles and every plotting
# script assume MODEL 1 in [0,1]^3. Original config.h is backed up and restored
# on exit.
#
# Discretization, cost metric, environment, model: identical to
# run_tree_checkpoint_dump.sh (same DELTA_LABEL/COST_MODE_VAL/ENV_NAME below)
# -- this script deliberately does not change the algorithms, parameters, or
# how much of the tree gets saved, only how many checkpoints get dumped.
#
# Usage:
#   cd scripts && bash run_tree_growth_animation_dump.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"

# Single delta: label W_R1 C_R1 V_R1  (V2's canonical "large"/coarse point -- same as
# run_tree_checkpoint_dump.sh)
DELTA_LABEL="large"
DELTA_W_R1=7
DELTA_C_R1=1
DELTA_V_R1=3

# Cost metric for this build: 2 = path time, 1 = control effort, 0 = workspace path length
# (same as run_tree_checkpoint_dump.sh)
COST_MODE_VAL=0

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

# Function to write complete Model 1 config.h (identical to run_tree_checkpoint_dump.sh's)
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
#define COST_MODE ${COST_MODE}  // path cost: 2 = path time (sum of edge dt), 1 = control effort ((ax^2+ay^2+az^2)*dt), 0 = workspace distance
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
echo "  Tree Growth Animation Dump"
echo "  Model: 1 (6D Double Integrator)"
echo "  Environment: ${ENV_NAME}"
echo "  Delta: ${DELTA_LABEL} | W_R1=${DELTA_W_R1} C_R1=${DELTA_C_R1} V_R1=${DELTA_V_R1} | Regions=${REGIONS}"
echo "  Cost metric: COST_MODE=${COST_MODE_VAL}"
echo "  Planners: KPAX, KinoPaxPlus, CountingStars"
echo "  Checkpoints: every 30ms out to 3000ms (100 checkpoints)"
echo "  Tree CSVs: 1% of leaf trajectories per checkpoint"
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
make TreeGrowthAnimationDump -j"$(nproc)" 2>&1 | tail -20
cd "$PROJECT_DIR"

# =============================================================================
# RUN
# =============================================================================
echo ""
echo "=== RUNNING (Env=${ENV_NAME}) ==="

cd "$BUILD_DIR"
./TreeGrowthAnimationDump "$ENV_OBSTACLES" "$ENV_NAME"
cd "$PROJECT_DIR"

echo ""
echo "======================================================="
echo "  TREE GROWTH ANIMATION DUMP COMPLETE"
echo "======================================================="
echo "Trees/trajectories in:  $BUILD_DIR/Data/Viz/TreeGrowthAnimation/"
echo "Plot with: python plots/tree_growth_animation.py"
echo "Config.h will be restored to original on exit."
