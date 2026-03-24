#!/bin/bash
# =============================================================================
# KinoPaxPlus Delta Benchmark Runner
#
# Iterates over 4 delta (region discretization) configs for Model 1
# (6D double integrator). For each config:
#   1. Writes a complete Model 1 config.h with the appropriate W_R1_LENGTH
#   2. Rebuilds the KinoPaxPlusDeltaBenchmark target
#   3. Runs the benchmark with the delta label
#
# Original config.h is backed up and restored on exit/error.
#
# Delta configs (C_R1=1, V_R1=3 throughout):
#   large:     W_R1=10 -> 10^3 * 27 =      27,000 regions (paper large-delta)
#   med_large: W_R1=20 -> 20^3 * 27 =     216,000 regions
#   med_small: W_R1=40 -> 40^3 * 27 =   1,728,000 regions (previous config)
#   small:     W_R1=72 -> 72^3 * 27 =  10,077,696 regions (paper small-delta)
#
# Usage: cd scripts && bash run_delta_benchmark.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"
OBSTACLE_SRC="$PROJECT_DIR/include/config/obstacles/quadTrees/obstacles.csv"
OBSTACLE_SCALED="/tmp/kpax_trees_scaled.csv"

# Delta configs: label W_R1_LENGTH
DELTA_LABELS=("large" "med_large" "med_small" "small")
DELTA_W_R1=(10 20 40 72)

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

# --- Scale obstacles from [0,100]^3 to [0,1]^3 ---
echo "Scaling obstacles to [0,1]^3 workspace..."
awk -F, '{for(i=1;i<=NF;i++) printf "%.6f%s", $i/100, (i<NF?",":"\n")}' \
    "$OBSTACLE_SRC" > "$OBSTACLE_SCALED"
echo "Scaled obstacles written to $OBSTACLE_SCALED"

# --- Ensure build directory exists ---
mkdir -p "$BUILD_DIR"

# Function to write complete Model 1 config.h
write_config() {
    local W_R1=$1
    cat > "$CONFIG_FILE" << CONFIGEOF
#pragma once
/***************************/
/* 6D DOUBLE INTEGRATOR CONFIG  */
/***************************/
#define MODEL 1
#define MAX_TREE_SIZE 1000000
#define MAX_FLOAT 1000000.0f
#define MAX_SOL_SET_SIZE 500
#define MAX_ITER 10
#define STEP_SIZE 0.1f
#define MAX_PROPAGATION_DURATION 10
#define ACCEPT 0.99f
#define AGENT_RADIUS 0.005f
#define GOAL_THRESH 0.05f
#define STATE_DIM 6
#define CONTROL_DIM 3
#define SAMPLE_DIM (STATE_DIM + CONTROL_DIM + 1)
#define W_DIM 3
#define C_DIM 1
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
#define C_R1_LENGTH 1
#define V_R1_LENGTH 3
#define W_R2_LENGTH 1
#define C_R2_LENGTH 1
#define V_R2_LENGTH 1
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
#define QUAD_MIN_Zc -2.0f
#define QUAD_MAX_Zc 2.0f
#define QUAD_MIN_Lc -2.0f
#define QUAD_MAX_Lc 2.0f
#define QUAD_MIN_Mc -2.0f
#define QUAD_MAX_Mc 2.0f
#define QUAD_MIN_Nc -2.0f
#define QUAD_MAX_Nc 2.0f
#define QUAD_MIN_YAW -M_PI
#define QUAD_MAX_YAW M_PI
#define QUAD_MIN_PITCH -M_PI / 3
#define QUAD_MAX_PITCH M_PI / 3
#define QUAD_MIN_ROLL -M_PI / 3
#define QUAD_MAX_ROLL M_PI / 3
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

# =============================================================================
# Run benchmarks
# =============================================================================
echo ""
echo "======================================================="
echo "  KinoPaxPlus Delta Benchmark Sweep"
echo "  Model: 1 (6D Double Integrator)"
echo "  Environment: Trees (scaled to [0,1]^3)"
echo "  MAX_TREE_SIZE: 1,000,000"
echo "  Deltas: ${DELTA_LABELS[*]}"
echo "======================================================="

for i in "${!DELTA_LABELS[@]}"; do
    LABEL="${DELTA_LABELS[$i]}"
    W_R1="${DELTA_W_R1[$i]}"
    REGIONS=$(( W_R1 * W_R1 * W_R1 * 27 ))

    echo ""
    echo "======================================================="
    echo "  Delta: $LABEL | W_R1=$W_R1 | Regions=$REGIONS"
    echo "======================================================="

    # Write config
    write_config "$W_R1"
    echo "  config.h written (W_R1_LENGTH=$W_R1)"

    # Build
    echo "  Building..."
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    make KinoPaxPlusDeltaBenchmark -j"$(nproc)" 2>&1 | tail -3
    echo "  Build complete"

    # Run
    echo "  Running benchmark..."
    ./KinoPaxPlusDeltaBenchmark "$LABEL" "$OBSTACLE_SCALED"

    cd "$PROJECT_DIR"
done

echo ""
echo "======================================================="
echo "  ALL DELTA BENCHMARKS COMPLETE"
echo "======================================================="
echo "Results in: $BUILD_DIR/Data/Benchmarks/KinoPaxPlusDelta/"
echo "Config.h will be restored to original on exit."
