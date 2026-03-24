#!/bin/bash
# =============================================================================
# KinoPaxPlus Delta Benchmark Runner
#
# Iterates over 4 delta (region discretization) configs for Model 3
# (12D nonlinear quadrotor). For each config:
#   1. Writes a complete Model 3 config.h with the appropriate R1 lengths
#   2. Rebuilds the KinoPaxPlusDeltaBenchmark target
#   3. Runs the benchmark with the delta label
#
# Original config.h is backed up and restored on exit/error.
#
# Delta configs (Model 3: W_DIM=3, C_DIM=3, V_DIM=3):
#   NUM_R1_REGIONS = W_R1^3 * C_R1^3 * V_R1^3
#
#   large:     W=8,  C=3, V=2 ->  512 *  27 *   8 =     110,592 regions
#   med_large: W=10, C=4, V=3 -> 1000 *  64 *  27 =   1,728,000 regions
#   med_small: W=12, C=4, V=4 -> 1728 *  64 *  64 =   7,077,888 regions
#   small:     W=14, C=4, V=4 -> 2744 *  64 *  64 =  11,239,424 regions
#
# Usage: cd scripts && bash run_delta_benchmark.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"

# Delta configs: label W_R1 C_R1 V_R1
DELTA_LABELS=("large" "med_large" "med_small" "small")
DELTA_W_R1=(8 10 12 14)
DELTA_C_R1=(3  4  4  4)
DELTA_V_R1=(2  3  4  4)

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

# Function to write complete Model 3 config.h
write_config() {
    local W_R1=$1
    local C_R1=$2
    local V_R1=$3
    cat > "$CONFIG_FILE" << CONFIGEOF
#pragma once
/***************************/
/* NON LINEAR QUAD CONFIG  */
/***************************/
#define MODEL 3
#define MAX_TREE_SIZE 1000000
#define MAX_FLOAT 1e38f
#define MAX_SOL_SET_SIZE 500
#define MAX_ITER 300
#define MAX_ITER_REKINO 20000
#define STEP_SIZE 0.1f
#define MAX_PROPAGATION_DURATION 10
#define ACCEPT 0.99f
#define AGENT_RADIUS 0.5f
#define GOAL_THRESH 5.0f
#define STATE_DIM 12
#define CONTROL_DIM 4
#define SAMPLE_DIM (STATE_DIM + CONTROL_DIM + 1)
#define W_DIM 3
#define C_DIM 3
#define V_DIM 3
#define W_MIN 0.0f
#define W_MAX 100.0f
#define W_SIZE 100.0f
#define C_MIN -M_PI
#define C_MAX M_PI
#define V_MIN -30.0f
#define V_MAX 30.0f
#define A_MIN -30.0f
#define A_MAX 30.0f
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
#define NUM_R1_REGIONS (W_R1_LENGTH * W_R1_LENGTH * W_R1_LENGTH * C_R1_LENGTH * C_R1_LENGTH * C_R1_LENGTH * V_R1_LENGTH * V_R1_LENGTH * V_R1_LENGTH)
#define NUM_R2_REGIONS (NUM_R1_REGIONS * W_R2_LENGTH * W_R2_LENGTH * W_R2_LENGTH * C_R2_LENGTH * C_R2_LENGTH * C_R2_LENGTH * V_R2_LENGTH * V_R2_LENGTH * V_R2_LENGTH)
#define NUM_R2_PER_R1 W_R2_LENGTH *W_R2_LENGTH *W_R2_LENGTH *C_R2_LENGTH *C_R2_LENGTH *C_R2_LENGTH *V_R2_LENGTH *V_R2_LENGTH *V_R2_LENGTH
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

# =============================================================================
# Run benchmarks
# =============================================================================
echo ""
echo "======================================================="
echo "  KinoPaxPlus Delta Benchmark Sweep"
echo "  Model: 3 (12D Nonlinear Quadrotor)"
echo "  Environment: Trees ([0,100]^3)"
echo "  MAX_TREE_SIZE: 100,000"
echo "  Deltas: ${DELTA_LABELS[*]}"
echo "======================================================="

for i in "${!DELTA_LABELS[@]}"; do
    LABEL="${DELTA_LABELS[$i]}"
    W_R1="${DELTA_W_R1[$i]}"
    C_R1="${DELTA_C_R1[$i]}"
    V_R1="${DELTA_V_R1[$i]}"
    REGIONS=$(( W_R1 * W_R1 * W_R1 * C_R1 * C_R1 * C_R1 * V_R1 * V_R1 * V_R1 ))

    echo ""
    echo "======================================================="
    echo "  Delta: $LABEL | W_R1=$W_R1 C_R1=$C_R1 V_R1=$V_R1 | Regions=$REGIONS"
    echo "======================================================="

    # Write config
    write_config "$W_R1" "$C_R1" "$V_R1"
    echo "  config.h written (W_R1=$W_R1, C_R1=$C_R1, V_R1=$V_R1)"

    # Build
    echo "  Building..."
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
        > /dev/null 2>&1
    make KinoPaxPlusDeltaBenchmark -j"$(nproc)" 2>&1 | tail -20
    echo "  Build complete"

    # Run
    echo "  Running benchmark..."
    ./KinoPaxPlusDeltaBenchmark "$LABEL"

    cd "$PROJECT_DIR"
done

echo ""
echo "======================================================="
echo "  ALL DELTA BENCHMARKS COMPLETE"
echo "======================================================="
echo "Results in: $BUILD_DIR/Data/Benchmarks/KinoPaxPlusDelta/"
echo "Config.h will be restored to original on exit."
