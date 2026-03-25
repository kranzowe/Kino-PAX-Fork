#!/bin/bash
# =============================================================================
# KinoPaxPlus Delta Benchmark Runner
#
# Iterates over 4 delta (region discretization) configs for Model 1
# (6D double integrator). For each config:
#   1. Writes a complete Model 1 config.h with the appropriate W_R1 length
#   2. Rebuilds the KinoPaxPlusDeltaBenchmark target
#   3. Runs the benchmark with the delta label
#
# Also runs a KPAX baseline (20 runs, once per environment on the first delta)
# for comparison against KinoPaxPlus convergence.
#
# Runs on two environments: Trees and House.
# Original config.h is backed up and restored on exit/error.
#
# Delta configs (Model 1: W_DIM=3, C_DIM=1, V_DIM=3):
#   NUM_R1_REGIONS = W_R1^3 * V_R1^3  (C_R1=1 throughout)
#
#   large:     W=10, V=3 ->  1000 * 27 =      27,000 regions
#   med_large: W=20, V=3 ->  8000 * 27 =     216,000 regions
#   med_small: W=40, V=3 -> 64000 * 27 =   1,728,000 regions
#   small:     W=72, V=3 -> 373248 * 27 = 10,077,696 regions
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
DELTA_W_R1=(10 20 40 72)
DELTA_C_R1=(1  1  1  1)
DELTA_V_R1=(3  3  3  3)

# Environments and their obstacle files (already in [0,1]^3 for Model 1)
ENV_NAMES=("trees" "house")
ENV_OBSTACLES=(
    "../include/config/obstacles/trees/obstacles.csv"
    "../include/config/obstacles/house/obstacles.csv"
)

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
    cat > "$CONFIG_FILE" << CONFIGEOF
#pragma once
/***************************/
/* 6D DOUBLE INTEGRATOR    */
/***************************/
#define MODEL 1
#define MAX_TREE_SIZE 2000000
#define MAX_FLOAT 1e38f
#define MAX_SOL_SET_SIZE 500
#define MAX_ITER 300
#define MAX_ITER_REKINO 20000
#define STEP_SIZE 0.1f
#define MAX_PROPAGATION_DURATION 10
#define ACCEPT 0.99f
#define AGENT_RADIUS 0.5f
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

# =============================================================================
# Run benchmarks
# =============================================================================
echo ""
echo "======================================================="
echo "  KinoPaxPlus Delta Benchmark Sweep"
echo "  Model: 1 (6D Double Integrator)"
echo "  Environments: ${ENV_NAMES[*]}"
echo "  MAX_TREE_SIZE: 2,000,000"
echo "  Deltas: ${DELTA_LABELS[*]}"
echo "  KPAX Baseline: 20 runs per environment"
echo "======================================================="

for env_idx in "${!ENV_NAMES[@]}"; do
    ENV="${ENV_NAMES[$env_idx]}"
    OBS="${ENV_OBSTACLES[$env_idx]}"
    FIRST_DELTA_FOR_ENV=true

    echo ""
    echo "======================================================="
    echo "  ENVIRONMENT: $ENV"
    echo "  Obstacles: $OBS"
    echo "======================================================="

    for i in "${!DELTA_LABELS[@]}"; do
        LABEL="${DELTA_LABELS[$i]}"
        W_R1="${DELTA_W_R1[$i]}"
        C_R1="${DELTA_C_R1[$i]}"
        V_R1="${DELTA_V_R1[$i]}"
        REGIONS=$(( W_R1**3 * V_R1**3 ))

        echo ""
        echo "-------------------------------------------------------"
        echo "  Delta: $LABEL | W_R1=$W_R1 C_R1=$C_R1 V_R1=$V_R1 | Regions=$REGIONS"
        echo "-------------------------------------------------------"

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

        # Run — pass --run-kpax only on first delta for this environment
        echo "  Running benchmark..."
        if [ "$FIRST_DELTA_FOR_ENV" = true ]; then
            ./KinoPaxPlusDeltaBenchmark "$LABEL" "$OBS" "$ENV" --run-kpax
            FIRST_DELTA_FOR_ENV=false
        else
            ./KinoPaxPlusDeltaBenchmark "$LABEL" "$OBS" "$ENV"
        fi

        cd "$PROJECT_DIR"
    done
done

echo ""
echo "======================================================="
echo "  ALL DELTA BENCHMARKS COMPLETE"
echo "======================================================="
echo "Results in: $BUILD_DIR/Data/Benchmarks/KinoPaxPlusDelta/"
echo "Config.h will be restored to original on exit."
