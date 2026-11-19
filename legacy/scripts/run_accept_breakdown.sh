#!/bin/bash
# =============================================================================
# Acceptance-Reason Breakdown Runner  (KinoPaxSTARCOMBO + a KinoPaxSTARCleanCost reference)
#
# Diagnostic, not a benchmark. Answers "WHY did this tuning admit the nodes it admitted", which the
# tuning sweep cannot: a candidate enters the frontier through one of three doors --
#
#   1. the region-best exemption   (cost <= minCostsR1[r])                    both planners
#   2. the R2 seeding exemption    CleanCost only, off by default; REMOVED in COMBO
#   3. the roll                    CleanCost: rand < cap * (w*pSyclop + (1-w)*pCost + floor)
#                                  COMBO:     rand < min(pMax, comboShape * pTargetAccept)
#
# It also logs the growth-controller and frontier-composition columns, because for COMBO the
# acceptance split is only half the rule. n_active and reactivated are the two to read when
# propagate falls onto the slow kernel2 path early: Part B re-activates the region best
# UNCONDITIONALLY, one per explored region, so the frontier has a hard floor at nActive and kernel2
# is forced once 32*F > remaining regardless of the acceptance tuning.
#
# -- and nothing in the normal output distinguishes them. Door 3 is a single Bernoulli draw against
# a weighted SUM, so each accepted node splits one unit of credit in proportion to each term's
# share; see the header of examples/gpu/kinopaxstar_accept_breakdown.cu.
#
# ONE run per grid point, ONE environment, ONE delta, ONE cost metric -- 21 runs, ~2 minutes, one
# build. Reports against ITERATION, never wall-clock: the counting atomics distort timing.
#
#   grid   COMBO: N {2, 3, 4, 5} sigma x kFan {0, 16, 32, 64}
#          = 13 points (not 16: at kFan = 0 every fan-out score is identical, so sigma is 0 and
#          all four N run the same uniform control arm -- only the derived N = 2 is kept there)
#   env    house
#   delta  large  (W_R1=10 C_R1=1 V_R1=3 -> 27,000 R1 regions)
#   metric workspace path length (COST_MODE 0)
#
# The env / delta / metric lists are arrays so more can be added by editing one line each.
# Original config.h is backed up and restored on exit/error.
#
# Usage:
#   cd scripts && bash run_accept_breakdown.sh
#   cd scripts && bash run_accept_breakdown.sh --skip-build
#
# Plot with: scripts/plot_accept_breakdown.m  (cd into the env folder first)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"

DELTA_LABEL="large"
DELTA_W_R1=10
DELTA_C_R1=1
DELTA_V_R1=3

COST_LABEL="length"
COST_MODE_VAL=0

ENV_NAMES=("house")
ENV_OBSTACLES=("../include/config/obstacles/house/obstacles.csv")

SKIP_BUILD=false
for arg in "$@"; do
    if [ "$arg" = "--skip-build" ]; then SKIP_BUILD=true; fi
done

# --- Auto-detect compilers (cluster has gcc-12, Jetson uses default gcc) ---
CMAKE_COMPILER_FLAGS=""
if command -v gcc-12 &> /dev/null; then
    echo "Detected gcc-12 (cluster environment)"
    CMAKE_COMPILER_FLAGS="-DCMAKE_C_COMPILER=$(which gcc-12) -DCMAKE_CXX_COMPILER=$(which g++-12) -DCMAKE_CUDA_HOST_COMPILER=$(which g++-12)"
else
    echo "Using default system compilers (Jetson/local environment)"
fi

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

echo "Backing up config.h..."
cp "$CONFIG_FILE" "$CONFIG_BACKUP"
mkdir -p "$BUILD_DIR"

# Same config heredoc every runner in this repo carries -- kept local so the script is
# self-contained and can back up / restore independently.
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
REGIONS=$(( DELTA_W_R1**3 * DELTA_V_R1**3 ))

echo ""
echo "======================================================="
echo "  Acceptance-Reason Breakdown (COMBO grid + CleanCost reference)"
echo "  Delta: ${DELTA_LABEL} | W_R1=${DELTA_W_R1} C_R1=${DELTA_C_R1} V_R1=${DELTA_V_R1} | Regions=${REGIONS}"
echo "  Cost metric: ${COST_LABEL} (COST_MODE=${COST_MODE_VAL})"
echo "  Environments: ${ENV_NAMES[*]}"
echo "  COMBO: N {2,3,4,5} sigma x kFan {0,16,32,64} x rf 0.1 = 13 points"
echo "  Reference: CleanCost w 0.9, k 1, cap 0.03 (r2 off) = 1 point"
echo "  1 run each, reported against ITERATION (counting atomics distort wall-clock)"
echo "======================================================="

# =============================================================================
# BUILD
# =============================================================================
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "=== BUILDING (delta=${DELTA_LABEL}, cost=${COST_LABEL}, Regions=${REGIONS}) ==="
    write_config "$DELTA_W_R1" "$DELTA_C_R1" "$DELTA_V_R1" "$COST_MODE_VAL"
    cd "$BUILD_DIR"
    # shellcheck disable=SC2086
    cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_COMPILER_FLAGS 2>&1 | tail -5
    make KinoPaxStarAcceptBreakdown -j"$(nproc)" 2>&1 | tail -20
    cp KinoPaxStarAcceptBreakdown "KinoPaxStarAcceptBreakdown_${DELTA_LABEL}_${COST_LABEL}"
    cd "$PROJECT_DIR"
else
    echo ""
    echo "=== SKIPPING BUILD (using cached binary) ==="
    if [ ! -f "$BUILD_DIR/KinoPaxStarAcceptBreakdown_${DELTA_LABEL}_${COST_LABEL}" ]; then
        echo "ERROR: cached binary not found. Run without --skip-build first."
        exit 1
    fi
fi

# =============================================================================
# RUN
# =============================================================================
cd "$BUILD_DIR"
for i in "${!ENV_NAMES[@]}"; do
    EN="${ENV_NAMES[$i]}"
    EO="${ENV_OBSTACLES[$i]}"
    echo ""
    echo "=== RUNNING (Env=${EN}) ==="
    # argv[1] carries delta+metric so it lands in every filename; argv[3] picks the subfolder.
    "./KinoPaxStarAcceptBreakdown_${DELTA_LABEL}_${COST_LABEL}"         "${DELTA_LABEL}_${COST_LABEL}" "$EO" "$EN"
done
cd "$PROJECT_DIR"

echo ""
echo "======================================================="
echo "  ACCEPTANCE BREAKDOWN COMPLETE"
echo "======================================================="
for EN in "${ENV_NAMES[@]}"; do
    echo "Results in: $BUILD_DIR/Data/Benchmarks/KinoPaxStarAcceptBreakdown/${EN}/"
done
echo "Plot: cd into that folder, addpath('<repo>/scripts'), then run plot_accept_breakdown"
echo "Config.h will be restored to original on exit."
