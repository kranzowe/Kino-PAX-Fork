#!/bin/bash
# =============================================================================
# Sample Trajectory Runner
#
# Builds the existing TreeCheckpointDump tool once, then runs it against EACH
# of House, NarrowPassage, and Zigzag in turn, and pulls CountingStars's own
# solved trajectory (whichever checkpoint file has one, preferring the latest/
# most-refined -- 4000ms down to 50ms) out into plots/DATA/environment_trajectories/
# as "<env>_trajectory.csv", ready for plots/view_environment.py to pick up and
# overlay automatically.
#
# TreeCheckpointDump already runs THREE planners (KPAX, KinoPaxPlus,
# CountingStars) at five wall-clock checkpoints and dumps each one's tree +
# trajectory -- reused as-is rather than writing a new single-planner CUDA path,
# since it already does exactly this. This script only cares about
# CountingStars's own output; KPAX/KinoPaxPlus's dumps for these runs are left
# in build/Data/Viz/TreeCheckpoints/ alongside it (harmless, just unused here).
#
# Model/discretization/cost-metric match scripts/run_tree_checkpoint_dump.sh's
# own zigzag-only setup exactly (Model 1, W7/C1/V3 "large"/coarse, COST_MODE 2 =
# path time), so all three environments' sample trajectories are produced the
# same way and are comparable. The config.h rewrite is NOT optional -- see that
# script's own header for why; original config.h is backed up and restored on
# exit here too.
#
# Usage:
#   cd scripts && bash run_sample_trajectories.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"
DUMP_DIR="$BUILD_DIR/Data/Viz/TreeCheckpoints"
OUT_DIR="$PROJECT_DIR/plots/DATA/environment_trajectories"

# Same canonical Model-1 "large"/coarse point as run_tree_checkpoint_dump.sh.
DELTA_W_R1=7
DELTA_C_R1=1
DELTA_V_R1=3
COST_MODE_VAL=2   # 2 = path time, 1 = control effort, 0 = workspace distance

ENV_NAMES=(house narrowPassage zigzag)
# Checkpoints TreeCheckpointDump dumps, largest (most-refined solution) first --
# used below to pick the best available solved trajectory per environment.
CHECKPOINTS_DESC=(4000 1000 150 100 50)

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
mkdir -p "$BUILD_DIR" "$OUT_DIR"

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
echo "  Sample Trajectory Runner"
echo "  Model: 1 (6D Double Integrator)"
echo "  Environments: ${ENV_NAMES[*]}"
echo "  Delta: W_R1=${DELTA_W_R1} C_R1=${DELTA_C_R1} V_R1=${DELTA_V_R1} | Regions=${REGIONS}"
echo "  Cost metric: path time (COST_MODE=${COST_MODE_VAL})"
echo "  Trajectory taken from: CountingStars"
echo "  Output: ${OUT_DIR}"
echo "======================================================="

echo ""
echo "=== BUILDING (delta=large, COST_MODE=${COST_MODE_VAL}, Regions=${REGIONS}) ==="
write_config "$DELTA_W_R1" "$DELTA_C_R1" "$DELTA_V_R1" "$COST_MODE_VAL"
cd "$BUILD_DIR"
# shellcheck disable=SC2086
cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_COMPILER_FLAGS 2>&1 | tail -5
make TreeCheckpointDump -j"$(nproc)" 2>&1 | tail -20
cd "$PROJECT_DIR"

for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_OBSTACLES="$PROJECT_DIR/include/config/obstacles/${ENV_NAME}/obstacles.csv"
    if [ ! -f "$ENV_OBSTACLES" ]; then
        echo ""
        echo "[warn] no obstacle file for '${ENV_NAME}' at ${ENV_OBSTACLES} -- skipping."
        continue
    fi

    echo ""
    echo "=== RUNNING (Env=${ENV_NAME}) ==="
    cd "$BUILD_DIR"
    ./TreeCheckpointDump "$ENV_OBSTACLES" "$ENV_NAME"
    cd "$PROJECT_DIR"

    FOUND=0
    for MS in "${CHECKPOINTS_DESC[@]}"; do
        TRAJ_FILE="$DUMP_DIR/${ENV_NAME}_CountingStars_t${MS}ms_traj.csv"
        if [ -f "$TRAJ_FILE" ] && [ "$(wc -l < "$TRAJ_FILE")" -gt 1 ]; then
            DEST="$OUT_DIR/${ENV_NAME}_trajectory.csv"
            cp "$TRAJ_FILE" "$DEST"
            echo "  CountingStars solved by t=${MS}ms -- copied to ${DEST}"
            FOUND=1
            break
        fi
    done
    if [ "$FOUND" -eq 0 ]; then
        echo "  [warn] CountingStars never solved '${ENV_NAME}' by the last checkpoint (4000ms) --"
        echo "         no trajectory CSV written for this environment. Consider re-running with a"
        echo "         longer last checkpoint if this keeps happening."
    fi
done

echo ""
echo "======================================================="
echo "  SAMPLE TRAJECTORIES COMPLETE"
echo "======================================================="
echo "Trajectory CSVs in: ${OUT_DIR}"
echo "Plot with: python plots/view_environment.py"
echo "Config.h will be restored to original on exit."
