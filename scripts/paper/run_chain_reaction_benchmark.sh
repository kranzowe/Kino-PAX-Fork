#!/bin/bash
# =============================================================================
# Chain Reaction vs CountingStars -- pared-down paper benchmark.
#
# Builds and runs ChainReactionBenchmark (examples/gpu/paper/chain_reaction_benchmark.cu) on the
# 6D Dubins Airplane (Model 2), the "large"/coarse discretization (W7/C3/V3, 9,261 regions -- only
# CountingStars uses them; ChainReaction has no regions), path-length cost (COST_MODE 0), three
# environments, 5 runs per planner, 4s cap, MAX_TREE_SIZE 3,000,000, CountingStars ramp length 700 --
# the settings the main paper data (plots/DATA/ZEPHYR_30_runs) was recorded with, i.e.
# paper_benchmark_v2.cu, NOT v3's 1M-node / ramp-210 / 3s variant. The Model-2 config.h block is
# copied from run_paper_benchmark_v3.sh's write_config(); the original config.h is backed up and
# restored on exit.
#
# Like run_paper_benchmark_v3.sh, each (environment, planner) is its own process wrapped in an
# external `timeout`, so a CUDA hang is logged and skipped rather than freezing the sweep.
#
# Output (per-run CSVs the plots read via zephyr_common.load_runs):
#   build/Data/Benchmarks/ChainReaction/COARSE/<env>/<env>_<planner label>_deltam2_large_length_run<N>.csv
#
# --viz additionally builds TreeGrowthAnimationDump and dumps tree growth for CountingStars and
# ChainReaction on VIZ_ENV into build/Data/Viz/ChainReaction/ (render with
# plots/paper/chain_reaction_growth_animation.py).
#
# Usage:
#   bash scripts/paper/run_chain_reaction_benchmark.sh [--viz] [--skip-build] [--runs N] [--budget B]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"

# Model 2 (Dubins Airplane), "large" point, path-length cost. TAG matches what zephyr_common expects
# in the filename ("m2_large_length").
MODEL=2
DELTA_LABEL="large"
DELTA_FOLDER="COARSE"
W_R1=7
C_R1=3
V_R1=3
COST_LABEL="length"
COST_MODE_VAL=0
TAG="m${MODEL}_${DELTA_LABEL}_${COST_LABEL}"

ENV_NAMES=("house" "narrowPassage" "zigzag")
ENV_OBSTACLES=(
    "../include/config/obstacles/house/obstacles.csv"
    "../include/config/obstacles/narrowPassage/obstacles.csv"
    "../include/config/obstacles/zigzag/obstacles.csv"
)
PLANNER_FLAGS=("--only-chainreaction" "--only-countingstars")
PLANNER_NAMES=("ChainReaction"        "CountingStars")

NUM_RUNS=5
CR_BUDGET=5000          # ChainReaction's constant candidate budget B
RUN_TIMEOUT_S=180       # 5 runs x 4s cap + inter-run sleeps + startup, with margin
VIZ_TIMEOUT_S=1800     # 100 checkpoints x 2 planners, each copying a (up to 3M-node) tree to the host
KILL_AFTER_S=30
VIZ_ENV="zigzag"
VIZ_OBSTACLES="../include/config/obstacles/zigzag/obstacles.csv"
CS_FILL_ITERS=700       # CountingStars ramp length at 3M nodes (same as paper_benchmark_v2.cu / ZEPHYR_30_runs)

SKIP_BUILD=false
DO_VIZ=false
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) SKIP_BUILD=true ;;
        --viz)        DO_VIZ=true ;;
        --runs)       NUM_RUNS="$2"; shift ;;
        --budget)     CR_BUDGET="$2"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

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

# Model 2 (Dubins Airplane) config, verbatim from run_paper_benchmark_v3.sh's write_config() arm 2.
write_config() {
    local W_R1=$1
    local C_R1=$2
    local V_R1=$3
    local COST_MODE=$4
        cat > "$CONFIG_FILE" << CONFIGEOF
#pragma once
/***************************/
/* 6D DUBINS AIRPLANE      */
/***************************/
#define MODEL 2
#define COST_MODE ${COST_MODE}  // path cost: 1 = control effort ((yawRate^2+pitchRate^2+a^2)*dt), 0 = workspace distance
#define MAX_TREE_SIZE 3000000
#define MAX_FLOAT 1e38f
#define MAX_SOL_SET_SIZE 500
#define MAX_ITER 3000
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
#define C_DIM 2
#define V_DIM 1
#define W_MIN 0.0f
#define W_MAX 1.0f
#define W_SIZE 1.0f
#define C_MIN -M_PI
#define C_MAX M_PI
#define V_MIN 0.0f
#define V_MAX 0.3f
#define A_MIN -0.3f
#define A_MAX 0.3f
#define W_R1_LENGTH ${W_R1}
#define C_R1_LENGTH ${C_R1}
#define V_R1_LENGTH ${V_R1}
#define W_R2_LENGTH 2
#define C_R2_LENGTH 2
#define V_R2_LENGTH 2
#define W_R1_SIZE ((W_MAX - W_MIN) / W_R1_LENGTH)
#define C_R1_SIZE ((C_MAX - C_MIN) / C_R1_LENGTH)
#define V_R1_SIZE ((V_MAX - V_MIN) / V_R1_LENGTH)
#define W_R1_VOL (W_R1_SIZE * W_R1_SIZE * W_R1_SIZE)
#define NUM_R1_REGIONS (W_R1_LENGTH * W_R1_LENGTH * W_R1_LENGTH * C_R1_LENGTH * C_R1_LENGTH * V_R1_LENGTH)
#define NUM_R2_REGIONS (NUM_R1_REGIONS * W_R2_LENGTH * W_R2_LENGTH * W_R2_LENGTH * C_R2_LENGTH * C_R2_LENGTH * V_R2_LENGTH)
#define NUM_R2_PER_R1 W_R2_LENGTH *W_R2_LENGTH *W_R2_LENGTH *C_R2_LENGTH *C_R2_LENGTH *V_R2_LENGTH
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

REGIONS=$(( W_R1**3 * C_R1**2 * V_R1 ))

echo ""
echo "======================================================="
echo "  Chain Reaction benchmark"
echo "  Model: 2 (6D Dubins Airplane) | delta=${DELTA_LABEL} [${DELTA_FOLDER}] | Regions=${REGIONS} (CountingStars only)"
echo "  Cost: ${COST_LABEL} (COST_MODE=${COST_MODE_VAL}) | Runs/planner: ${NUM_RUNS} | ChainReaction B=${CR_BUDGET}"
echo "  Environments: ${ENV_NAMES[*]}"
echo "======================================================="

if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "=== BUILDING ==="
    write_config "$W_R1" "$C_R1" "$V_R1" "$COST_MODE_VAL"
    cd "$BUILD_DIR"
    # shellcheck disable=SC2086
    cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_COMPILER_FLAGS 2>&1 | tail -5
    make ChainReactionBenchmark -j"$(nproc)" 2>&1 | tail -20
    if [ "$DO_VIZ" = true ]; then
        make TreeGrowthAnimationDump -j"$(nproc)" 2>&1 | tail -20
    fi
    cd "$PROJECT_DIR"
fi

OUTPUT_ROOT="$BUILD_DIR/Data/Benchmarks/ChainReaction"
FAILURE_LOG="$OUTPUT_ROOT/failures.log"
mkdir -p "$OUTPUT_ROOT"
: > "$FAILURE_LOG"

cd "$BUILD_DIR"
for e in "${!ENV_NAMES[@]}"; do
    EN="${ENV_NAMES[$e]}"
    EO="${ENV_OBSTACLES[$e]}"
    for p in "${!PLANNER_NAMES[@]}"; do
        PN="${PLANNER_NAMES[$p]}"
        PF="${PLANNER_FLAGS[$p]}"

        echo ""
        echo "=== RUNNING (planner=${PN}, env=${EN}) ==="

        ec=0
        timeout --kill-after="${KILL_AFTER_S}s" "${RUN_TIMEOUT_S}s" \
            ./ChainReactionBenchmark "$TAG" "$EO" "$EN" "$DELTA_FOLDER" "$PF" --runs "$NUM_RUNS" --budget "$CR_BUDGET" || ec=$?

        if [ "$ec" -ne 0 ]; then
            if   [ "$ec" -eq 124 ]; then STATUS="TIMEOUT"
            elif [ "$ec" -ge 128 ]; then STATUS="CRASHED_SIG$((ec - 128))"
            else                          STATUS="ERROR"
            fi
            echo "  *** FAILURE: exit=${ec} status=${STATUS} -- logged to ${FAILURE_LOG} ***"
            printf '%s planner=%s env=%s exit=%s status=%s\n' \
                "$(date -Iseconds)" "$PN" "$EN" "$ec" "$STATUS" >> "$FAILURE_LOG"
        fi
    done
done

if [ "$DO_VIZ" = true ]; then
    echo ""
    echo "=== TREE GROWTH DUMP (env=${VIZ_ENV}) ==="
    ec=0
    timeout --kill-after="${KILL_AFTER_S}s" "${VIZ_TIMEOUT_S}s" \
        ./TreeGrowthAnimationDump "$VIZ_OBSTACLES" "$VIZ_ENV" --planners=CountingStars,ChainReaction \
        --out Data/Viz/ChainReaction --cs-fill-iters "$CS_FILL_ITERS" --cr-budget "$CR_BUDGET" || ec=$?
    if [ "$ec" -ne 0 ]; then
        echo "  *** tree growth dump failed: exit=${ec} ***"
        printf '%s tree-growth-dump env=%s exit=%s\n' "$(date -Iseconds)" "$VIZ_ENV" "$ec" >> "$FAILURE_LOG"
    fi
fi
cd "$PROJECT_DIR"

echo ""
echo "======================================================="
echo "  CHAIN REACTION BENCHMARK COMPLETE"
echo "======================================================="
echo "Per-run CSVs:  $OUTPUT_ROOT/${DELTA_FOLDER}/<env>/"
echo "  copy to:     plots/DATA/CHAIN_REACTION_runs/discretization${DELTA_FOLDER}/<env>/"
echo "  then:        python plots/paper/chain_reaction_compare.py"
if [ "$DO_VIZ" = true ]; then
    echo "Tree dumps:    $BUILD_DIR/Data/Viz/ChainReaction/"
    echo "  copy to:     plots/DATA/tree_animation_chain_reaction/TreeGrowthAnimation/"
    echo "  then:        python plots/paper/chain_reaction_growth_animation.py"
fi
if [ -s "$FAILURE_LOG" ]; then
    echo "Failures logged in: $FAILURE_LOG"
fi
echo "Config.h will be restored to original on exit."
