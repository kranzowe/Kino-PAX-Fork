#!/bin/bash
# =============================================================================
# Paper Benchmark Runner
#
# A FIXED, FIVE-WAY COMPARISON, not a sweep. Every series below is an already-chosen operating
# point; the only things varying across runs are discretization, environment, and cost metric.
# Modeled on run_countingstars_sweep.sh's two-phase build-then-run structure, but simpler: there is
# no per-delta arm partition (--only-kinopaxplus has no equivalent here) because every series runs
# at every delta -- that is the comparison this suite exists to make.
#
# THE FIVE SERIES (fixed inside examples/gpu/paper_benchmark.cu, not swept here):
#   KPAX                     defaults
#   KinoPaxPlus               defaults
#   KinoPaxSTARCleanCost      r2 off, w 0.9, k 1.0, cap 0.03
#   CountingStars (slope 1.4) explore_frac 0.3, cost_frac 0.3, bufferFloor 0.05
#   CountingStars (slope 1.8) same, bufferSlope 1.8
#
# THREE DELTAS, ALL FIVE SERIES AT EACH:
#   large  W_R1=10  C_R1=1  V_R1=3  ->  10^3 * 3^3 =  27,000 regions
#   fine   W_R1=20  C_R1=1  V_R1=3  ->  20^3 * 3^3 = 216,000 regions
#   tiny   W_R1=15  C_R1=1  V_R1=4  ->  15^3 * 4^3 = 216,000 regions
# "fine" and "tiny" are a CONTROLLED PAIR at the identical region count, refined on different axes
# (workspace vs. velocity) -- same convention as countingstars_sweep.cu's fine/fine_control pair.
# C_R1 stays at 1 everywhere: this config sets C_DIM 0, so control refinement has nowhere to act
# except V_R1.
#
# FOUR ENVIRONMENTS: empty, house, narrowPassage, zigzag. zigzag's five doorway gaps were tightened
# from 0.10 to 0.02 wide (include/config/obstacles/zigzag/obstacles.csv) to match narrowPassage's
# clearance exactly -- expect both to show materially lower success rates than empty/house.
#
# TWO COST METRICS (length, effort), each its own full build.
#
# MAX_TREE_SIZE (3,000,000) and the per-run wall-clock cap (10s, compiled into
# examples/gpu/paper_benchmark.cu as MAX_TIME_MS) are meant to be the actual stop conditions. The
# harness's own outer iteration loop is capped at 20,000 -- high enough that it should not bind.
#
# MAX_ITER IN config.h IS DELIBERATELY LEFT AT 1000, THE SAME VALUE EVERY OTHER SWEEP SCRIPT IN
# THIS REPO USES -- NOT bumped to 20,000 to "match" the harness's own loop cap above. CountingStars'
# buffer ramp reads h_fillIters_ (defaults to MAX_ITER) to compute x = itr/fill_iters, clamped to
# [0,1]; MAX_ITER sits in that denominator, so raising it would silently shrink B at every point on
# the ramp rather than just letting iterations run longer. Leaving it at 1000 means any run that
# outlives 1000 iterations (nearly all of them, since MAX_TREE_SIZE/MAX_TIME_MS are the intended
# limiters) has x pinned at 1 and B plateaued at its ramp maximum for the rest of the run --
# already-supported, intended behavior, not a new edge case.
#
# SCALE: 5 series x 3 deltas x 4 environments x 2 cost metrics x 10 runs = 1,200 runs, each capped
# at 10s. Worst case a few hours; most runs stop earlier (tree-full or an early success).
#
# NUM_R1_REGIONS and COST_MODE are both COMPILE-TIME, so neither can vary within one binary. Same
# build-cache pattern as run_countingstars_sweep.sh: write config.h and build once per (delta, cost
# metric) = 6 binaries, cached under a suffixed name, then run each once per environment.
#
# Original config.h is backed up and restored on exit/error.
#
# Usage:
#   cd scripts && bash run_paper_benchmark.sh
#   cd scripts && bash run_paper_benchmark.sh --skip-build   # run only (cached binaries)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"

# Deltas: parallel arrays of label / W_R1 / C_R1 / V_R1. Every series in paper_benchmark.cu runs at
# every delta -- there is no arm partition to configure here.
DELTA_LABELS=("large" "fine" "tiny")
DELTA_W_R1S=(10 20 15)
DELTA_C_R1S=(1  1  1)   # inert for Model 1 (C_DIM 0); control refinement rides on V_R1
DELTA_V_R1S=(3  3  4)

# Cost metric axis: label + COST_MODE (0 = workspace distance, 1 = control effort). Both this pass
# -- one full build of every delta for each.
COST_LABELS=("length" "effort")
COST_MODES=(0 1)

# Environments (obstacles already in [0,1]^3 for Model 1). Each gets its own output subfolder.
ENV_NAMES=("empty" "house" "narrowPassage" "zigzag")
ENV_OBSTACLES=(
    "../include/config/obstacles/empty/obstacles.csv"
    "../include/config/obstacles/house/obstacles.csv"
    "../include/config/obstacles/narrowPassage/obstacles.csv"
    "../include/config/obstacles/zigzag/obstacles.csv"
)

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

# Function to write complete Model 1 config.h. Identical to run_countingstars_sweep.cu's --
# MAX_ITER stays at 1000 regardless of delta/metric; see the header comment above for why.
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
#define MAX_ITER 1000
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
// checked-in config.h selects. KinoPaxPlus is a headline series here (at all three
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
echo "  Paper Benchmark"
echo "  Model: 1 (6D Double Integrator)"
echo "  Environments: ${ENV_NAMES[*]}  (separate output subfolders)"
for i in "${!DELTA_LABELS[@]}"; do
    R=$(( DELTA_W_R1S[i]**3 * DELTA_V_R1S[i]**3 ))
    echo "  Delta: ${DELTA_LABELS[$i]} | W_R1=${DELTA_W_R1S[$i]} C_R1=${DELTA_C_R1S[$i]} V_R1=${DELTA_V_R1S[$i]} | Regions=${R} | all 5 series"
done
echo "  Cost metrics: ${COST_LABELS[*]}  (one build each)"
echo "  Series (fixed, all 3 deltas x all 4 environments):"
echo "    KPAX                      defaults"
echo "    KinoPaxPlus                defaults"
echo "    KinoPaxSTARCleanCost       r2 off, w 0.9, k 1.0, cap 0.03"
echo "    CountingStars (slope 1.4)  explore_frac 0.3, cost_frac 0.3, bufferFloor 0.05"
echo "    CountingStars (slope 1.8)  explore_frac 0.3, cost_frac 0.3, bufferFloor 0.05"
echo "  10 runs per (series, delta, environment, metric)."
echo "  Limits: MAX_TREE_SIZE 3,000,000 | 10s per-run timeout | 20,000 outer-loop iteration cap"
echo "          (non-binding by design -- tree size and wall-clock are meant to stop every run)"
echo "  config.h MAX_ITER stays at 1000 (unchanged) -- see the header comment in this script and"
echo "  in examples/gpu/paper_benchmark.cu for why raising it would corrupt CountingStars' buffer ramp."
echo "  Total: 5 x 3 x 4 x 2 x 10 = 1,200 runs"
echo "======================================================="

# =============================================================================
# BUILD — compile once per (delta, cost metric), caching each binary
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
            make PaperBenchmark -j"$(nproc)" 2>&1 | tail -20
            # Cache under a (delta, metric)-suffixed name so the run phase needs no rebuild
            cp PaperBenchmark "PaperBenchmark_${DL}_${CL}"
            cd "$PROJECT_DIR"
        done
    done
else
    echo ""
    echo "=== SKIPPING BUILD PHASE (using cached binaries) ==="
    cd "$BUILD_DIR"
    for DL in "${DELTA_LABELS[@]}"; do
        for CL in "${COST_LABELS[@]}"; do
            if [ ! -f "PaperBenchmark_${DL}_${CL}" ]; then
                echo "ERROR: Cached binary not found: PaperBenchmark_${DL}_${CL}"
                echo "Run without --skip-build first to create cached binaries."
                exit 1
            fi
        done
    done
    echo "  All cached binaries found."
    cd "$PROJECT_DIR"
fi

# =============================================================================
# RUN — one pass per cost metric x environment x delta, using the cached binaries. Each invocation
# internally runs all 5 series x 10 runs.
# =============================================================================
cd "$BUILD_DIR"
for CL in "${COST_LABELS[@]}"; do
    for i in "${!ENV_NAMES[@]}"; do
        EN="${ENV_NAMES[$i]}"
        EO="${ENV_OBSTACLES[$i]}"
        for d in "${!DELTA_LABELS[@]}"; do
            DL="${DELTA_LABELS[$d]}"
            echo ""
            echo "=== RUNNING (delta=${DL}, cost=${CL}, Env=${EN}) ==="
            # argv[1] carries the discretization and cost metric, so it lands in every output
            # filename as _delta${DL}_${CL}; argv[3] selects the per-environment subfolder.
            "./PaperBenchmark_${DL}_${CL}" "${DL}_${CL}" "$EO" "$EN"
        done
    done
done
cd "$PROJECT_DIR"

echo ""
echo "======================================================="
echo "  PAPER BENCHMARK COMPLETE"
echo "======================================================="
for EN in "${ENV_NAMES[@]}"; do
    echo "Results in: $BUILD_DIR/Data/Benchmarks/Paper/${EN}/"
done
echo "Plot with:  scripts/process_paper_benchmark_and_plot.m (set envName at the top to match, run"
echo "            it once per environment from that directory)."
echo "Config.h will be restored to original on exit."
