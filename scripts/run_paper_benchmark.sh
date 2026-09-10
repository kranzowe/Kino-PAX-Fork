#!/bin/bash
# =============================================================================
# Paper Benchmark Runner
#
# A FIXED, FOUR-WAY COMPARISON, not a sweep. Every series below is an already-chosen operating
# point; the only things varying across runs are discretization, environment, and cost metric.
# Modeled on run_countingstars_sweep.sh's two-phase build-then-run structure, but simpler: there is
# no per-delta arm partition (--only-kinopaxplus has no equivalent here) because every series runs
# at every delta -- that is the comparison this suite exists to make.
#
# MODEL 2 THIS PASS -- 6D DUBINS AIRPLANE, not the 6D Double Integrator earlier passes ran. See
# write_config() below: STATE_DIM/CONTROL_DIM/SAMPLE_DIM/W_DIM/C_DIM/V_DIM are UNCHANGED from
# Model 1 (Dubins Airplane's state [x,y,z,yaw,pitch,v] and control-like [yawRate,pitchRate,a,dt]
# occupy the exact same SAMPLE_DIM slots the double integrator's [x,y,z,vx,vy,vz]/[ax,ay,az,dt]
# did, confirmed against propagateAndCheckDubinsAirplaneRungeKutta() in statePropagator.cu), so
# the delta/region machinery below needs no dimensional changes to switch models.
#
# CAVEAT, NOT FIXED HERE: getRegion()/getSubRegion() (Graph.cu) bin coord[3], coord[4], coord[5]
# (the V_DIM=3 slots) uniformly against ONE shared V_MIN/V_MAX = [-0.3, 0.3]. For Model 1 that is
# vx/vy/vz, all genuinely in that range. For Model 2 those same three slots are yaw ([-pi,pi]),
# pitch ([-pi/3,pi/3]) and v ([-0.3,0.3]) -- only v is actually calibrated to V_MIN/V_MAX; yaw and
# pitch values outside [-0.3,0.3] (i.e. almost all of them) clamp into the nearest end bucket, so
# the SYCLOP-style region signal is degenerate along those two axes specifically. This does NOT
# affect correctness (collision checking and the goal test are workspace-only, see distance() in
# helper.cuh), only how well regions distinguish states by orientation. Fixing it properly means
# giving yaw/pitch their own calibrated axes (e.g. C_DIM=1 at yaw using the existing C_MIN/C_MAX =
# -pi/pi, which fits exactly) -- a bigger change, not attempted in this pass.
#
# THE FOUR SERIES (fixed inside examples/gpu/paper_benchmark.cu, not swept here):
#   KPAX                     defaults
#   KinoPaxPlus               defaults
#   KinoPaxSTARTrue           h_syclopCap_ 1.0 (no cap), h_ancestorPrune_ 1 -- KPAX's exploration
#                             accept OR-fused with KinoPaxPlus's region-best accept, no cost
#                             shaping at all, plus the cost-guarded stale-best prune on top.
#                             Replaces KinoPaxSTARCleanCost as the non-CountingStars "STAR"
#                             reference this pass -- CleanCost told a "beats one already-tuned
#                             competitor" story; this tells "beats the naive fusion of its two
#                             parents" instead. A second point at h_ancestorPrune_ = 0 (the pure
#                             fusion, == stock KinoPaxSTARNoGoalBias) ran alongside this one in an
#                             earlier pass to isolate what the guarded prune buys on top of the
#                             naive fusion; it was dropped from this comparison (recoverable from
#                             git history as KinoPaxSTARTrue_cap100_anc0).
#   CountingStars             bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15, cost_frac 0.75,
#                             h_hopelessGuard_ PERMANENTLY ON (v3.5) -- countingstars_sweep.cu's
#                             own on/off sweep confirmed the guard helps, and this is the re-tuned
#                             bufferSlope/bufferFloor/explore_frac/cost_frac point that followed.
#                             Replaces the earlier two-point (bf0.3/bf0.6), unguarded arm. Runs on
#                             the permanently-budgeted CountingStars (OPTIMAL nodes are limited by
#                             budget in BOTH acceptance and reactivation, neither uncapped) -- see
#                             CS_DOORBIT_GUAR / CS_DOORBIT_OPTIMAL in CountingStars.cuh.
#
# THREE DELTAS, ALL FOUR SERIES AT EACH:
#   large  W_R1=7   C_R1=1  V_R1=3  ->   7^3 * 3^3 =   9,261 regions
#   fine   W_R1=16  C_R1=1  V_R1=4  ->  16^3 * 4^3 = 262,144 regions
#   tiny   W_R1=14  C_R1=1  V_R1=6  ->  14^3 * 6^3 = 592,704 regions
# "fine" and "tiny" refine different axes (workspace vs. velocity) -- same convention as
# countingstars_sweep.cu's fine/fine_control pair -- but are NOT an identical-region-count pair.
#
# TINY CHANGED FROM (W_R1=17, V_R1=5) THIS PASS. That discretization crashed with a
# cudaErrorIllegalAddress in the `empty` environment (a confirmed buffer-overflow bug in
# KPAX.cu/KinoPaxPlus.cu's goal-path reconstruction, fixed at three sites -- but a hang with the
# same symptoms persisted afterward, root cause still open). countingstars_sweep.cu stood up a
# dedicated bug-isolation harness at (W_R1=14, V_R1=6) instead and confirmed it clean across all
# five series (KPAX, KinoPaxPlus, KinoPaxSTARTrue anc0/anc1, CountingStars) on zigzag/effort. This
# pass reuses that same, confirmed-safe discretization here rather than the one that hangs -- see
# scripts/run_countingstars_sweep.sh's header for the fuller debugging history. THAT "CONFIRMED
# CLEAN" CLAIM NEVER ACTUALLY COVERED `empty` (only zigzag) OR MODEL 2 (only Model 1) -- and a run
# at tiny/empty on Model 2 has since frozen with the same symptoms. `empty` is pulled out of
# ENV_NAMES below (see the note there) to isolate whether it's specifically responsible.
#
# C_R1 stays at 1 everywhere: this config sets C_DIM 0, so control refinement has nowhere to act
# except V_R1.
#
# THREE ENVIRONMENTS THIS PASS: house, narrowPassage, zigzag -- `empty` pulled out as a diagnostic
# (see ENV_NAMES below); normally FOUR (empty, house, narrowPassage, zigzag). zigzag's five doorway
# gaps were tightened from 0.10 to 0.02 wide (include/config/obstacles/zigzag/obstacles.csv) to
# match narrowPassage's clearance exactly -- expect both to show materially lower success rates
# than house (and empty, once it's back).
#
# BOTH COST METRICS THIS PASS (length AND effort -- see COST_LABELS/COST_MODES below). EFFORT
# NEEDED A REAL MODEL-2 BRANCH FIRST: edgeCost() (include/helper/helper.cuh) used to gate its
# control-effort formula on `MODEL == 1` specifically, so COST_MODE=1 would have silently fallen
# through to the SAME workspace-distance formula COST_MODE=0 uses for any other model -- running
# "both" would have measured the same thing twice under different labels. A `MODEL == 2` branch
# was added (yawRate/pitchRate/a occupy the exact SAME x1[6..8] slots ax/ay/az did for Model 1, so
# the formula is the identical shape: (yawRate^2 + pitchRate^2 + a^2) * dt) before this pass ran.
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
# SCALE: 4 series x 3 deltas x 3 environments x 2 cost metrics x 5 runs = 360 runs this pass
# (480 with all four environments), each capped at 10s. Worst case several hours; most runs stop
# earlier (tree-full or an early success).
#
# NUM_R1_REGIONS and COST_MODE are both COMPILE-TIME, so neither can vary within one binary. Same
# build-cache pattern as run_countingstars_sweep.sh: write config.h and build once per (delta, cost
# metric) = 6 binaries this pass (3 deltas x 2 metrics), cached under a suffixed name, then run
# each once per environment.
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
DELTA_W_R1S=(7 16 14)
DELTA_C_R1S=(1  1  1)   # inert for Model 2 (C_DIM 0, unchanged from Model 1); control refinement rides on V_R1
DELTA_V_R1S=(3  4  6)

# Cost metric axis: label + COST_MODE (0 = workspace distance, 1 = control effort). LENGTH ONLY
# this pass -- effort disabled for now, not removed; uncomment the line below to restore it.
COST_LABELS=("length" "effort")
COST_MODES=(0 1)
# COST_LABELS=("length")
# COST_MODES=(0)

# Environments (obstacles already in [0,1]^3 workspace boxes -- model-agnostic, since collision
# checking and the goal test only read x,y,z; reused as-is for Model 2). Each gets its own output
# subfolder.
ENV_NAMES=("house" "narrowPassage" "zigzag")
ENV_OBSTACLES=(
    "../include/config/obstacles/house/obstacles.csv"
    "../include/config/obstacles/narrowPassage/obstacles.csv"
    "../include/config/obstacles/zigzag/obstacles.csv"
)
# `empty` REMOVED THIS PASS -- DIAGNOSTIC, NOT PERMANENT. A run at tiny/empty froze (no progress,
# GPU idle, host CPU spinning -- the same signature as the still-unresolved hang this suite has
# hit before). `empty` was the one environment tiny's (W_R1=14, V_R1=6) "confirmed clean" claim
# never actually covered -- countingstars_sweep.cu's isolation harness only ever ran zigzag (see
# the NOTE a few lines up) -- and this is also the very first pass on Model 2 (Dubins Airplane), a
# combination genuinely never tested before. Pulling `empty` out isolates whether it (or the
# empty/tiny pairing specifically) is what's triggering it, before spending more time chasing it
# blind. Restore the line below once that's answered.
# ENV_NAMES=("empty" "house" "narrowPassage" "zigzag")
# ENV_OBSTACLES=(
#     "../include/config/obstacles/empty/obstacles.csv"
#     "../include/config/obstacles/house/obstacles.csv"
#     "../include/config/obstacles/narrowPassage/obstacles.csv"
#     "../include/config/obstacles/zigzag/obstacles.csv"
# )

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

# Function to write complete Model 2 config.h. Identical structure to run_countingstars_sweep.cu's --
# MAX_ITER stays at 1000 regardless of delta/metric; see the header comment above for why.
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
echo "  Model: 2 (6D Dubins Airplane)"
echo "  Environments: ${ENV_NAMES[*]}  (separate output subfolders)"
for i in "${!DELTA_LABELS[@]}"; do
    R=$(( DELTA_W_R1S[i]**3 * DELTA_V_R1S[i]**3 ))
    echo "  Delta: ${DELTA_LABELS[$i]} | W_R1=${DELTA_W_R1S[$i]} C_R1=${DELTA_C_R1S[$i]} V_R1=${DELTA_V_R1S[$i]} | Regions=${R} | all 4 series"
done
echo "  Cost metrics: ${COST_LABELS[*]}  (one build each)"
echo "  Series (fixed, all 3 deltas x all 4 environments x both cost metrics):"
echo "    KPAX                      defaults"
echo "    KinoPaxPlus                defaults"
echo "    KinoPaxSTARTrue            h_syclopCap_ 1.0 (no cap), h_ancestorPrune_ 1 -- naive OR-fusion"
echo "                               + cost-guarded stale-best prune"
echo "    CountingStars              bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15, cost_frac 0.75,"
echo "                               hopelessGuard ON (v3.5, permanent)"
echo "  5 runs per (series, delta, environment, metric)."
echo "  Limits: MAX_TREE_SIZE 3,000,000 | 10s per-run timeout | 20,000 outer-loop iteration cap"
echo "          (non-binding by design -- tree size and wall-clock are meant to stop every run)"
echo "  config.h MAX_ITER stays at 1000 (unchanged) -- see the header comment in this script and"
echo "  in examples/gpu/paper_benchmark.cu for why raising it would corrupt CountingStars' buffer ramp."
echo "  Total: 4 x 3 x 3 x 2 x 5 = 360 runs"
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
# internally runs all 4 series x 5 runs.
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
