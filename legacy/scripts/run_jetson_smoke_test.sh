#!/bin/bash
# =============================================================================
# Jetson Smoke Test Runner
#
# Builds and runs examples/gpu/jetson_smoke_test.cu against the SWEEP'S REAL CONFIG -- MODEL 1,
# MAX_TREE_SIZE 3000000, the `large` delta (W_R1=10, V_R1=3, 27,000 regions), COST_MODE 0 -- the
# same config scripts/run_countingstars_sweep.sh writes. This is NOT a smaller "does it link"
# config: verifying the actual sweep footprint (~630 MiB per live planner) fits in an embedded
# GPU's shared memory is the entire point. A smaller tree would pass without proving anything.
#
# THIS IS A SMOKE TEST, NOT A BENCHMARK. No comparable numbers come out of it. It answers three
# yes/no questions per planner -- see the header of jetson_smoke_test.cu for what each means and
# why the old version (CUDA-error-only, no solve check, duplicated PruneKPAX, no CountingStars
# coverage, hardcoded Model-3-only start/goal) could not answer any of them honestly:
#
#   1. Did it SOLVE (not just "no CUDA error")
#   2. Did it LEAK (cudaMemGetInfo before/after each planner's scope)
#   3. Did it HANG (15s wall-clock budget per planner)
#
# Six planners: KPAX, PruneKPAX, KinoPaxSTAR, KinoPaxSTARCleanCost (the arm CountingStars is
# actually measured against), KinoPaxPlus, CountingStars.
#
# PREREQUISITES on a fresh JetPack image -- find_package(Python3 ... NumPy REQUIRED) and
# find_package(OpenMP REQUIRED) in CMakeLists.txt are hard requirements for EVERY target in this
# project, including this one, even though it uses neither:
#
#   sudo apt install python3-dev python3-numpy libomp-dev
#
# Usage:
#   cd scripts && bash run_jetson_smoke_test.sh
#   cd scripts && bash run_jetson_smoke_test.sh --skip-build   # run only (reuses the last build)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
CONFIG_BACKUP="$CONFIG_FILE.bak"
BUILD_DIR="$PROJECT_DIR/build"

# --- Parse arguments ---
SKIP_BUILD=false
for arg in "$@"; do
    if [ "$arg" = "--skip-build" ]; then
        SKIP_BUILD=true
    fi
done

# --- Auto-detect compilers (cluster has gcc-12, Jetson uses default gcc). Verbatim from the other
# scripts/run_*.sh -- this is the repo's one existing piece of Jetson-aware build logic. ---
CMAKE_COMPILER_FLAGS=""
if command -v gcc-12 &> /dev/null; then
    echo "Detected gcc-12 (cluster environment)"
    CMAKE_COMPILER_FLAGS="-DCMAKE_C_COMPILER=$(which gcc-12) -DCMAKE_CXX_COMPILER=$(which g++-12) -DCMAKE_CUDA_HOST_COMPILER=$(which g++-12)"
else
    echo "Using default system compilers (Jetson/local environment)"
fi

# --- Cap parallel compilation on Tegra. Every other scripts/run_*.sh uses `make -j"$(nproc)"`
# unconditionally; on a 6-core Orin Nano with 8 GB shared RAM, compiling KPAX_lib's 30 CUDA files
# with CUDA_SEPARABLE_COMPILATION ON at full parallelism will very likely OOM the host (not the
# GPU -- the compiler). /etc/nv_tegra_release exists on every JetPack image and nowhere else. ---
if [ -f /etc/nv_tegra_release ]; then
    echo "Detected Tegra (Jetson) -- capping build parallelism at -j2"
    BUILD_JOBS=2
else
    BUILD_JOBS="$(nproc)"
fi

# Restore config.h on exit, same as every other sweep script
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

# The sweep's exact `large`-delta Model 1 config -- see run_countingstars_sweep.sh's write_config
# for the same body; duplicated here rather than sourced because these are two independent CMake
# targets built from two independent scripts, and this one has exactly one operating point rather
# than a delta/cost-metric matrix.
write_config() {
    cat > "$CONFIG_FILE" << CONFIGEOF
#pragma once
/***************************/
/* 6D DOUBLE INTEGRATOR    */
/***************************/
#define MODEL 1
#define COST_MODE 0  // path cost: 1 = control effort ((ax^2+ay^2+az^2)*dt), 0 = workspace distance
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
#define W_R1_LENGTH 10
#define C_R1_LENGTH 1
#define V_R1_LENGTH 3
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

REGIONS=$(( 10**3 * 3**3 ))

echo ""
echo "======================================================="
echo "  JETSON SMOKE TEST"
echo "  Model: 1 (6D Double Integrator), MAX_TREE_SIZE=3000000"
echo "  Delta: large  | W_R1=10 C_R1=1 V_R1=3 | Regions=${REGIONS}"
echo "  (the sweep's real config -- this is what has to fit, not a smaller stand-in)"
echo "  Build jobs: ${BUILD_JOBS}"
echo "======================================================="

if [ "$SKIP_BUILD" = false ]; then
    write_config
    cd "$BUILD_DIR"
    # shellcheck disable=SC2086
    cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_COMPILER_FLAGS 2>&1 | tail -5
    make JetsonSmokeTest -j"$BUILD_JOBS" 2>&1 | tail -40
    cd "$PROJECT_DIR"
else
    echo ""
    echo "=== SKIPPING BUILD PHASE (using cached binary) ==="
    if [ ! -f "$BUILD_DIR/JetsonSmokeTest" ]; then
        echo "ERROR: $BUILD_DIR/JetsonSmokeTest not found. Run without --skip-build first."
        exit 1
    fi
fi

# The obstacle path in jetson_smoke_test.cu is relative ("../include/config/obstacles/..."), so
# this must run from build/, matching every other harness in this repo.
cd "$BUILD_DIR"
echo ""
echo "=== RUNNING ==="
set +e
./JetsonSmokeTest
STATUS=$?
set -e
cd "$PROJECT_DIR"

echo ""
echo "======================================================="
if [ "$STATUS" -eq 0 ]; then
    echo "  JETSON SMOKE TEST: ALL PLANNERS PASSED"
else
    echo "  JETSON SMOKE TEST: FAILED (exit ${STATUS}) -- see PASS/FAIL lines above"
fi
echo "======================================================="
echo "Config.h will be restored to original on exit."

exit "$STATUS"
