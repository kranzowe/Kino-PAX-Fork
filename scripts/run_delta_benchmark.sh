#!/bin/bash
# =============================================================================
# KinoPaxPlus Delta Benchmark Runner
#
# Iterates over 4 delta (region discretization) configurations, modifies
# config.h, recompiles, and runs the benchmark for each.
#
# Delta configs for MODEL 3 (12D nonlinear quadcopter):
#   large:     W=8  C=3  V=2  ->   110,592 regions (~10^5)
#   med_large: W=10 C=4  V=3  -> 1,728,000 regions (~1.7x10^6)
#   med_small: W=12 C=4  V=4  -> 7,077,888 regions (~7x10^6)
#   small:     W=14 C=4  V=4  -> 11,239,424 regions (~1.1x10^7)
#
# Usage: bash run_delta_benchmark.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/include/config/config.h"
BUILD_DIR="$PROJECT_DIR/build"

# Back up original config.h and restore on exit
cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
trap 'cp "$CONFIG_FILE.bak" "$CONFIG_FILE"; rm -f "$CONFIG_FILE.bak"; echo ""; echo "Restored original config.h"' EXIT

# Delta configurations: label, W_R1_LENGTH, C_R1_LENGTH, V_R1_LENGTH
LABELS=("large" "med_large" "med_small" "small")
W_VALS=(8 10 12 14)
C_VALS=(3 4 4 4)
V_VALS=(2 3 4 4)

echo "============================================================"
echo "  KinoPaxPlus Delta Benchmark"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "Config:  $CONFIG_FILE"
echo ""
echo "Configurations:"
for i in "${!LABELS[@]}"; do
    W="${W_VALS[$i]}"
    C="${C_VALS[$i]}"
    V="${V_VALS[$i]}"
    REGIONS=$(python3 -c "print($W**3 * $C**3 * $V**3)")
    echo "  ${LABELS[$i]}: W=$W C=$C V=$V -> $REGIONS regions"
done
echo "============================================================"
echo ""

mkdir -p "$BUILD_DIR"

for i in "${!LABELS[@]}"; do
    LABEL="${LABELS[$i]}"
    W="${W_VALS[$i]}"
    C="${C_VALS[$i]}"
    V="${V_VALS[$i]}"
    REGIONS=$(python3 -c "print($W**3 * $C**3 * $V**3)")

    echo ""
    echo "========================================"
    echo "  Delta: $LABEL (W=$W, C=$C, V=$V) -> $REGIONS regions"
    echo "========================================"

    # Modify the active (uncommented) R1 length defines in config.h
    sed -i "s/^#define W_R1_LENGTH.*/#define W_R1_LENGTH $W/" "$CONFIG_FILE"
    sed -i "s/^#define C_R1_LENGTH.*/#define C_R1_LENGTH $C/" "$CONFIG_FILE"
    sed -i "s/^#define V_R1_LENGTH.*/#define V_R1_LENGTH $V/" "$CONFIG_FILE"

    echo "  config.h updated"

    # Build
    echo "  Building..."
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    make KinoPaxPlusDeltaBenchmark -j$(nproc) 2>&1 | tail -3
    echo "  Build complete"

    # Run
    echo "  Running benchmark..."
    ./KinoPaxPlusDeltaBenchmark "$LABEL"

    cd "$PROJECT_DIR"
done

echo ""
echo "============================================================"
echo "  ALL DELTA BENCHMARKS COMPLETE"
echo "============================================================"
echo "Results in: $BUILD_DIR/Data/Benchmarks/KinoPaxPlusDelta/"
echo "============================================================"
