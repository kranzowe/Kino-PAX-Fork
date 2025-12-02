#!/bin/bash

# Build script for the propagation profiler

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Building propagation profiler..."
echo "Project root: $PROJECT_ROOT"

# CUDA compiler
NVCC=nvcc

# Compiler flags
NVCC_FLAGS="-I${PROJECT_ROOT}/include -O3 -std=c++17 --expt-relaxed-constexpr"

# Architecture (adjust based on your GPU)
# Common options: sm_60 (Pascal), sm_70 (Volta), sm_75 (Turing), sm_80 (Ampere), sm_86 (RTX 30xx), sm_89 (RTX 40xx)
ARCH="-arch=sm_86"

# Source files
SOURCES=(
    "${PROJECT_ROOT}/tools/profilePropagation.cu"
    "${PROJECT_ROOT}/src/statePropagator/statePropagatorProfiled.cu"
    "${PROJECT_ROOT}/src/statePropagator/statePropagator.cu"
    "${PROJECT_ROOT}/src/collisionCheck/collisionCheck.cu"
)

# Output executable
OUTPUT="${PROJECT_ROOT}/tools/profilePropagation"

# Build command
echo "Compiling with: $NVCC $NVCC_FLAGS $ARCH"
$NVCC $NVCC_FLAGS $ARCH "${SOURCES[@]}" -o "$OUTPUT"

if [ $? -eq 0 ]; then
    echo "Build successful! Executable: $OUTPUT"
    echo ""
    echo "Usage: $OUTPUT [num_tests] [num_obstacles]"
    echo "Example: $OUTPUT 10000 100"
else
    echo "Build failed!"
    exit 1
fi
