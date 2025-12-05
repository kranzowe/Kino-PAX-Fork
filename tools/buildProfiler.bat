@echo off
REM Build script for the propagation profiler (Windows)

echo Building propagation profiler...

REM Set project root (parent of tools directory)
set PROJECT_ROOT=%~dp0..

echo Project root: %PROJECT_ROOT%

REM CUDA compiler
set NVCC=nvcc

REM Compiler flags
set NVCC_FLAGS=-I"%PROJECT_ROOT%\include" -O3 -std=c++17 --expt-relaxed-constexpr

REM Architecture (adjust based on your GPU)
REM Common options: sm_60 (Pascal), sm_70 (Volta), sm_75 (Turing), sm_80 (Ampere), sm_86 (RTX 30xx), sm_89 (RTX 40xx)
set ARCH=-arch=sm_86

REM Source files
set SOURCES="%PROJECT_ROOT%\tools\profilePropagation.cu" "%PROJECT_ROOT%\src\statePropagator\statePropagatorProfiled.cu" "%PROJECT_ROOT%\src\statePropagator\statePropagator.cu" "%PROJECT_ROOT%\src\collisionCheck\collisionCheck.cu"

REM Output executable
set OUTPUT="%PROJECT_ROOT%\tools\profilePropagation.exe"

REM Build command
echo Compiling with: %NVCC% %NVCC_FLAGS% %ARCH%
%NVCC% %NVCC_FLAGS% %ARCH% %SOURCES% -o %OUTPUT%

if %ERRORLEVEL% EQU 0 (
    echo Build successful! Executable: %OUTPUT%
    echo.
    echo Usage: profilePropagation.exe [num_tests] [num_obstacles]
    echo Example: profilePropagation.exe 10000 100
) else (
    echo Build failed!
    exit /b 1
)
