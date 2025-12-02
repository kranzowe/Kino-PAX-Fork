# Propagation Profiler

This tool profiles the `propagateAndCheckQuadRungeKutta` function to identify performance bottlenecks.

## Building

### Windows
```bash
cd tools
buildProfiler.bat
```

### Linux/Mac
```bash
cd tools
chmod +x buildProfiler.sh
./buildProfiler.sh
```

**Note:** You may need to adjust the GPU architecture in the build script. Open `buildProfiler.bat` or `buildProfiler.sh` and modify the `ARCH` variable:
- RTX 40xx series: `sm_89`
- RTX 30xx series: `sm_86`
- RTX 20xx series (Turing): `sm_75`
- GTX 10xx series (Pascal): `sm_60`

## Running

```bash
# Basic usage (10,000 tests with 100 obstacles)
./profilePropagation

# Custom number of tests and obstacles
./profilePropagation 50000 200

# Windows
profilePropagation.exe 50000 200
```

## Output

The profiler will show:
1. Total kernel execution time
2. Average time per propagation
3. Total iterations executed across all tests
4. Average iterations per propagation
5. **Detailed breakdown** showing cycles, percentage, and time for each component:
   - Random Generation (curand calls)
   - ODE Computation (Runge-Kutta integration)
   - Dynamics Check (velocity bounds checking)
   - Workspace Check (position bounds checking)
   - BB Construction (bounding box computation)
   - Collision Check (isMotionValid call)

## What to Look For

The breakdown will help you identify:
- **If collision checking is the bottleneck**: Look for high percentage in "Collision Check"
- **If integration is slow**: Check "ODE Computation" percentage
- **If random number generation is costly**: Check "Random Generation" percentage

This data will guide optimization efforts (e.g., spatial hashing for collision, early termination, etc.)

## Limitations

- This profiler uses `clock64()` which measures GPU cycles, not wall-clock time
- Atomic operations for accumulating stats add some overhead
- The profiler only works with the Quad model (MODEL=3)
- Results may vary based on:
  - Number and distribution of obstacles
  - Random seed affecting propagation duration
  - GPU architecture and clock speed

## Next Steps

Based on profiling results, consider:
1. **If collision is >50%**: Implement spatial hashing
2. **If ODE is >30%**: Consider lower-order integration or adaptive step sizes
3. **If early termination would help**: Add collision exit optimization
