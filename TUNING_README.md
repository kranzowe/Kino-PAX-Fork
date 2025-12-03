# ReKinoLite Parameter Tuning Guide

This guide explains how to tune ReKinoLite's exploration parameters to optimize performance.

## Parameters

### 1. **samplesPerThread** (default: 1)
- How many control samples each thread generates per iteration
- Higher values = more exploration (each warp samples 32 × samplesPerThread controls)
- Range: 1-16 recommended
- Impact: More samples → better paths but slower execution

### 2. **epsilonGreedy** (default: 0.0)
- Probability of choosing a random valid sample instead of the greedy best
- 0.0 = pure greedy (always pick closest to goal)
- 1.0 = pure random (ignore goal distance)
- Range: 0.0-0.5 recommended
- Impact: Higher epsilon → more exploration, less exploitation

## How Tuning Works

**Greedy Strategy (epsilon=0.0, samples=1)**
- Each warp samples 32 random controls
- Always picks the one closest to goal
- Fast but may get stuck in local minima
- Expected: Tight paths toward goal

**More Samples (epsilon=0.0, samples=16)**
- Each warp samples 512 random controls (32 threads × 16 samples)
- Still picks the best one (greedy)
- Slower but more likely to find good paths
- Expected: Still converge toward goal, but from larger sample set

**Epsilon-Greedy (epsilon=0.2, samples=1)**
- 80% of the time: pick best of 32 samples
- 20% of the time: pick random valid sample
- Balanced exploration/exploitation
- Expected: More diverse paths, better obstacle avoidance

## Running the Tuning Script

```bash
# 1. Build the tuning script
cd build
cmake .. && make rekino_lite_tuning

# 2. Run tuning (tests multiple parameter combinations)
./rekino_lite_tuning

# 3. Analyze results
cd ../viz
python analyze_tuning.py
```

## What Gets Generated

### During Tuning:
- `Data/Tuning/tuning_results.csv` - Performance metrics for each configuration
- `Data/ReKinoLiteTree/rekino_lite_tree.csv` - Tree from first trial of each config

### After Analysis:
- `figs/tuning/success_rate_heatmap.png` - Success rate by parameters
- `figs/tuning/avg_time_heatmap.png` - Execution time by parameters
- `figs/tuning/samples_effect.png` - How samples per thread affects performance
- `figs/tuning/epsilon_effect.png` - How epsilon affects performance
- `figs/tuning/pareto_front.png` - Success vs time tradeoff

## Setting Custom Parameters

To use specific parameters in your code:

```cpp
ReKinoLite planner;

// Set parameters before calling plan()
planner.h_samplesPerThread_ = 8;    // Sample 8 controls per thread
planner.h_epsilonGreedy_ = 0.1f;    // 10% random exploration

planner.plan(h_initial, h_goal, d_obstacles, numObstacles, true);
```

## Interpreting Results

**High samples, low epsilon:**
- Best for finding optimal paths
- Slower execution
- Paths converge tightly toward goal

**Low samples, high epsilon:**
- Faster execution
- More diverse exploration
- May miss optimal paths but good for obstacle avoidance

**Sweet spot (hypothesis):**
- samples = 4-8
- epsilon = 0.1-0.2
- Good balance of speed and path quality

## Visualizing Individual Configurations

After tuning, visualize a specific configuration's tree:

```bash
# Set desired parameters in rekino_lite_main.cu
planner.h_samplesPerThread_ = 8;
planner.h_epsilonGreedy_ = 0.1f;

# Rebuild and run
make rekino_lite_main && ./rekino_lite_main

# Plot the tree
cd ../viz
matlab -nodisplay -r "plot_rekino_tree; exit"
```

## Expected Outcomes

Based on your observation that paths are more diverse than expected with default settings (samples=1, epsilon=0), testing higher sample counts should show:

1. **samples=16, epsilon=0**: Very tight convergence to goal (pure greedy with massive sampling)
2. **samples=1, epsilon=0.3**: More diverse paths (random exploration)
3. **samples=8, epsilon=0.1**: Best of both worlds?

The tuning script will help you find the optimal balance for your specific problem!
