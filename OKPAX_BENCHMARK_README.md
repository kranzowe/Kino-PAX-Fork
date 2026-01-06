# OKPAX Comprehensive Benchmark

A comprehensive benchmark suite for evaluating OKPAX performance with detailed iteration tracking and cost convergence analysis.

## Overview

This benchmark runs the OKPAX planner for **300 iterations** across **10 runs** in **3 different environments**, tracking:
- Minimum cost at each iteration
- Tree size growth
- Goal set size
- Per-iteration execution time
- Total execution time

## Features

- **Detailed Iteration Tracking**: Records cost, tree size, and timing for every iteration
- **Multiple Environments**: Tests on Empty, Pillars, and Narrow Passage scenarios
- **Statistical Analysis**: Computes means, standard deviations, and convergence metrics
- **Rich Visualizations**: Generates plots showing cost convergence, tree growth, and performance comparison
- **Markdown Reports**: Auto-generates summary statistics in markdown format

## Building

The benchmark executable is automatically built with the project:

```bash
cd build
cmake ..
make OKPAXComprehensiveBenchmark
```

## Running the Benchmark

### Run the Full Benchmark (300 iterations × 10 runs × 3 environments)

```bash
cd build
./OKPAXComprehensiveBenchmark
```

**Expected runtime**: 20-40 minutes depending on hardware

### Output Files

Results are saved to `Data/Benchmarks/OKPAX/`:

- `okpax_benchmark_{timestamp}_summary.csv` - Summary of all runs
- `okpax_benchmark_{timestamp}_{environment}_{planner}_run{N}.csv` - Detailed iteration data for each run

### Summary CSV Format

```csv
environment,planner_type,run_number,total_time_seconds,final_cost,final_iterations,final_tree_size
Empty,OKPAX_Original,0,12.345,145.678,287,15234
...
```

### Detailed CSV Format

```csv
iteration,min_cost,execution_time_ms,tree_size,goal_set_size
1,999.999,45.2,512,0
2,856.432,48.1,1024,1
3,723.891,51.3,1536,3
...
```

## Analyzing Results

### Generate Visualizations (MATLAB)

1. **Download results from Linux machine**:
   ```bash
   # On your local machine (from Linux SSH session)
   scp -r username@server:path/to/Kino-PAX/build/Data/Benchmarks/OKPAX ./
   ```

2. **Open MATLAB and run visualization script**:
   ```matlab
   % In MATLAB, navigate to the scripts directory
   cd scripts

   % Edit the DATA_DIR variable in plot_okpax_benchmark.m if needed
   % Then run:
   plot_okpax_benchmark
   ```

This generates:
- **Cost Convergence Plots**: Shows how cost improves over iterations
  - All individual runs (transparent lines)
  - Mean convergence (bold line)
  - Standard deviation bands
- **Tree Size Growth**: Shows memory usage over time
- **Iteration Timing**: Shows computational cost per iteration
- **Summary Statistics**: Box plots comparing final costs, times, and tree sizes

### Output

Plots are saved to `Data/Benchmarks/OKPAX/plots/`:
- `cost_convergence_Empty.png` / `.fig`
- `cost_convergence_Pillars.png` / `.fig`
- `cost_convergence_NarrowPassage.png` / `.fig`
- `summary_statistics.png` / `.fig`
- `benchmark_report.txt`

The `.fig` files can be opened in MATLAB for interactive viewing and further customization.

## Example Results Interpretation

### Cost Convergence

The benchmark tracks `h_minCost_` which represents the best (lowest) cost path found to the goal. Key insights:

- **Early iterations (1-50)**: Rapid cost improvement as planner explores and finds initial solutions
- **Middle iterations (50-150)**: Slower improvement as planner refines paths
- **Late iterations (150-300)**: Diminishing returns, approaching optimal solution

### Tree Size Growth

Monitoring tree size helps understand:
- **Memory usage**: Larger trees require more GPU memory
- **Pruning effectiveness**: Good pruning keeps tree size manageable
- **Exploration vs exploitation**: Rapid growth = exploration, plateau = exploitation

### Convergence Criteria

The planner stops early if:
- `h_propIterations_ == 0`: No more profitable nodes to expand
- All regions have been thoroughly explored

## Customization

### Modify Environments

Edit `examples/gpu/okpax_comprehensive_benchmark.cu`:

```cpp
// Add new environment
{
    float h_initial[SAMPLE_DIM] = {...};
    float h_goal[SAMPLE_DIM]    = {...};

    runEnvironmentBenchmark(
        "MyEnvironment",
        "../include/config/obstacles/myenv/obstacles.csv",
        h_initial, h_goal, all_results, NUM_RUNS, MAX_ITERATIONS);
}
```

### Adjust Parameters

```cpp
const int NUM_RUNS = 10;        // Number of runs per environment
const int MAX_ITERATIONS = 300; // Maximum iterations per run
```

### Compare Original vs Spatial Hash

To compare OKPAX with the spatial hash version (PruneKPAX), uncomment the PruneKPAX section in the benchmark code (currently commented out as it requires interface adaptation).

## Understanding the Metrics

### Minimum Cost (`min_cost`)
- The cost of the best path found so far
- Lower is better
- Should monotonically decrease or stay constant

### Tree Size (`tree_size`)
- Number of nodes in the planning tree
- Grows as planner explores
- Affected by pruning strategies

### Goal Set Size (`goal_set_size`)
- Number of nodes that reached the goal
- More solutions = better chance of finding optimal path
- OKPAX can track multiple goal configurations

### Execution Time
- **Per-iteration time**: Time to propagate frontier and update
- **Total time**: End-to-end benchmark duration
- Useful for comparing algorithmic efficiency

## Troubleshooting

### Out of Memory
If you encounter OOM errors, reduce:
- `MAX_ITERATIONS` (fewer iterations)
- `NUM_RUNS` (fewer runs)
- Or run one environment at a time

### Missing Dependencies
Ensure you have MATLAB installed (any recent version should work).

### No Results Generated
Check that:
1. Benchmark completed successfully
2. `Data/Benchmarks/OKPAX/` directory exists
3. CSV files were created

## Performance Tips

### For Faster Benchmarking
- Reduce `MAX_ITERATIONS` to 100-150
- Reduce `NUM_RUNS` to 3-5
- Test one environment at a time

### For More Detailed Analysis
- Increase `NUM_RUNS` to 20-30 for statistical significance
- Run with different random seeds
- Compare across different obstacle densities

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{okpax_benchmark,
  title = {OKPAX Comprehensive Benchmark Suite},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo/Kino-PAX}
}
```

## Contributing

To add new benchmark scenarios or metrics:
1. Modify `okpax_comprehensive_benchmark.cu`
2. Update `IterationData` struct for new metrics
3. Extend visualization scripts as needed

## License

Same as Kino-PAX project license.
