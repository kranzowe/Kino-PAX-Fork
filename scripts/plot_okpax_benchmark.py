#!/usr/bin/env python3
"""
Visualization script for OKPAX comprehensive benchmark results.
Plots cost improvements over iterations and compares planner performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_benchmark_data(benchmark_dir):
    """Load all benchmark CSV files from a directory."""

    # Load summary
    summary_files = glob.glob(os.path.join(benchmark_dir, "*_summary.csv"))
    if not summary_files:
        print(f"No summary file found in {benchmark_dir}")
        return None, []

    summary_df = pd.read_csv(summary_files[0])

    # Load detailed iteration files
    detail_files = glob.glob(os.path.join(benchmark_dir, "*_run*.csv"))

    detailed_data = []
    for file in detail_files:
        filename = os.path.basename(file)
        parts = filename.replace('.csv', '').split('_')

        # Extract metadata from filename
        # Format: {timestamp}_{environment}_{planner_type}_run{N}.csv
        if len(parts) >= 4:
            environment = parts[-3]
            planner_type = parts[-2]
            run_number = int(parts[-1].replace('run', ''))

            df = pd.read_csv(file)
            df['environment'] = environment
            df['planner_type'] = planner_type
            df['run_number'] = run_number

            detailed_data.append(df)

    return summary_df, detailed_data

def plot_cost_convergence(detailed_data, output_dir):
    """Plot cost convergence over iterations for each environment."""

    if not detailed_data:
        print("No detailed data to plot")
        return

    # Combine all data
    all_data = pd.concat(detailed_data, ignore_index=True)

    environments = all_data['environment'].unique()
    planner_types = all_data['planner_type'].unique()

    for env in environments:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'OKPAX Performance Analysis - {env} Environment', fontsize=16, fontweight='bold')

        env_data = all_data[all_data['environment'] == env]

        # Plot 1: Cost over iterations (all runs)
        ax = axes[0, 0]
        for planner in planner_types:
            planner_data = env_data[env_data['planner_type'] == planner]
            for run in planner_data['run_number'].unique():
                run_data = planner_data[planner_data['run_number'] == run]
                ax.plot(run_data['iteration'], run_data['min_cost'],
                       alpha=0.3, linewidth=1)

        # Plot mean
        for planner in planner_types:
            planner_data = env_data[env_data['planner_type'] == planner]
            mean_data = planner_data.groupby('iteration')['min_cost'].mean().reset_index()
            ax.plot(mean_data['iteration'], mean_data['min_cost'],
                   label=planner, linewidth=3, alpha=0.9)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Minimum Cost')
        ax.set_title('Cost Convergence (all runs + mean)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Mean cost with confidence intervals
        ax = axes[0, 1]
        for planner in planner_types:
            planner_data = env_data[env_data['planner_type'] == planner]
            grouped = planner_data.groupby('iteration')['min_cost']
            mean = grouped.mean()
            std = grouped.std()

            ax.plot(mean.index, mean.values, label=planner, linewidth=2)
            ax.fill_between(mean.index,
                           mean.values - std.values,
                           mean.values + std.values,
                           alpha=0.2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Minimum Cost')
        ax.set_title('Mean Cost ± Std Dev')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Tree size growth
        ax = axes[1, 0]
        for planner in planner_types:
            planner_data = env_data[env_data['planner_type'] == planner]
            mean_data = planner_data.groupby('iteration')['tree_size'].mean().reset_index()
            ax.plot(mean_data['iteration'], mean_data['tree_size'],
                   label=planner, linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Tree Size')
        ax.set_title('Mean Tree Size Growth')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Iteration execution time
        ax = axes[1, 1]
        for planner in planner_types:
            planner_data = env_data[env_data['planner_type'] == planner]
            mean_data = planner_data.groupby('iteration')['execution_time_ms'].mean().reset_index()
            ax.plot(mean_data['iteration'], mean_data['execution_time_ms'],
                   label=planner, linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Mean Iteration Execution Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'cost_convergence_{env}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_summary_statistics(summary_df, output_dir):
    """Plot summary statistics comparing planners."""

    if summary_df is None or summary_df.empty:
        print("No summary data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OKPAX Summary Statistics', fontsize=16, fontweight='bold')

    # Plot 1: Final cost distribution
    ax = axes[0, 0]
    summary_df.boxplot(column='final_cost', by=['environment', 'planner_type'], ax=ax)
    ax.set_title('Final Cost Distribution')
    ax.set_ylabel('Final Cost')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')

    # Plot 2: Total time comparison
    ax = axes[0, 1]
    summary_df.boxplot(column='total_time_seconds', by=['environment', 'planner_type'], ax=ax)
    ax.set_title('Total Execution Time')
    ax.set_ylabel('Time (seconds)')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')

    # Plot 3: Final iterations
    ax = axes[1, 0]
    summary_df.boxplot(column='final_iterations', by=['environment', 'planner_type'], ax=ax)
    ax.set_title('Iterations to Convergence')
    ax.set_ylabel('Iterations')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')

    # Plot 4: Final tree size
    ax = axes[1, 1]
    summary_df.boxplot(column='final_tree_size', by=['environment', 'planner_type'], ax=ax)
    ax.set_title('Final Tree Size')
    ax.set_ylabel('Tree Size')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'summary_statistics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def generate_markdown_report(summary_df, detailed_data, output_dir):
    """Generate a markdown report with statistics."""

    if summary_df is None:
        return

    report_file = os.path.join(output_dir, 'benchmark_report.md')

    with open(report_file, 'w') as f:
        f.write("# OKPAX Comprehensive Benchmark Report\n\n")
        f.write("## Summary Statistics\n\n")

        for env in summary_df['environment'].unique():
            f.write(f"### Environment: {env}\n\n")
            env_data = summary_df[summary_df['environment'] == env]

            for planner in env_data['planner_type'].unique():
                planner_data = env_data[env_data['planner_type'] == planner]

                f.write(f"#### {planner}\n\n")
                f.write(f"- **Runs**: {len(planner_data)}\n")
                f.write(f"- **Mean Final Cost**: {planner_data['final_cost'].mean():.3f} ± {planner_data['final_cost'].std():.3f}\n")
                f.write(f"- **Mean Total Time**: {planner_data['total_time_seconds'].mean():.3f}s ± {planner_data['total_time_seconds'].std():.3f}s\n")
                f.write(f"- **Mean Iterations**: {planner_data['final_iterations'].mean():.1f} ± {planner_data['final_iterations'].std():.1f}\n")
                f.write(f"- **Mean Final Tree Size**: {planner_data['final_tree_size'].mean():.0f} ± {planner_data['final_tree_size'].std():.0f}\n")
                f.write(f"- **Best Cost**: {planner_data['final_cost'].min():.3f}\n")
                f.write(f"- **Worst Cost**: {planner_data['final_cost'].max():.3f}\n\n")

    print(f"Saved: {report_file}")

def main():
    # Find the most recent benchmark directory
    benchmark_base = Path("Data/Benchmarks/OKPAX")

    if not benchmark_base.exists():
        print(f"Error: Benchmark directory not found: {benchmark_base}")
        print("Please run the benchmark first: ./OKPAXComprehensiveBenchmark")
        sys.exit(1)

    # Create output directory for plots
    output_dir = benchmark_base / "plots"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading benchmark data from {benchmark_base}...")
    summary_df, detailed_data = load_benchmark_data(str(benchmark_base))

    if summary_df is None:
        print("Error: No benchmark data found")
        sys.exit(1)

    print(f"Loaded {len(summary_df)} benchmark runs")
    print(f"Loaded {len(detailed_data)} detailed iteration files")

    # Generate plots
    print("\nGenerating convergence plots...")
    plot_cost_convergence(detailed_data, str(output_dir))

    print("\nGenerating summary statistics plots...")
    plot_summary_statistics(summary_df, str(output_dir))

    print("\nGenerating markdown report...")
    generate_markdown_report(summary_df, detailed_data, str(output_dir))

    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nDone!")

if __name__ == "__main__":
    main()
