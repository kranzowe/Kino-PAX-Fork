#!/usr/bin/env python3
"""
Analyze ReKinoLite parameter tuning results and generate visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Read results
results_path = Path("../build/Data/Tuning/tuning_results.csv")
if not results_path.exists():
    print(f"Error: Results file not found at {results_path}")
    print("Please run the tuning script first!")
    exit(1)

df = pd.read_csv(results_path)

print("=== ReKinoLite Tuning Analysis ===\n")
print(f"Total configurations: {len(df)}")
print(f"Sample values tested: {sorted(df['samplesPerThread'].unique())}")
print(f"Epsilon values tested: {sorted(df['epsilonGreedy'].unique())}")
print()

# Create output directory
fig_dir = Path("../figs/tuning")
fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Heatmap: Success Rate vs Parameters
plt.figure(figsize=(10, 8))
pivot_success = df.pivot(index='epsilonGreedy', columns='samplesPerThread', values='successRate')
sns.heatmap(pivot_success * 100, annot=True, fmt='.1f', cmap='RdYlGn',
            cbar_kws={'label': 'Success Rate (%)'})
plt.title('Success Rate by Parameters', fontsize=14, fontweight='bold')
plt.xlabel('Samples Per Thread', fontsize=12)
plt.ylabel('Epsilon Greedy', fontsize=12)
plt.tight_layout()
plt.savefig(fig_dir / 'success_rate_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / 'success_rate_heatmap.png'}")

# 2. Heatmap: Average Time vs Parameters
plt.figure(figsize=(10, 8))
pivot_time = df.pivot(index='epsilonGreedy', columns='samplesPerThread', values='avgTime')
sns.heatmap(pivot_time, annot=True, fmt='.3f', cmap='RdYlGn_r',
            cbar_kws={'label': 'Avg Time (s)'})
plt.title('Average Execution Time by Parameters', fontsize=14, fontweight='bold')
plt.xlabel('Samples Per Thread', fontsize=12)
plt.ylabel('Epsilon Greedy', fontsize=12)
plt.tight_layout()
plt.savefig(fig_dir / 'avg_time_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / 'avg_time_heatmap.png'}")

# 3. Effect of samplesPerThread (grouped by epsilon)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for epsilon in sorted(df['epsilonGreedy'].unique()):
    subset = df[df['epsilonGreedy'] == epsilon]
    ax1.plot(subset['samplesPerThread'], subset['successRate'] * 100,
             marker='o', label=f'ε={epsilon:.1f}', linewidth=2)
    ax2.plot(subset['samplesPerThread'], subset['avgTime'],
             marker='o', label=f'ε={epsilon:.1f}', linewidth=2)

ax1.set_xlabel('Samples Per Thread', fontsize=12)
ax1.set_ylabel('Success Rate (%)', fontsize=12)
ax1.set_title('Success Rate vs Samples Per Thread', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Samples Per Thread', fontsize=12)
ax2.set_ylabel('Average Time (s)', fontsize=12)
ax2.set_title('Execution Time vs Samples Per Thread', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'samples_effect.png', dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / 'samples_effect.png'}")

# 4. Effect of epsilon (grouped by samplesPerThread)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for samples in sorted(df['samplesPerThread'].unique()):
    subset = df[df['samplesPerThread'] == samples]
    ax1.plot(subset['epsilonGreedy'], subset['successRate'] * 100,
             marker='o', label=f'{samples} samples', linewidth=2)
    ax2.plot(subset['epsilonGreedy'], subset['avgTime'],
             marker='o', label=f'{samples} samples', linewidth=2)

ax1.set_xlabel('Epsilon Greedy', fontsize=12)
ax1.set_ylabel('Success Rate (%)', fontsize=12)
ax1.set_title('Success Rate vs Epsilon Greedy', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Epsilon Greedy', fontsize=12)
ax2.set_ylabel('Average Time (s)', fontsize=12)
ax2.set_title('Execution Time vs Epsilon Greedy', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'epsilon_effect.png', dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / 'epsilon_effect.png'}")

# 5. Pareto front: Success Rate vs Time
plt.figure(figsize=(10, 8))

# Color by samplesPerThread, marker by epsilon
for samples in sorted(df['samplesPerThread'].unique()):
    for epsilon in sorted(df['epsilonGreedy'].unique()):
        subset = df[(df['samplesPerThread'] == samples) & (df['epsilonGreedy'] == epsilon)]
        if not subset.empty:
            marker = ['o', 's', '^', 'D'][int(epsilon * 10) % 4]
            plt.scatter(subset['avgTime'], subset['successRate'] * 100,
                       s=150, marker=marker, alpha=0.7,
                       label=f's={samples}, ε={epsilon:.1f}')

plt.xlabel('Average Time (s)', fontsize=12)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.title('Pareto Front: Success Rate vs Execution Time', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'pareto_front.png', dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / 'pareto_front.png'}")

# Print summary statistics
print("\n=== Top 5 Configurations by Success Rate ===")
top_success = df.nlargest(5, 'successRate')
for idx, row in top_success.iterrows():
    print(f"  samples={row['samplesPerThread']:2d}, ε={row['epsilonGreedy']:.2f}: "
          f"{row['successRate']*100:.1f}% success, {row['avgTime']:.3f}s avg time")

print("\n=== Top 5 Configurations by Speed (among 100% success) ===")
perfect_success = df[df['successRate'] >= 0.99]
if len(perfect_success) > 0:
    top_speed = perfect_success.nsmallest(5, 'avgTime')
    for idx, row in top_speed.iterrows():
        print(f"  samples={row['samplesPerThread']:2d}, ε={row['epsilonGreedy']:.2f}: "
              f"{row['successRate']*100:.1f}% success, {row['avgTime']:.3f}s avg time")
else:
    print("  No configurations achieved 100% success rate")

print("\n=== Analysis Complete ===")
print(f"All figures saved to: {fig_dir}/")
