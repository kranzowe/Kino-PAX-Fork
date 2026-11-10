"""Chain Reaction vs CountingStars (KinoPax#): per-environment comparison from the pared-down
ChainReactionBenchmark runs (scripts/paper/run_chain_reaction_benchmark.sh).

For every discretization folder and environment found under DATASET_DIR, writes one figure with
  row 1: best cost vs elapsed time (mean +/- 1 std over runs; unsolved samples are dropped, and a run
         that ended early -- tree full / empty frontier -- holds its last value),
  row 2: tree size vs elapsed time (mean over runs),
plus a summary CSV (success rate, time to first solution, final cost, final tree size, run time).

Input layout (same convention as zephyr_common.load_runs; copy the benchmark's output here):
    plots/DATA/CHAIN_REACTION_runs/discretizationCOARSE/<env>/<env>_<planner label>_deltam2_large_length_run<N>.csv
Chain Reaction has no regions, so the discretization only changes the CountingStars side.

Edit DATASET_DIR / MODEL_ID / METRIC below, then run:
    python plots/paper/chain_reaction_compare.py
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # plots/, for zephyr_common
from zephyr_common import (  # noqa: E402
    BASE_COLORS,
    BASE_DISPLAY,
    CHAIN_REACTION,
    COST_METRIC_LABELS,
    KINOPAX_STAR,
    MAX_FLOAT_THRESH,
    MODEL_NAMES,
    discover_discretization_dirs,
    discover_environments,
    discretization_display,
    discretization_label_from_dir,
    env_display_name,
    first_sol_time,
    load_runs,
)

PLOTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================================================================================================
# EDIT THESE
# ================================================================================================
DATASET_DIR = os.path.join(PLOTS_DIR, "DATA", "CHAIN_REACTION_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "chain_reaction")
MODEL_ID = 2          # 6D Dubins Airplane (what run_chain_reaction_benchmark.sh builds)
METRIC = "length"     # cost metric the benchmark was built with
PLANNERS = [KINOPAX_STAR, CHAIN_REACTION]   # CountingStars (displayed "KinoPax#") first, then Chain Reaction
COLUMNS = ("best_cost", "elapsed_time_ms", "tree_size", "frontier_size")
GRID_POINTS = 300


def resample(run: pd.DataFrame, column: str, grid: np.ndarray, drop_unsolved: bool) -> np.ndarray:
    """`column` of one run sampled at `grid` times, holding the last logged value after the run ends
    and NaN before its first logged iteration."""
    df = run[["elapsed_time_ms", column]].copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    if drop_unsolved:
        df.loc[df[column] > MAX_FLOAT_THRESH, column] = np.nan
    sampled = pd.merge_asof(pd.DataFrame({"elapsed_time_ms": grid}), df.sort_values("elapsed_time_ms"),
                            on="elapsed_time_ms", direction="backward")
    return sampled[column].to_numpy(dtype=float)


def stack(runs: list, column: str, grid: np.ndarray, drop_unsolved: bool) -> np.ndarray:
    return np.vstack([resample(r, column, grid, drop_unsolved) for r in runs])


def summarize(runs: list) -> dict:
    ttfs = [first_sol_time(r) for r in runs]
    solved = [t for t in ttfs if t >= 0]
    finals = []
    for r in runs:
        costs = pd.to_numeric(r["best_cost"], errors="coerce")
        costs = costs[costs < MAX_FLOAT_THRESH]
        if not costs.empty:
            finals.append(float(costs.iloc[-1]))
    return {
        "n_runs": len(runs),
        "success_rate": len(solved) / len(runs) if runs else math.nan,
        "mean_ttfs_ms": float(np.mean(solved)) if solved else math.nan,
        "std_ttfs_ms": float(np.std(solved)) if len(solved) > 1 else (0.0 if solved else math.nan),
        "mean_final_cost": float(np.mean(finals)) if finals else math.nan,
        "mean_final_tree_size": float(np.mean([r["tree_size"].iloc[-1] for r in runs])) if runs else math.nan,
        "mean_run_time_ms": float(np.mean([r["elapsed_time_ms"].iloc[-1] for r in runs])) if runs else math.nan,
    }


def plot_discretization(disc_dir: str, disc_label: str, summary_rows: list) -> str | None:
    envs = discover_environments(disc_dir)
    data = {}   # (env, planner) -> runs
    for env in envs:
        env_dir = os.path.join(disc_dir, env)
        for planner in PLANNERS:
            runs = load_runs(env_dir, env, planner, MODEL_ID, disc_label, metrics=(METRIC,), columns=COLUMNS)
            if runs:
                data[(env, planner)] = runs
                summary_rows.append({"Discretization": disc_label, "Environment": env,
                                     "Planner": BASE_DISPLAY[planner], **summarize(runs)})
    envs = [e for e in envs if any((e, p) in data for p in PLANNERS)]
    if not envs:
        print(f"  [warn] no run files for {PLANNERS} under {disc_dir}")
        return None

    t_max = max(float(r["elapsed_time_ms"].iloc[-1]) for runs in data.values() for r in runs)
    grid = np.linspace(0.0, t_max, GRID_POINTS)

    fig, axes = plt.subplots(2, len(envs), figsize=(5.2 * len(envs), 7.4), squeeze=False)
    for col, env in enumerate(envs):
        ax_cost, ax_tree = axes[0][col], axes[1][col]
        for planner in PLANNERS:
            runs = data.get((env, planner))
            if not runs:
                continue
            color, label = BASE_COLORS[planner], f"{BASE_DISPLAY[planner]} (n={len(runs)})"

            cost = stack(runs, "best_cost", grid, drop_unsolved=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN before anyone solves
                mean, std = np.nanmean(cost, axis=0), np.nanstd(cost, axis=0)
            if not np.all(np.isnan(mean)):
                ax_cost.plot(grid, mean, color=color, linewidth=2.0, label=label)
                ax_cost.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)

            tree = stack(runs, "tree_size", grid, drop_unsolved=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ax_tree.plot(grid, np.nanmean(tree, axis=0), color=color, linewidth=2.0, label=label)

        ax_cost.set_title(env_display_name(env), fontsize=13, fontweight="bold")
        ax_cost.set_ylabel(f"Best {COST_METRIC_LABELS[METRIC].lower()}" if col == 0 else "")
        ax_tree.set_ylabel("Tree size (nodes)" if col == 0 else "")
        ax_tree.set_xlabel("Elapsed planner time (ms)")
        for ax in (ax_cost, ax_tree):
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax_cost.legend(fontsize=9, loc="best")

    fig.suptitle(f"Chain Reaction vs {BASE_DISPLAY[KINOPAX_STAR]} -- {MODEL_NAMES[MODEL_ID]}, "
                 f"{discretization_display(disc_label)} discretization", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    base = f"chain_reaction_vs_countingstars_{MODEL_NAMES[MODEL_ID]}_{disc_label}"
    png_path = os.path.join(OUT_DIR, f"{base}.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{base}.svg"), bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    if not os.path.isdir(DATASET_DIR):
        raise SystemExit(
            f"DATASET_DIR does not exist: {DATASET_DIR!r} -- copy build/Data/Benchmarks/ChainReaction/<COARSE>/ "
            "into <DATASET_DIR>/discretizationCOARSE/ (see scripts/paper/run_chain_reaction_benchmark.sh)."
        )
    disc_dirs = discover_discretization_dirs(DATASET_DIR)
    if not disc_dirs:
        raise SystemExit(f"No 'discretization<LABEL>' folders found under {DATASET_DIR!r}.")

    os.makedirs(OUT_DIR, exist_ok=True)
    summary_rows: list = []
    for disc_dir in disc_dirs:
        label = discretization_label_from_dir(disc_dir)
        path = plot_discretization(disc_dir, label, summary_rows)
        if path:
            print(f"Wrote {path} + .svg")

    if summary_rows:
        csv_path = os.path.join(OUT_DIR, "chain_reaction_summary.csv")
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")
        print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
