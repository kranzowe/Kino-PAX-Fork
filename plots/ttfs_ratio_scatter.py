"""Time-to-first-solution ratio scatter: Kino-PAX+ (y-axis) vs every other planner (x-axis).

One figure per model. Within each figure, one point per (other-algorithm x environment x
discretization) triple, averaged over every successful run (pooled across the length and effort
cost-metric sweeps, since the cost metric doesn't affect solve timing). The "empty" environment is
excluded (not interesting -- every planner solves it trivially). Color = algorithm, shape =
discretization (only "tiny" exists today; "fine"/"coarse" are picked up automatically once those
sweeps are added alongside it -- no code changes needed). Every algorithm compared within one
(environment, discretization) cell shares Kino-PAX+'s mean as its y-value, so they land on a
shared horizontal line -- that line is labeled directly with the environment name. The y=x line
marks "as fast as Kino-PAX+". See plots/zephyr_common.py for the on-disk naming conventions this
ports from scripts/process_paper_benchmark_improvement_scatter.m.

Edit DATASET_DIR / OUT_DIR below to point at the dataset you want to plot, then run:
    python plots/ttfs_ratio_scatter.py
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import (  # noqa: E402
    BASE_COLORS,
    BASE_DISPLAY,
    KINOPAX_PLUS,
    MODEL_IDS,
    MODEL_NAMES,
    OTHER_PLANNERS,
    aggregate_ttfs,
    discover_discretization_dirs,
    discover_environments,
    discretization_display,
    discretization_label_from_dir,
    discretization_marker,
    env_display_name,
    load_runs,
    sanitize_name,
    warn_on_unexpected_star_suffixes,
)

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================
# EDIT THESE to point at the dataset and output location you want.
# DATASET_DIR must directly contain one or more "discretization<LABEL>" folders (tiny/fine/coarse),
# each of which directly contains the empty/house/narrowPassage/zigzag subfolders.
# ================================================================================================
DATASET_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "ttfs_ratio")

EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison


def build_model_table(model_id: int, discretization_dirs: list[str]) -> pd.DataFrame:
    rows = []
    for disc_dir in discretization_dirs:
        discretization_label = discretization_label_from_dir(disc_dir)
        environments = [e for e in discover_environments(disc_dir) if e.lower() not in EXCLUDED_ENVIRONMENTS]
        for env in environments:
            env_dir = os.path.join(disc_dir, env)
            warn_on_unexpected_star_suffixes(env_dir)
            plus_stats = aggregate_ttfs(load_runs(env_dir, env, KINOPAX_PLUS, model_id, discretization_label))
            rows.append({
                "Discretization": discretization_label,
                "Environment": env,
                "Model": MODEL_NAMES[model_id],
                "Algorithm": BASE_DISPLAY[KINOPAX_PLUS],
                "N_Success": plus_stats.n_success,
                "N_Total": plus_stats.n_total,
                "Mean_TTFS_ms": plus_stats.mean,
                "Std_TTFS_ms": plus_stats.std,
                "KinoPaxPlus_Mean_TTFS_ms": plus_stats.mean,
                "Ratio_to_KinoPaxPlus": 1.0 if not math.isnan(plus_stats.mean) else math.nan,
            })
            for planner in OTHER_PLANNERS:
                stats = aggregate_ttfs(load_runs(env_dir, env, planner, model_id, discretization_label))
                ratio = math.nan
                if not math.isnan(stats.mean) and not math.isnan(plus_stats.mean) and plus_stats.mean > 0:
                    ratio = stats.mean / plus_stats.mean
                rows.append({
                    "Discretization": discretization_label,
                    "Environment": env,
                    "Model": MODEL_NAMES[model_id],
                    "Algorithm": BASE_DISPLAY[planner],
                    "N_Success": stats.n_success,
                    "N_Total": stats.n_total,
                    "Mean_TTFS_ms": stats.mean,
                    "Std_TTFS_ms": stats.std,
                    "KinoPaxPlus_Mean_TTFS_ms": plus_stats.mean,
                    "Ratio_to_KinoPaxPlus": ratio,
                })
    return pd.DataFrame(rows)


def plot_model(table: pd.DataFrame, model_id: int, discretization_labels: list[str], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    legends = []

    plotted = table[
        (table["Algorithm"] != BASE_DISPLAY[KINOPAX_PLUS])
        & table["Mean_TTFS_ms"].notna()
        & table["KinoPaxPlus_Mean_TTFS_ms"].notna()
    ]

    if plotted.empty:
        ax.text(0.5, 0.5, "No successful runs found for this model yet.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#666666")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        xs = plotted["Mean_TTFS_ms"].to_numpy()
        ys = plotted["KinoPaxPlus_Mean_TTFS_ms"].to_numpy()
        lo = min(xs.min(), ys.min()) / 1.5
        hi = max(xs.max(), ys.max()) * 1.5

        ax.plot([lo, hi], [lo, hi], linestyle="--", color=BASE_COLORS[KINOPAX_PLUS], linewidth=1.4, zorder=1)

        display_to_token = {v: k for k, v in BASE_DISPLAY.items()}
        for _, row in plotted.iterrows():
            planner_token = display_to_token[row["Algorithm"]]
            ax.scatter(
                row["Mean_TTFS_ms"], row["KinoPaxPlus_Mean_TTFS_ms"],
                s=150, marker=discretization_marker(row["Discretization"]),
                facecolors=BASE_COLORS[planner_token], edgecolors="black", linewidths=0.8,
                zorder=3,
            )

        # Every algorithm compared within one (environment, discretization) cell shares that
        # cell's Kino-PAX+ mean as its y-value, so they land on one shared horizontal line --
        # draw that line explicitly and label it with the environment name. Rows can end up at
        # nearly the same y (e.g. two environments of similar difficulty for one model), which
        # would otherwise print two labels on top of each other -- nudge later labels down (in
        # sorted order) to keep a minimum multiplicative gap; the guide line itself always stays
        # at the true y.
        MIN_LABEL_Y_RATIO = 1.35
        row_groups = sorted(
            plotted.groupby(["Environment", "Discretization"]),
            key=lambda item: item[1]["KinoPaxPlus_Mean_TTFS_ms"].iloc[0],
            reverse=True,
        )
        prev_label_y = None
        for (env, _disc), group in row_groups:
            y = group["KinoPaxPlus_Mean_TTFS_ms"].iloc[0]
            x_min = group["Mean_TTFS_ms"].min()
            x_max = group["Mean_TTFS_ms"].max()
            ax.plot([x_min / 1.08, x_max * 1.08], [y, y], color="#999999", linestyle=":",
                     linewidth=1.0, zorder=0)
            label_y = y
            if prev_label_y is not None and prev_label_y / label_y < MIN_LABEL_Y_RATIO:
                label_y = prev_label_y / MIN_LABEL_Y_RATIO
            prev_label_y = label_y
            # When a label had to be nudged away from its line's true y, draw a thin leader back
            # to it -- otherwise the text would float with no visible connection to its row.
            arrowprops = None
            if label_y != y:
                arrowprops = dict(arrowstyle="-", color="#bbbbbb", lw=0.7, shrinkA=2, shrinkB=2)
            ax.annotate(
                env_display_name(env), xy=(x_max, y), xycoords="data",
                xytext=(x_max * 1.18, label_y), textcoords="data",
                ha="left", va="center", fontsize=8.5, color="#555555", arrowprops=arrowprops,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        algo_handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BASE_COLORS[p],
                   markeredgecolor="black", markersize=9, label=BASE_DISPLAY[p])
            for p in OTHER_PLANNERS
        ]
        algo_handles.append(
            Line2D([0], [0], color=BASE_COLORS[KINOPAX_PLUS], linestyle="--", linewidth=1.4,
                   label="Kino-PAX+ (y = x)")
        )
        disc_handles = [
            Line2D([0], [0], marker=discretization_marker(d), linestyle="", markerfacecolor="#888888",
                   markeredgecolor="black", markersize=9, label=discretization_display(d))
            for d in discretization_labels
        ]

        legend1 = ax.legend(handles=algo_handles, title="Algorithm (color)", loc="upper left",
                             bbox_to_anchor=(1.02, 1.0), fontsize=9, title_fontsize=9, frameon=True)
        ax.add_artist(legend1)
        legend2 = ax.legend(handles=disc_handles, title="Discretization (shape)", loc="upper left",
                             bbox_to_anchor=(1.02, 0.62), fontsize=9, title_fontsize=9, frameon=True)
        legends = [legend1, legend2]

    ax.set_xlabel("Time to First Solution (ms) — other algorithms (Kino-PAX, SimpleCombo, Kino-PAX*)")
    ax.set_ylabel("Time to First Solution (ms) — Kino-PAX+")
    ax.set_title(
        f"Time to First Solution vs Kino-PAX+ — {MODEL_NAMES[model_id]} (m{model_id})\n"
        "(pooled across length & effort sweeps, mean of successful runs)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", bbox_extra_artists=legends)
    fig.savefig(os.path.splitext(out_path)[0] + ".svg", bbox_inches="tight", bbox_extra_artists=legends)
    plt.close(fig)


def main() -> None:
    if not os.path.isdir(DATASET_DIR):
        raise SystemExit(f"DATASET_DIR does not exist: {DATASET_DIR!r} -- edit it at the top of this file.")
    discretization_dirs = discover_discretization_dirs(DATASET_DIR)
    if not discretization_dirs:
        raise SystemExit(f"No 'discretization<LABEL>' folders found under {DATASET_DIR!r}.")
    discretization_labels = [discretization_label_from_dir(d) for d in discretization_dirs]

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Discretizations found: {discretization_labels}")

    any_data = False
    for model_id in MODEL_IDS:
        model_name = MODEL_NAMES[model_id]
        table = build_model_table(model_id, discretization_dirs)
        n_total_runs = int(table["N_Total"].sum())
        if n_total_runs == 0:
            print(f"  [m{model_id} {model_name}] no run files found -- skipping (folder likely still empty).")
        else:
            any_data = True

        csv_path = os.path.join(OUT_DIR, f"ttfs_ratio_m{model_id}_{sanitize_name(model_name)}.csv")
        table.to_csv(csv_path, index=False)
        png_path = os.path.join(OUT_DIR, f"ttfs_ratio_m{model_id}_{sanitize_name(model_name)}.png")
        plot_model(table, model_id, discretization_labels, png_path)
        print(f"  [m{model_id} {model_name}] wrote {csv_path}, {png_path} + .svg ({n_total_runs} runs pooled)")

    if not any_data:
        print(
            "\nNo run CSVs were found anywhere under this dataset -- the tables above are empty "
            "and the plots just say so. Drop the archived per-run CSVs into the environment "
            "folders and rerun this script."
        )


if __name__ == "__main__":
    main()
