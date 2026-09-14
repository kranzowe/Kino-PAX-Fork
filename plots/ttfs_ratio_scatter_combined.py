"""Time-to-first-solution ratio scatter: Kino-PAX+ (y-axis) vs every other planner (x-axis).

Same idea as ttfs_ratio_scatter.py, but instead of faceting into separate figures, everything is
squeezed onto ONE scatter: only the House and Narrow Passage environments (ZigZag/empty excluded),
all three models, one point per (other-algorithm x model x environment) triple. Pooled across the
length and effort cost-metric sweeps, since the cost metric doesn't affect solve timing. The y=x
line marks "as fast as Kino-PAX+". See plots/zephyr_common.py for the on-disk naming conventions
this ports from scripts/process_paper_benchmark_improvement_scatter.m.

All algorithms compared against the same (model, environment) cell share Kino-PAX+'s mean TTFS
as their y-value, so they land on a shared horizontal line -- that line is labeled directly with
the environment name instead of adding a third marker channel (color=algorithm, shape=model
already cover the other two dimensions).

Edit DISCRETIZATION_DIR / OUT_DIR below to point at the dataset you want to plot, then run:
    python plots/ttfs_ratio_scatter_combined.py
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
    MODEL_LABELS,
    MODEL_MARKERS,
    MODEL_NAMES,
    OTHER_PLANNERS,
    aggregate_ttfs,
    declutter_label_ys,
    discover_environments,
    env_display_name,
    load_runs,
    sanitize_name,
    style_log_axis,
    warn_on_unexpected_star_suffixes,
)

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================
# EDIT THESE to point at the dataset/discretization folder and output location you want.
# DISCRETIZATION_DIR must directly contain the empty/house/narrowPassage/zigzag subfolders.
# ================================================================================================
DISCRETIZATION_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs", "discretizationTINY")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "ttfs_ratio")

INCLUDED_ENVIRONMENTS = {"house", "narrowpassage"}  # only these two go on the combined scatter


def build_combined_table(environments: list[str], discretization_label: str) -> pd.DataFrame:
    rows = []
    for env in environments:
        env_dir = os.path.join(DISCRETIZATION_DIR, env)
        warn_on_unexpected_star_suffixes(env_dir)
        for model_id in MODEL_IDS:
            plus_stats = aggregate_ttfs(load_runs(env_dir, env, KINOPAX_PLUS, model_id, discretization_label))
            rows.append({
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
                if (not math.isnan(stats.mean) and not math.isnan(plus_stats.mean)
                        and plus_stats.mean > 0):
                    ratio = stats.mean / plus_stats.mean
                rows.append({
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


def plot_combined(table: pd.DataFrame, environments: list[str], discretization_label: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    legends = []

    plotted = table[
        (table["Algorithm"] != BASE_DISPLAY[KINOPAX_PLUS])
        & table["Mean_TTFS_ms"].notna()
        & table["KinoPaxPlus_Mean_TTFS_ms"].notna()
    ]

    if plotted.empty:
        ax.text(0.5, 0.5, "No successful runs found yet.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#666666")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        xs = plotted["Mean_TTFS_ms"].to_numpy()
        ys = plotted["KinoPaxPlus_Mean_TTFS_ms"].to_numpy()
        lo = min(xs.min(), ys.min()) / 1.5
        hi = max(xs.max(), ys.max()) * 1.5

        # Set the final scale/limits/aspect BEFORE anything below measures pixel positions
        # (declutter_label_ys needs ax.transData to already reflect the real rendered layout).
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.plot([lo, hi], [lo, hi], linestyle="--", color=BASE_COLORS[KINOPAX_PLUS], linewidth=1.4, zorder=1)

        model_name_to_id = {v: k for k, v in MODEL_NAMES.items()}
        display_to_token = {v: k for k, v in BASE_DISPLAY.items()}
        for _, row in plotted.iterrows():
            model_id = model_name_to_id[row["Model"]]
            planner_token = display_to_token[row["Algorithm"]]
            ax.scatter(
                row["Mean_TTFS_ms"], row["KinoPaxPlus_Mean_TTFS_ms"],
                s=150, marker=MODEL_MARKERS[model_id],
                facecolors=BASE_COLORS[planner_token], edgecolors="black", linewidths=0.8,
                zorder=3,
            )

        # Every algorithm compared within one (environment, model) cell shares that cell's
        # Kino-PAX+ mean as its y-value, so they land on one shared horizontal line -- draw that
        # line explicitly, mark Kino-PAX+'s own position on it (exactly where that row crosses
        # y=x) with a small blue symbol, and label the row with the environment name instead of
        # adding a third marker channel (color=algorithm and shape=model already cover the other
        # two). All labels line up in one shared column rather than each sitting just past its
        # own row's rightmost point, so leader lines converge instead of crossing.
        row_groups = sorted(
            plotted.groupby(["Environment", "Model"]),
            key=lambda item: item[1]["KinoPaxPlus_Mean_TTFS_ms"].iloc[0],
            reverse=True,
        )
        label_x = max(group["Mean_TTFS_ms"].max() for _, group in row_groups) * 1.35
        label_ys = declutter_label_ys(
            ax, [(key, group["KinoPaxPlus_Mean_TTFS_ms"].iloc[0]) for key, group in row_groups]
        )

        for (env, model_name), group in row_groups:
            model_id = model_name_to_id[model_name]
            y = group["KinoPaxPlus_Mean_TTFS_ms"].iloc[0]
            x_min = group["Mean_TTFS_ms"].min()
            x_max = group["Mean_TTFS_ms"].max()
            ax.plot([x_min / 1.08, x_max * 1.08], [y, y], color="#999999", linestyle=":",
                     linewidth=1.0, zorder=0)
            ax.scatter(y, y, s=45, marker=MODEL_MARKERS[model_id],
                       facecolors=BASE_COLORS[KINOPAX_PLUS], edgecolors="black", linewidths=0.8, zorder=6)
            ax.annotate(
                env_display_name(env), xy=(x_max, y), xycoords="data",
                xytext=(label_x, label_ys[(env, model_name)]), textcoords="data",
                ha="left", va="center", fontsize=8.5, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.7, shrinkA=2, shrinkB=2),
            )

        style_log_axis(ax)

        algo_handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BASE_COLORS[p],
                   markeredgecolor="black", markersize=9, label=BASE_DISPLAY[p])
            for p in OTHER_PLANNERS
        ]
        algo_handles.append(
            Line2D([0], [0], color=BASE_COLORS[KINOPAX_PLUS], linestyle="--", linewidth=1.4,
                   marker="o", markersize=6, markeredgecolor="black", label="Kino-PAX+ (y = x)")
        )
        model_handles = [
            Line2D([0], [0], marker=MODEL_MARKERS[m], linestyle="", markerfacecolor="#888888",
                   markeredgecolor="black", markersize=9, label=MODEL_LABELS[m])
            for m in MODEL_IDS
        ]
        legend1 = ax.legend(handles=algo_handles, title="Algorithm (color)", loc="upper left",
                             bbox_to_anchor=(1.02, 1.0), fontsize=9, title_fontsize=9, frameon=True)
        ax.add_artist(legend1)
        legend2 = ax.legend(handles=model_handles, title="Model (shape)", loc="upper left",
                             bbox_to_anchor=(1.02, 0.62), fontsize=9, title_fontsize=9, frameon=True)
        legends = [legend1, legend2]

    ax.set_xlabel("Time to First Solution (ms) — other algorithms")
    ax.set_ylabel("Time to First Solution (ms) — Kino-PAX+")
    env_names = " & ".join(env_display_name(e) for e in environments)
    ax.set_title(
        f"Time to First Solution vs Kino-PAX+ — {env_names}, All Models\n"
        f"(discretization: {discretization_label}; pooled across length & effort sweeps, "
        "mean of successful runs)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", bbox_extra_artists=legends)
    fig.savefig(os.path.splitext(out_path)[0] + ".svg", bbox_inches="tight", bbox_extra_artists=legends)
    plt.close(fig)


def main() -> None:
    disc_dir = DISCRETIZATION_DIR
    if not os.path.isdir(disc_dir):
        raise SystemExit(f"DISCRETIZATION_DIR does not exist: {disc_dir!r} -- edit it at the top of this file.")
    discretization_label = os.path.basename(disc_dir.rstrip("/\\")).replace("discretization", "").replace(
        "Discretization", "").lower()
    all_environments = discover_environments(disc_dir)
    environments = [e for e in all_environments if e.lower() in INCLUDED_ENVIRONMENTS]
    if not environments:
        raise SystemExit(
            f"None of {sorted(INCLUDED_ENVIRONMENTS)} were found as subfolders under {disc_dir!r} "
            f"(found: {all_environments})."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Data source: {disc_dir}  (discretization label: {discretization_label!r})")
    print(f"Environments: {environments}")

    table = build_combined_table(environments, discretization_label)
    n_total_runs = int(table["N_Total"].sum())

    csv_path = os.path.join(OUT_DIR, "ttfs_ratio_combined_house_narrowPassage.csv")
    table.to_csv(csv_path, index=False)
    png_path = os.path.join(OUT_DIR, "ttfs_ratio_combined_house_narrowPassage.png")
    plot_combined(table, environments, discretization_label, png_path)
    print(f"Wrote {csv_path} and {png_path} ({n_total_runs} runs pooled)")

    if n_total_runs == 0:
        print(
            "\nNo run CSVs were found -- the table above is empty and the plot just says so. "
            "Drop the archived per-run CSVs into the environment folders and rerun this script."
        )


if __name__ == "__main__":
    main()
