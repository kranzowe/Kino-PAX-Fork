"""Time-to-first-solution ratio scatter: Kino-PAX+ (y-axis) vs every other planner (x-axis).

Same idea and same per-model data as ttfs_ratio_scatter.py, but all three models are laid out
side by side as one panel of subplots instead of three separate figures, sharing one x/y axis
range, one pair of legends (Algorithm color, Discretization shape), and one pair of axis labels.
No overall title -- meant to be dropped into LaTeX, which adds its own caption. Each subplot gets
the model as its own subtitle. Color = algorithm, shape = discretization (only "tiny" exists
today; "fine"/"coarse" are picked up automatically once those sweeps are added alongside it).
Every algorithm compared
within one (environment, discretization) cell shares Kino-PAX+'s mean as its y-value, so they land
on a shared horizontal line -- that line is labeled directly with the environment name. The
"empty" environment is excluded (not interesting -- every planner solves it trivially). The y=x
line marks "as fast as Kino-PAX+". See plots/zephyr_common.py for the on-disk naming conventions
this ports from scripts/process_paper_benchmark_improvement_scatter.m.

Edit DATASET_DIR / OUT_DIR below to point at the dataset you want to plot, then run:
    python plots/ttfs_ratio_scatter_panel.py
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
    OTHER_PLANNERS,
    aggregate_ttfs,
    declutter_label_ys,
    discover_discretization_dirs,
    discover_environments,
    discretization_display,
    discretization_label_from_dir,
    discretization_marker,
    env_display_name,
    load_runs,
    style_log_axis,
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

# Exact model subtitles requested -- deliberately spelled out in full rather than reusing the
# shorter MODEL_LABELS from zephyr_common.py (those read e.g. "m2 -- Dubins Airplane 6D").
MODEL_NAMES_LOCAL = {1: "DoubleIntegrator", 2: "DubinsAirplane", 3: "Quad"}
PANEL_SUBTITLES = {
    1: "6D Double Integrator",
    2: "6D Dubins Airplane",
    3: "12D Nonlinear Drone",
}


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
                "Model": MODEL_NAMES_LOCAL[model_id],
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
                    "Model": MODEL_NAMES_LOCAL[model_id],
                    "Algorithm": BASE_DISPLAY[planner],
                    "N_Success": stats.n_success,
                    "N_Total": stats.n_total,
                    "Mean_TTFS_ms": stats.mean,
                    "Std_TTFS_ms": stats.std,
                    "KinoPaxPlus_Mean_TTFS_ms": plus_stats.mean,
                    "Ratio_to_KinoPaxPlus": ratio,
                })
    return pd.DataFrame(rows)


def plot_model_panel(ax, table: pd.DataFrame, subtitle: str, lo: float, hi: float, model_id: int) -> None:
    plotted = table[
        (table["Algorithm"] != BASE_DISPLAY[KINOPAX_PLUS])
        & table["Mean_TTFS_ms"].notna()
        & table["KinoPaxPlus_Mean_TTFS_ms"].notna()
    ]

    # Set the final scale/limits/aspect BEFORE anything below measures pixel positions
    # (declutter_label_ys needs ax.transData to already reflect the real rendered layout).
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    if plotted.empty:
        ax.text(0.5, 0.5, "No successful runs found yet.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666666")
    else:
        ax.plot([lo, hi], [lo, hi], linestyle="--", color=BASE_COLORS[KINOPAX_PLUS], linewidth=1.4, zorder=1)

        display_to_token = {v: k for k, v in BASE_DISPLAY.items()}
        for _, row in plotted.iterrows():
            planner_token = display_to_token[row["Algorithm"]]
            ax.scatter(
                row["Mean_TTFS_ms"], row["KinoPaxPlus_Mean_TTFS_ms"],
                s=130, marker=discretization_marker(row["Discretization"]),
                facecolors=BASE_COLORS[planner_token], edgecolors="black", linewidths=0.8,
                zorder=3,
            )

        # Every algorithm compared within one (environment, discretization) cell shares that
        # cell's Kino-PAX+ mean as its y-value, so they land on one shared horizontal line --
        # draw that line explicitly, mark Kino-PAX+'s own position on it (exactly where that row
        # crosses y=x) with a small blue symbol, and label the row with the environment name. All
        # of one subplot's labels line up in one shared column (past that subplot's own widest
        # row) rather than each sitting just past its own row's rightmost point -- with many rows
        # (3 environments x 3 discretizations), per-row label columns produced crossing leader
        # lines. Rows can still land at nearly the same y (e.g. two
        # environments of similar difficulty), which would print two labels on top of each other
        # -- nudge later labels down (in sorted order) to keep a minimum multiplicative gap; the
        # guide line and Kino-PAX+ marker always stay at the true y.
        row_groups = sorted(
            plotted.groupby(["Environment", "Discretization"]),
            key=lambda item: item[1]["KinoPaxPlus_Mean_TTFS_ms"].iloc[0],
            reverse=True,
        )
        # Based on THIS subplot's own rows, not the shared hi -- with axes shared across panels
        # of very different scale (Drone vs. Double Integrator), a fraction of the shared hi left
        # virtually no room for label text on whichever subplot's own data sits closest to it.
        # Double Integrator's own data sits well clear of the shared hi, so its labels read fine
        # to the right of the data (nudged a bit further out); the other two panels' data crowds
        # much closer to hi, leaving little room on the right but plenty on the left instead.
        if model_id == 1:
            label_x = max(group["Mean_TTFS_ms"].max() for _, group in row_groups) * 1.7
            anchor_side, ha = "max", "left"
        else:
            label_x = min(group["Mean_TTFS_ms"].min() for _, group in row_groups) / 1.45
            anchor_side, ha = "min", "right"
        label_ys = declutter_label_ys(
            ax, [(key, group["KinoPaxPlus_Mean_TTFS_ms"].iloc[0]) for key, group in row_groups]
        )

        for (env, disc), group in row_groups:
            y = group["KinoPaxPlus_Mean_TTFS_ms"].iloc[0]
            x_min = group["Mean_TTFS_ms"].min()
            x_max = group["Mean_TTFS_ms"].max()
            anchor_x = x_max if anchor_side == "max" else x_min
            ax.plot([x_min / 1.08, x_max * 1.08], [y, y], color="#999999", linestyle=":",
                     linewidth=1.0, zorder=0)
            ax.scatter(y, y, s=40, marker=discretization_marker(disc),
                       facecolors=BASE_COLORS[KINOPAX_PLUS], edgecolors="black", linewidths=0.8, zorder=6)
            ax.annotate(
                env_display_name(env), xy=(anchor_x, y), xycoords="data",
                xytext=(label_x, label_ys[(env, disc)]), textcoords="data",
                ha=ha, va="center", fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.7, shrinkA=2, shrinkB=2),
            )

    style_log_axis(ax)
    ax.set_title(subtitle, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time to First Solution (ms, log scale) — other algorithms", fontsize=10)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)


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

    tables = [build_model_table(model_id, discretization_dirs) for model_id in MODEL_IDS]
    n_total_runs = sum(int(t["N_Total"].sum()) for t in tables)

    # Shared axis range across all three panels, computed from every model's data combined, so
    # absolute magnitudes are directly comparable panel to panel (not just each panel's own ratio).
    combined = pd.concat(tables, ignore_index=True)
    plotted_all = combined[
        (combined["Algorithm"] != BASE_DISPLAY[KINOPAX_PLUS])
        & combined["Mean_TTFS_ms"].notna()
        & combined["KinoPaxPlus_Mean_TTFS_ms"].notna()
    ]
    if not plotted_all.empty:
        all_vals = pd.concat([plotted_all["Mean_TTFS_ms"], plotted_all["KinoPaxPlus_Mean_TTFS_ms"]])
        lo, hi = all_vals.min() / 1.5, all_vals.max() * 1.5
    else:
        lo, hi = 1.0, 10.0

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.0), sharex=True, sharey=True,
                              gridspec_kw={"wspace": 0.06})

    for ax, model_id, table in zip(axes, MODEL_IDS, tables):
        plot_model_panel(ax, table, PANEL_SUBTITLES[model_id], lo, hi, model_id)
        ax.label_outer()

    # X label is duplicated per subplot (set inside plot_model_panel) rather than one shared
    # label below all three -- Y stays a single shared label since it's not per-model here.
    supylabel = fig.supylabel("Time to First Solution (ms, log scale) — Kino-PAX+", fontsize=11, x=0.02)

    algo_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BASE_COLORS[p],
               markeredgecolor="black", markersize=9, label=BASE_DISPLAY[p])
        for p in OTHER_PLANNERS
    ]
    algo_handles.append(
        Line2D([0], [0], color=BASE_COLORS[KINOPAX_PLUS], linestyle="--", linewidth=1.4,
               marker="o", markersize=6, markeredgecolor="black", label="Kino-PAX+ (y = x)")
    )
    disc_handles = [
        Line2D([0], [0], marker=discretization_marker(d), linestyle="", markerfacecolor="#888888",
               markeredgecolor="black", markersize=9, label=discretization_display(d))
        for d in discretization_labels
    ]

    # Both legends fit inside the Drone panel's bottom-right corner -- that panel's own data all
    # sits in the upper portion of the shared range, leaving that corner empty.
    last_ax = axes[-1]
    legend2 = last_ax.legend(handles=disc_handles, title="Discretization (shape)", loc="lower right",
                              bbox_to_anchor=(0.99, 0.02), fontsize=9, title_fontsize=9, frameon=True)
    last_ax.add_artist(legend2)
    legend1 = last_ax.legend(handles=algo_handles, title="Algorithm (color)", loc="lower right",
                              bbox_to_anchor=(0.99, 0.22), fontsize=9, title_fontsize=9, frameon=True)
    legends = [legend1, legend2, supylabel]

    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.1)

    csv_path = os.path.join(OUT_DIR, "ttfs_ratio_panel_all_models.csv")
    pd.concat(tables, ignore_index=True).to_csv(csv_path, index=False)
    png_path = os.path.join(OUT_DIR, "ttfs_ratio_panel_all_models.png")
    svg_path = os.path.join(OUT_DIR, "ttfs_ratio_panel_all_models.svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.05, bbox_extra_artists=legends)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05, bbox_extra_artists=legends)
    plt.close(fig)
    print(f"Wrote {csv_path}, {png_path} + .svg ({n_total_runs} runs pooled across all models)")

    if n_total_runs == 0:
        print(
            "\nNo run CSVs were found anywhere under this dataset -- the panels above are empty "
            "placeholders. Drop the archived per-run CSVs into the environment folders and rerun "
            "this script."
        )


if __name__ == "__main__":
    main()
