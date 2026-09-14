"""Final-cost ratio scatter for the 6D Double Integrator (Model 1) only -- EXACT DUPLICATE of
cost_ratio_scatter_panel_m1.py with the per-row environment-name callouts ("House"/"Narrow"/
"Windows" + leader line) turned off. Kept as a separate file rather than a toggle in the original
so both versions can be generated side by side without re-running twice -- see that file for the
full design writeup (fixed hand-picked linear ranges, INCLUDE_SIMPLECOMBO, etc.), which all still
applies here unchanged. The shared-row dotted horizontal line and Kino-PAX+'s own marker on it
stay (they're not "environment labels", and still convey "these points share a y-reference");
only the text callout naming which environment that row is goes away.

Edit DATASET_DIR / OUT_DIR / METRIC_RANGES below to point at the dataset or adjust the fixed axis
ranges, then run:
    python plots/cost_ratio_scatter_panel_m1_no_env_labels.py
"""
from __future__ import annotations

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
    COST_METRICS,
    COST_METRIC_LABELS,
    KINOPAX_PLUS,
    MODEL_NAMES,
    OTHER_PLANNERS,
    SIMPLECOMBO,
    aggregate_final_cost,
    discover_discretization_dirs,
    discover_environments,
    discretization_display,
    discretization_label_from_dir,
    discretization_marker,
    load_runs,
    style_linear_axis,
    warn_on_unexpected_star_suffixes,
)

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================
# EDIT THESE to point at the dataset and output location you want.
# DATASET_DIR must directly contain one or more "discretization<LABEL>" folders (tiny/fine/coarse),
# each of which directly contains the empty/house/narrowPassage/zigzag subfolders.
# ================================================================================================
DATASET_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "cost_ratio")

INCLUDE_SIMPLECOMBO = False  # <-- TOGGLE. Flip to True to bring SimpleCombo back into this panel.
PLOTTED_OTHER_PLANNERS = OTHER_PLANNERS if INCLUDE_SIMPLECOMBO else [p for p in OTHER_PLANNERS if p != SIMPLECOMBO]

MODEL_ID = 1  # this panel is Model 1 (6D Double Integrator) only
EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison

# Fixed (lo, hi) per metric -- hand-picked to comfortably fit this model's actual data with a
# little headroom, not auto-computed the way every other script here does it.
METRIC_RANGES = {"length": (1.0, 3.0), "effort": (0.0, 0.8)}
# METRIC_RANGES = {"length": (1.0, 5.0), "effort": (3.0, 16.0)}


def build_cost_table(metric: str, discretization_dirs: list[str]) -> pd.DataFrame:
    rows = []
    for disc_dir in discretization_dirs:
        discretization_label = discretization_label_from_dir(disc_dir)
        environments = [e for e in discover_environments(disc_dir) if e.lower() not in EXCLUDED_ENVIRONMENTS]
        for env in environments:
            env_dir = os.path.join(disc_dir, env)
            warn_on_unexpected_star_suffixes(env_dir)
            plus_stats = aggregate_final_cost(
                load_runs(env_dir, env, KINOPAX_PLUS, MODEL_ID, discretization_label, metrics=(metric,))
            )
            rows.append({
                "Discretization": discretization_label,
                "Environment": env,
                "Algorithm": BASE_DISPLAY[KINOPAX_PLUS],
                "N_Success": plus_stats.n_success,
                "N_Total": plus_stats.n_total,
                "Mean_Cost": plus_stats.mean,
                "KinoPaxPlus_Mean_Cost": plus_stats.mean,
            })
            for planner in PLOTTED_OTHER_PLANNERS:
                stats = aggregate_final_cost(
                    load_runs(env_dir, env, planner, MODEL_ID, discretization_label, metrics=(metric,))
                )
                rows.append({
                    "Discretization": discretization_label,
                    "Environment": env,
                    "Algorithm": BASE_DISPLAY[planner],
                    "N_Success": stats.n_success,
                    "N_Total": stats.n_total,
                    "Mean_Cost": stats.mean,
                    "KinoPaxPlus_Mean_Cost": plus_stats.mean,
                })
    return pd.DataFrame(rows)


def plot_metric_panel(ax, table: pd.DataFrame, subtitle: str, lo: float, hi: float) -> None:
    plotted = table[
        (table["Algorithm"] != BASE_DISPLAY[KINOPAX_PLUS])
        & table["Mean_Cost"].notna()
        & table["KinoPaxPlus_Mean_Cost"].notna()
    ]

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    # X label duplicated per subplot rather than one shared label below both -- Y stays a
    # single shared label since it's the same "Final Cost -- Kino-PAX+" for both metrics. Names
    # the actually-plotted "other" algorithms so it doesn't misstate whichever are excluded by
    # INCLUDE_SIMPLECOMBO.
    other_algo_names = ", ".join(BASE_DISPLAY[p] for p in PLOTTED_OTHER_PLANNERS)
    ax.set_xlabel(f"Final Cost — other algorithms ({other_algo_names})", fontsize=10)

    if plotted.empty:
        ax.text(0.5, 0.5, "No successful runs found yet.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666666")
        ax.set_title(subtitle, fontsize=12, fontweight="bold")
        return

    ax.plot([lo, hi], [lo, hi], linestyle="--", color=BASE_COLORS[KINOPAX_PLUS], linewidth=1.4, zorder=1)

    display_to_token = {v: k for k, v in BASE_DISPLAY.items()}
    for _, row in plotted.iterrows():
        planner_token = display_to_token[row["Algorithm"]]
        ax.scatter(
            row["Mean_Cost"], row["KinoPaxPlus_Mean_Cost"],
            s=130, marker=discretization_marker(row["Discretization"]),
            facecolors=BASE_COLORS[planner_token], edgecolors="black", linewidths=0.8,
            zorder=3,
        )

    # Every algorithm compared within one (environment, discretization) cell shares that cell's
    # Kino-PAX+ mean as its y-value, so they land on one shared horizontal line -- draw that line
    # and mark Kino-PAX+'s own position on it (exactly where that row crosses y=x), same as the
    # labeled version, just with no env-name callout attached to it here.
    span = hi - lo
    line_pad = span * 0.015
    for (env, disc), group in plotted.groupby(["Environment", "Discretization"]):
        y = group["KinoPaxPlus_Mean_Cost"].iloc[0]
        x_min = group["Mean_Cost"].min()
        x_max = group["Mean_Cost"].max()
        ax.plot([x_min - line_pad, x_max + line_pad], [y, y], color="#999999", linestyle=":",
                 linewidth=1.0, zorder=0)
        ax.scatter(y, y, s=40, marker=discretization_marker(disc),
                   facecolors=BASE_COLORS[KINOPAX_PLUS], edgecolors="black", linewidths=0.8, zorder=6)

    style_linear_axis(ax)
    ax.set_title(subtitle, fontsize=12, fontweight="bold")
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

    tables = {metric: build_cost_table(metric, discretization_dirs) for metric in COST_METRICS}
    n_total_runs = sum(int(t["N_Total"].sum()) for t in tables.values())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), gridspec_kw={"wspace": 0.28})

    for ax, metric in zip(axes, COST_METRICS):
        lo, hi = METRIC_RANGES[metric]
        plot_metric_panel(ax, tables[metric], COST_METRIC_LABELS[metric], lo, hi)

    model_name = MODEL_NAMES[MODEL_ID]
    # X label is duplicated per subplot (set inside plot_metric_panel); Y stays a single shared
    # label since it's the same "Final Cost -- Kino-PAX+" for both metrics.
    supylabel = fig.supylabel("Final Cost — Kino-PAX+", fontsize=11, x=0.01)

    algo_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BASE_COLORS[p],
               markeredgecolor="black", markersize=9, label=BASE_DISPLAY[p])
        for p in PLOTTED_OTHER_PLANNERS
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

    # Both legends fit inside the Control Effort panel's top-left corner, side by side -- that
    # corner sits well above the y=x diagonal, so it's clear of data either way.
    effort_ax = axes[1]
    legend1 = effort_ax.legend(handles=algo_handles, title="Algorithm (color)", loc="upper left",
                                bbox_to_anchor=(0.02, 0.98), fontsize=9, title_fontsize=9, frameon=True)
    effort_ax.add_artist(legend1)
    legend2 = effort_ax.legend(handles=disc_handles, title="Discretization (shape)", loc="upper left",
                                bbox_to_anchor=(0.42, 0.98), fontsize=9, title_fontsize=9, frameon=True)
    legends = [legend1, legend2, supylabel]

    fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.12)

    base_name = f"cost_ratio_panel_m{MODEL_ID}_{model_name}_no_env_labels"
    csv_path = os.path.join(OUT_DIR, f"{base_name}.csv")
    pd.concat([t.assign(CostMetric=m) for m, t in tables.items()], ignore_index=True).to_csv(csv_path, index=False)
    png_path = os.path.join(OUT_DIR, f"{base_name}.png")
    svg_path = os.path.join(OUT_DIR, f"{base_name}.svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.05, bbox_extra_artists=legends)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05, bbox_extra_artists=legends)
    plt.close(fig)
    print(f"Wrote {csv_path}, {png_path} + .svg ({n_total_runs} runs)")

    if n_total_runs == 0:
        print(
            "\nNo run CSVs were found -- the panels above are empty placeholders. Drop the "
            "archived per-run CSVs into the environment folders and rerun this script."
        )


if __name__ == "__main__":
    main()
