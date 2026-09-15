"""Cost scatter, 4-SECOND-TRUNCATED, ALL SIX (cost metric x model) combinations in one figure:
two rows (Control Effort, Workspace Path Length) x three columns (one per model, left to right).
Reference planner is Kino-PAX+ -- every other algorithm (Kino-PAX, Kino-PAX#; SimpleCombo
excluded by default, see INCLUDE_SIMPLECOMBO) is plotted against it, color = algorithm, shape =
discretization, LINEAR (not log) tight-to-data axes, y=x "as good as Kino-PAX+" line, no
per-point environment-name labels.

Design lineage: direct adaptation of cost_big_panel_linear.py -- same linear-axis/no-env-label/
tight-fit/legend-on-drone-column design -- with three changes: (1) BOTH rows now read
plots/DATA/ZEPHYR_30_runs (that file's two rows pulled from two different datasets/metrics for an
unrelated reason -- see its own docstring); here row 1 = effort, row 2 = length, both from the
same dataset, (2) REFERENCE_PLANNER is Kino-PAX+ again (that file switched it to Kino-PAX for a
reason specific to its Elapsed-Time row, which doesn't apply here), and (3) no per-panel
model-name title, and axis label TEXT (not tick labels, already bold via style_linear_axis) is
bold.

TRUNCATION: cost is read at the SAME 4000ms checkpoint used throughout this dataset elsewhere in
the suite (zephyr_common.cost_at_time / aggregate_cost_at_time) -- "the best cost achieved by the
4s mark," not each run's own final cost. best_cost is monotonically non-increasing over
elapsed_time_ms (an anytime algorithm never un-finds a better solution), so the last logged row at
or before 4000ms IS that checkpoint's value, no interpolation needed.

AXES ARE NOT SHARED ACROSS COLUMNS (Quad's [0,100]^3 workspace vs. the other two's [0,1]^3) or
ROWS (control effort and path length are different units) -- same reasoning as every other
big-panel script in this suite.

Edit ROWS / TARGET_TIME_MS / OUT_DIR below, then run:
    python plots/cost_panel_vs_kinopaxplus_4s.py
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
    BASE_NAMES,
    COST_METRIC_LABELS,
    KINOPAX_PLUS,
    MODEL_IDS,
    MODEL_NAMES,
    SIMPLECOMBO,
    aggregate_cost_at_time,
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
# EDIT THESE to change the datasets, checkpoint time, or output location.
# ================================================================================================
ROWS = [
    {"label": "Control Effort", "dataset_dir": os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs"),
     "metric": "effort"},
    {"label": "Workspace Path Length", "dataset_dir": os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs"),
     "metric": "length"},
]
TARGET_TIME_MS = 4000.0
OUT_DIR = os.path.join(PLOTS_DIR, "output", "scatter_panels_4s")
INDIVIDUAL_OUT_DIR = os.path.join(OUT_DIR, "individual")

REFERENCE_PLANNER = KINOPAX_PLUS  # <-- Kino-PAX+.
EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison

INCLUDE_SIMPLECOMBO = False  # <-- TOGGLE. Flip to True to bring SimpleCombo back into this panel.
PLOTTED_OTHER_PLANNERS = [
    p for p in BASE_NAMES
    if p != REFERENCE_PLANNER and (INCLUDE_SIMPLECOMBO or p != SIMPLECOMBO)
]

# How much room to pad on every side of a cell's own tight data extent, as a fraction of that
# extent's span -- NOT a snap to any "nice" number, just breathing room so points don't sit flush
# against the axis frame or have the y=x diagonal clip a corner marker.
AXIS_MARGIN_FRACTION = 0.08


def build_model_cost_table(model_id: int, row_cfg: dict, discretization_dirs: list) -> pd.DataFrame:
    metric = row_cfg["metric"]
    rows = []
    for disc_dir in discretization_dirs:
        discretization_label = discretization_label_from_dir(disc_dir)
        environments = [e for e in discover_environments(disc_dir) if e.lower() not in EXCLUDED_ENVIRONMENTS]
        for env in environments:
            env_dir = os.path.join(disc_dir, env)
            warn_on_unexpected_star_suffixes(env_dir)
            ref_stats = aggregate_cost_at_time(
                load_runs(env_dir, env, REFERENCE_PLANNER, model_id, discretization_label, metrics=(metric,)),
                TARGET_TIME_MS,
            )
            rows.append({
                "Discretization": discretization_label,
                "Environment": env,
                "Model": MODEL_NAMES[model_id],
                "CostMetric": metric,
                "Algorithm": BASE_DISPLAY[REFERENCE_PLANNER],
                "N_Success": ref_stats.n_success,
                "N_Total": ref_stats.n_total,
                "Mean_Cost": ref_stats.mean,
                "Std_Cost": ref_stats.std,
                "Reference_Mean_Cost": ref_stats.mean,
                "Ratio_to_Reference": 1.0 if not math.isnan(ref_stats.mean) else math.nan,
            })
            for planner in PLOTTED_OTHER_PLANNERS:
                stats = aggregate_cost_at_time(
                    load_runs(env_dir, env, planner, model_id, discretization_label, metrics=(metric,)),
                    TARGET_TIME_MS,
                )
                ratio = math.nan
                if not math.isnan(stats.mean) and not math.isnan(ref_stats.mean) and ref_stats.mean > 0:
                    ratio = stats.mean / ref_stats.mean
                rows.append({
                    "Discretization": discretization_label,
                    "Environment": env,
                    "Model": MODEL_NAMES[model_id],
                    "CostMetric": metric,
                    "Algorithm": BASE_DISPLAY[planner],
                    "N_Success": stats.n_success,
                    "N_Total": stats.n_total,
                    "Mean_Cost": stats.mean,
                    "Std_Cost": stats.std,
                    "Reference_Mean_Cost": ref_stats.mean,
                    "Ratio_to_Reference": ratio,
                })
    return pd.DataFrame(rows)


def plot_cell(ax, table: pd.DataFrame) -> None:
    """Draw one (row, model) cell: tight-to-data linear axes, y=x reference line, plain
    color/shape-coded scatter -- no env-name labels, no per-panel title (dropped; see docstring)."""
    plotted = table[
        (table["Algorithm"] != BASE_DISPLAY[REFERENCE_PLANNER])
        & table["Mean_Cost"].notna()
        & table["Reference_Mean_Cost"].notna()
    ]

    if plotted.empty:
        ax.text(0.5, 0.5, "No successful\nruns found yet.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666666")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
        return

    xs = plotted["Mean_Cost"].to_numpy()
    ys = plotted["Reference_Mean_Cost"].to_numpy()

    # Tight fit to the data's own extent, plus a small fixed-fraction margin -- no snapping to a
    # "nice"/power-of-ten bound.
    data_lo = min(xs.min(), ys.min())
    data_hi = max(xs.max(), ys.max())
    span = data_hi - data_lo
    margin = span * AXIS_MARGIN_FRACTION if span > 0 else max(data_hi, 1.0) * AXIS_MARGIN_FRACTION
    lo = max(data_lo - margin, 0.0)
    hi = data_hi + margin

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.plot([lo, hi], [lo, hi], linestyle="--", color=BASE_COLORS[REFERENCE_PLANNER], linewidth=1.4, zorder=1)

    display_to_token = {v: k for k, v in BASE_DISPLAY.items()}
    for _, row in plotted.iterrows():
        planner_token = display_to_token[row["Algorithm"]]
        ax.scatter(
            row["Mean_Cost"], row["Reference_Mean_Cost"],
            s=110, marker=discretization_marker(row["Discretization"]),
            facecolors=BASE_COLORS[planner_token], edgecolors="black", linewidths=0.7,
            zorder=3,
        )

    # Every algorithm compared within one (environment, discretization) cell shares that cell's
    # REFERENCE_PLANNER mean as its y-value -- shared dotted guide line + REFERENCE_PLANNER's own
    # marker on it, exactly where that row crosses y=x. Line span always includes
    # REFERENCE_PLANNER's own x (== y, on the diagonal) so it never shrinks to a stub that doesn't
    # visibly reach the diagonal marker.
    span = hi - lo
    line_pad = span * 0.015
    for (_env, disc), group in plotted.groupby(["Environment", "Discretization"]):
        y = group["Reference_Mean_Cost"].iloc[0]
        x_min = min(group["Mean_Cost"].min(), y)
        x_max = max(group["Mean_Cost"].max(), y)
        ax.plot([x_min - line_pad, x_max + line_pad], [y, y], color="#999999", linestyle=":",
                 linewidth=1.0, zorder=0)
        ax.scatter(y, y, s=38, marker=discretization_marker(disc),
                   facecolors=BASE_COLORS[REFERENCE_PLANNER], edgecolors="black", linewidths=0.7, zorder=6)

    style_linear_axis(ax)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)


def save_individual_cell(table: pd.DataFrame, metric: str, model_id: int,
                          algo_handles: list, disc_handles: list) -> None:
    """Standalone single-panel version of one (row, model) cell -- same plot_cell() drawing, but
    with its own axis labels (bold, matching the combined panel's wording) and its own copy of
    both legends, since a standalone image can't rely on a shared row/figure label or a
    neighboring panel's legend the way a cell inside the combined 2x3 figure can."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    plot_cell(ax, table)

    ref_display = BASE_DISPLAY[REFERENCE_PLANNER]
    ax.set_xlabel("Other algorithms' Cost @ 4s (linear scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{ref_display} Cost @ 4s (linear scale)", fontsize=11, fontweight="bold")

    legend_kwargs = dict(fontsize=8, title_fontsize=8, markerscale=0.9,
                          handletextpad=0.35, borderpad=0.35, labelspacing=0.3, frameon=True)
    leg_a = ax.legend(handles=algo_handles, title="Algorithm (color)", loc="upper left",
                       bbox_to_anchor=(0.02, 0.98), **legend_kwargs)
    ax.add_artist(leg_a)
    ax.legend(handles=disc_handles, title="Discretization (shape)", loc="upper left",
              bbox_to_anchor=(0.02, 0.80), **legend_kwargs)

    base = f"cost_vs_kinopaxplus_4s_{metric}_{MODEL_NAMES[model_id]}"
    fig.savefig(os.path.join(INDIVIDUAL_OUT_DIR, f"{base}.png"), dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(INDIVIDUAL_OUT_DIR, f"{base}.svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    for row_cfg in ROWS:
        if not os.path.isdir(row_cfg["dataset_dir"]):
            raise SystemExit(f"Dataset does not exist: {row_cfg['dataset_dir']!r} -- edit ROWS at the "
                              "top of this file.")

    discretization_dirs_by_row = {}
    all_discretization_labels = set()
    for row_cfg in ROWS:
        dirs = discover_discretization_dirs(row_cfg["dataset_dir"])
        if not dirs:
            raise SystemExit(f"No discretization folders found under {row_cfg['dataset_dir']!r}.")
        discretization_dirs_by_row[row_cfg["label"]] = dirs
        all_discretization_labels.update(discretization_label_from_dir(d) for d in dirs)
        print(f"{row_cfg['label']}: {row_cfg['dataset_dir']} -- "
              f"{[discretization_label_from_dir(d) for d in dirs]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(INDIVIDUAL_OUT_DIR, exist_ok=True)

    n_rows, n_cols = len(ROWS), len(MODEL_IDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 4.5 * n_rows + 0.5), squeeze=False,
                              gridspec_kw={"hspace": 0.12, "wspace": 0.12})

    ref_display = BASE_DISPLAY[REFERENCE_PLANNER]
    algo_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BASE_COLORS[p],
               markeredgecolor="black", markersize=9, label=BASE_DISPLAY[p])
        for p in PLOTTED_OTHER_PLANNERS
    ]
    algo_handles.append(
        Line2D([0], [0], color=BASE_COLORS[REFERENCE_PLANNER], linestyle="--", linewidth=1.4,
               marker="o", markersize=6, markeredgecolor="black", label=f"{ref_display} (y = x)")
    )
    disc_handles = [
        Line2D([0], [0], marker=discretization_marker(d), linestyle="", markerfacecolor="#888888",
               markeredgecolor="black", markersize=9, label=discretization_display(d))
        for d in sorted(all_discretization_labels)
    ]

    all_tables = []
    row_labels = []
    n_total_runs = 0
    drone_col = MODEL_IDS.index(3)
    for row, row_cfg in enumerate(ROWS):
        discretization_dirs = discretization_dirs_by_row[row_cfg["label"]]
        for col, model_id in enumerate(MODEL_IDS):
            table = build_model_cost_table(model_id, row_cfg, discretization_dirs)
            all_tables.append(table)
            n_total_runs += int(table["N_Total"].sum())
            ax = axes[row][col]
            plot_cell(ax, table)
            save_individual_cell(table, row_cfg["metric"], model_id, algo_handles, disc_handles)

            if col == drone_col:
                # Both Drone cells get their own copy of both legends, top-left -- that corner
                # sits clear of this model's data (which clusters along the diagonal toward the
                # upper-right).
                legend_kwargs = dict(fontsize=6.5, title_fontsize=6.5, markerscale=0.75,
                                      handletextpad=0.35, borderpad=0.35, labelspacing=0.3,
                                      frameon=True)
                leg_a = ax.legend(handles=algo_handles, title="Algorithm (color)", loc="upper left",
                                   bbox_to_anchor=(0.02, 0.98), **legend_kwargs)
                ax.add_artist(leg_a)
                ax.legend(handles=disc_handles, title="Discretization (shape)", loc="upper left",
                          bbox_to_anchor=(0.37, 0.98), **legend_kwargs)

        # Row label -- the cost metric, rotated in the left margin -- read AFTER this row's three
        # cells are drawn so get_position() reflects their final (post-aspect-lock) layout.
        axes[row][0].set_ylabel(f"{ref_display} Cost @ 4s (linear scale)",
                                 fontsize=10, fontweight="bold")
        pos_left = axes[row][0].get_position()
        y_mid = (pos_left.y0 + pos_left.y1) / 2.0
        row_labels.append(fig.text(0.038, y_mid, row_cfg["label"], rotation=90,
                                    ha="center", va="center", fontsize=13, fontweight="bold"))

    supxlabel = fig.supxlabel("Other algorithms' Cost @ 4s (linear scale)",
                               fontsize=11, fontweight="bold", x=0.53, y=0.015)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.08)

    base_name = "cost_vs_kinopaxplus_4s"
    csv_path = os.path.join(OUT_DIR, f"{base_name}.csv")
    pd.concat(all_tables, ignore_index=True).to_csv(csv_path, index=False)
    png_path = os.path.join(OUT_DIR, f"{base_name}.png")
    svg_path = os.path.join(OUT_DIR, f"{base_name}.svg")
    legends = [supxlabel, *row_labels]
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.03, bbox_extra_artists=legends)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03, bbox_extra_artists=legends)
    plt.close(fig)
    print(f"Wrote {csv_path}, {png_path} + .svg ({n_total_runs} runs across all six panels)")
    print(f"Wrote 6 standalone per-cell PNG+SVG pairs to {INDIVIDUAL_OUT_DIR}")

    if n_total_runs == 0:
        print(
            "\nNo run CSVs were found anywhere under either dataset -- all six panels are empty "
            "placeholders. Drop the archived per-run CSVs into the environment folders and rerun "
            "this script."
        )


if __name__ == "__main__":
    main()
