"""LINEAR-SCALE cost ratio panel -- Kino-PAX (not Kino-PAX+!) on y, the other algorithms on x,
color = algorithm, shape = discretization, y=x "as good as Kino-PAX" line, tight-to-data linear
axes (min/max plus a small fixed margin, not snapped to a "nice"/power-of-10 bound), no per-point
environment-name labels.

TWO ROWS, TWO DIFFERENT DATASETS, BOTH READ AT THE SAME 4s CHECKPOINT (see ROWS / TARGET_TIME_MS
below), not each run's own final cost:
- Row 1, Control Effort: plots/DATA/ZEPHYR_30_runs, metric "effort". Those runs go all the way to
  ~10s, but the comparison here is deliberately the cost already achieved by the 4s mark (best_cost
  is monotonically non-increasing over elapsed_time_ms -- an anytime algorithm never un-finds a
  better solution -- so "the last logged row at or before 4000ms" IS that checkpoint's value, no
  interpolation needed; see zephyr_common.cost_at_time).
- Row 2, Elapsed Time: plots/DATA/ZEPHYR_30_RUNS_3M_4s_TIME, metric "time" (a genuinely different
  cost function -- it minimizes elapsed time itself -- not a relabeling of length/effort). These
  runs are already capped at 4s max, so "cost @ 4s" and "final cost" coincide here; it still goes
  through the same cost_at_time checkpoint lookup as row 1 for consistency (and correctness, in
  case a run's very last logged row lands a hair past 4000ms).

REFERENCE_PLANNER IS KINO-PAX, NOT KINO-PAX+ (unlike every earlier version of this panel): Kino-PAX+
solves the Elapsed-Time-objective sweep (row 2) so rarely that it would be a useless, mostly-"--"
denominator for that whole row. Rather than mix reference algorithms between rows in one figure
(confusing -- the two rows would mean different things on the same-looking y-axis), BOTH rows are
relative to Kino-PAX. Kino-PAX+ is simply one of the "other algorithms" plotted on the x-axis now.

NO ENV-NAME LABELS, DISCRETIZATION-SHAPE LEGEND IS THE UNION ACROSS BOTH ROWS' DATASETS (row 1's
ZEPHYR_30_runs has coarse/fine/tiny; row 2's ZEPHYR_30_RUNS_3M_4s_TIME only has tiny so far) -- and
AXES ARE NOT SHARED ACROSS COLUMNS (Quad's [0,100]^3 workspace vs. the other two's [0,1]^3) or
ROWS (control effort and elapsed-time are different units entirely) -- same reasoning as every
earlier version of this panel and cost_big_panel.py.

Every algorithm compared within one (environment, discretization) cell still shares Kino-PAX's own
checkpoint cost as its y-value, so they land on a shared horizontal line -- that dotted guide line
(and Kino-PAX's own small marker on it, exactly where the row crosses y=x) is drawn even without
the text label. The line's x-span always includes Kino-PAX's own x (which equals y, since it's on
the diagonal) even when only one other algorithm has a data point for that row -- otherwise the
line can shrink to a tiny stub around a single far-off point that never visually reaches back to
the diagonal marker, which can look like the line was simply missing.

Edit ROWS / TARGET_TIME_MS / OUT_DIR below to change the datasets, checkpoint time, or output
location, then run:
    python plots/cost_big_panel_linear.py
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
    KPAX,
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
# EDIT THESE to change the datasets, checkpoint time, or output location -- see module docstring
# for why each row reads a different dataset/metric.
# ================================================================================================
ROWS = [
    {"label": "Control Effort", "dataset_dir": os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs"),
     "metric": "effort"},
    {"label": "Elapsed Time", "dataset_dir": os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_RUNS_3M_4s_TIME"),
     "metric": "time"},
]
TARGET_TIME_MS = 4000.0
OUT_DIR = os.path.join(PLOTS_DIR, "output", "cost_ratio")

REFERENCE_PLANNER = KPAX  # <-- Kino-PAX, not Kino-PAX+ -- see module docstring.
EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison

INCLUDE_SIMPLECOMBO = False  # <-- TOGGLE. Flip to True to bring SimpleCombo back into this panel.
PLOTTED_OTHER_PLANNERS = [
    p for p in BASE_NAMES
    if p != REFERENCE_PLANNER and (INCLUDE_SIMPLECOMBO or p != SIMPLECOMBO)
]

PANEL_SUBTITLES = {
    1: "6D Double Integrator",
    2: "6D Dubins Airplane",
    3: "12D Nonlinear Drone",
}

# How much room to pad on every side of a cell's own tight data extent, as a fraction of that
# extent's span -- NOT a snap to any "nice" number, just breathing room so points don't sit flush
# against the axis frame or the y=x diagonal don't clip a corner marker.
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


def plot_cell(ax, table: pd.DataFrame, model_id: int, is_top_row: bool) -> None:
    """Draw one (model, row) cell: tight-to-data linear axes, y=x reference line, and a plain
    color/shape-coded scatter -- no env-name labels (see module docstring)."""
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
    else:
        xs = plotted["Mean_Cost"].to_numpy()
        ys = plotted["Reference_Mean_Cost"].to_numpy()

        # Tight fit to the data's own extent, plus a small fixed-fraction margin -- no snapping to
        # a "nice"/power-of-ten bound (see AXIS_MARGIN_FRACTION and the module docstring).
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

        # Every algorithm compared within one (environment, discretization) cell shares that
        # cell's Kino-PAX mean as its y-value -- draw that shared row explicitly as a dotted
        # guide line, with Kino-PAX's own position on it (exactly where the row crosses y=x)
        # marked in its own color. Line span always includes Kino-PAX's own x (== y, on the
        # diagonal) even when only one other algorithm has a point for this row, so it never
        # shrinks to a stub that doesn't visibly reach the diagonal marker.
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

    if is_top_row:
        ax.set_title(PANEL_SUBTITLES[model_id], fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)


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

    n_rows, n_cols = len(ROWS), len(MODEL_IDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 4.5 * n_rows + 0.5), squeeze=False,
                              gridspec_kw={"hspace": 0.12, "wspace": 0.12})

    algo_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BASE_COLORS[p],
               markeredgecolor="black", markersize=9, label=BASE_DISPLAY[p])
        for p in PLOTTED_OTHER_PLANNERS
    ]
    ref_display = BASE_DISPLAY[REFERENCE_PLANNER]
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
            plot_cell(ax, table, model_id, is_top_row=(row == 0))

            if col == drone_col:
                # Both 12D-Drone cells get their own copy of both legends, top-left -- that
                # corner sits clear of this model's data (which clusters along the diagonal
                # toward the upper-right) in both rows.
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
        axes[row][0].set_ylabel(f"{ref_display} Final Cost (linear scale)",
                                 fontsize=10)
        pos_left = axes[row][0].get_position()
        y_mid = (pos_left.y0 + pos_left.y1) / 2.0
        row_labels.append(fig.text(0.038, y_mid, row_cfg["label"], rotation=90,
                                    ha="center", va="center", fontsize=13, fontweight="bold"))

    supxlabel = fig.supxlabel(f"Kino-PAX+ and Kino-PAX# Final Cost (linear scale)",
                               fontsize=11, x=0.53, y=0.015)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.08)

    base_name = "cost_ratio_big_panel_all_models_linear"
    csv_path = os.path.join(OUT_DIR, f"{base_name}.csv")
    pd.concat(all_tables, ignore_index=True).to_csv(csv_path, index=False)
    png_path = os.path.join(OUT_DIR, f"{base_name}.png")
    svg_path = os.path.join(OUT_DIR, f"{base_name}.svg")
    legends = [supxlabel, *row_labels]
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.03, bbox_extra_artists=legends)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03, bbox_extra_artists=legends)
    plt.close(fig)
    print(f"Wrote {csv_path}, {png_path} + .svg ({n_total_runs} runs across all six panels)")

    if n_total_runs == 0:
        print(
            "\nNo run CSVs were found anywhere under either dataset -- all six panels are empty "
            "placeholders. Drop the archived per-run CSVs into the environment folders and rerun "
            "this script."
        )


if __name__ == "__main__":
    main()
