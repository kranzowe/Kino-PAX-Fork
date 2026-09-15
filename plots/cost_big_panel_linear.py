"""LINEAR-SCALE variant of cost_big_panel.py -- same data, same six-cell (metric x model) layout,
same per-cell design (Kino-PAX+ on y, the other three algorithms on x, color = algorithm, shape =
discretization, y=x "as good as Kino-PAX+" line, red note for any discretization Kino-PAX+ itself
never solved) -- the axes are linear instead of log-log, tightly fit to each cell's own data
(min/max plus a small fixed margin) rather than snapped to a "nice"/power-of-10 bound, and there
are no per-point environment-name labels.

Both of those are deliberate departures from cost_big_panel.py, not oversights:
- Nice-bound snapping (the log version's tightest-enclosing-power-of-ten trick) makes sense on a
  log axis, where tick spacing is inherently uneven and a snapped edge lines up with a labeled
  tick. On a linear axis the "nice number" steps (1/2/2.5/5/10 x 10^k) are coarse enough that data
  topping out just past 0.5 rounds all the way up to 1.0 -- a big, arbitrary-looking jump that
  wastes axis space the data never uses. Fitting tightly to the data's own extent (see
  AXIS_MARGIN_FRACTION below) keeps every cell's frame matched to what's actually plotted.
- The env-name labels (with leader lines to a decluttered y-position) were carried over from the
  log version's design, but the label-decluttering logic measures available room in RENDERED
  space -- and once one cell's data clustered near the bottom of a tall, coarsely-snapped linear
  axis (Double Integrator / Control Effort, when this still had nice-bound snapping), there wasn't
  enough vertical room left for 9 labels and they collapsed into an unreadable pile. Rather than
  patch that one cell, the labels are dropped from this version entirely -- the color (algorithm)
  + shape (discretization) encoding is still there, just without a text tag identifying which
  environment each point came from.

AXES ARE STILL NOT SHARED ACROSS COLUMNS, for the same reason as cost_big_panel.py: final path
cost is not comparable across models (Quad's [0,100]^3 workspace vs. the other two's [0,1]^3), and
on a LINEAR scale that mismatch would be even more punishing than on log -- there's no log
compression left to keep a shared range from squashing the smaller models flat. Each panel keeps
auto-scaling to its own (model, metric) data.

Edit DATASET_DIR / OUT_DIR below to point at the dataset you want to plot, then run:
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
    COST_METRICS,
    COST_METRIC_LABELS,
    KINOPAX_PLUS,
    MODEL_IDS,
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
# DATASET_DIR must directly contain one or more discretization-level folders (tiny/fine/coarse,
# either bare or "discretization"-prefixed -- see zephyr_common.discover_discretization_dirs),
# each of which directly contains the empty/house/narrowPassage/zigzag subfolders.
#
# SHORT-TIMEOUT BRANCH: pointed at ZEPHYR_30_runs_SHORT (1M-node / 3s-timeout sweep) instead of the
# main ZEPHYR_30_runs dataset -- own output folder/filenames so the two never overwrite each other.
# ================================================================================================
DATASET_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs_SHORT")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "cost_ratio_short")

EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison

INCLUDE_SIMPLECOMBO = False  # <-- TOGGLE. Flip to True to bring SimpleCombo back into this panel.
PLOTTED_OTHER_PLANNERS = OTHER_PLANNERS if INCLUDE_SIMPLECOMBO else [p for p in OTHER_PLANNERS if p != SIMPLECOMBO]

PANEL_SUBTITLES = {
    1: "6D Double Integrator",
    2: "6D Dubins Airplane",
    3: "12D Nonlinear Drone",
}

# How much room to pad on every side of a cell's own tight data extent, as a fraction of that
# extent's span -- NOT a snap to any "nice" number, just breathing room so points don't sit flush
# against the axis frame or the y=x diagonal don't clip a corner marker.
AXIS_MARGIN_FRACTION = 0.08


def build_model_cost_table(model_id: int, metric: str, discretization_dirs: list[str]) -> pd.DataFrame:
    rows = []
    for disc_dir in discretization_dirs:
        discretization_label = discretization_label_from_dir(disc_dir)
        environments = [e for e in discover_environments(disc_dir) if e.lower() not in EXCLUDED_ENVIRONMENTS]
        for env in environments:
            env_dir = os.path.join(disc_dir, env)
            warn_on_unexpected_star_suffixes(env_dir)
            plus_stats = aggregate_final_cost(
                load_runs(env_dir, env, KINOPAX_PLUS, model_id, discretization_label, metrics=(metric,))
            )
            rows.append({
                "Discretization": discretization_label,
                "Environment": env,
                "Model": MODEL_NAMES[model_id],
                "CostMetric": metric,
                "Algorithm": BASE_DISPLAY[KINOPAX_PLUS],
                "N_Success": plus_stats.n_success,
                "N_Total": plus_stats.n_total,
                "Mean_Cost": plus_stats.mean,
                "Std_Cost": plus_stats.std,
                "KinoPaxPlus_Mean_Cost": plus_stats.mean,
                "Ratio_to_KinoPaxPlus": 1.0 if not math.isnan(plus_stats.mean) else math.nan,
            })
            for planner in PLOTTED_OTHER_PLANNERS:
                stats = aggregate_final_cost(
                    load_runs(env_dir, env, planner, model_id, discretization_label, metrics=(metric,))
                )
                ratio = math.nan
                if not math.isnan(stats.mean) and not math.isnan(plus_stats.mean) and plus_stats.mean > 0:
                    ratio = stats.mean / plus_stats.mean
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
                    "KinoPaxPlus_Mean_Cost": plus_stats.mean,
                    "Ratio_to_KinoPaxPlus": ratio,
                })
    return pd.DataFrame(rows)


def zero_success_discretizations(table: pd.DataFrame) -> list[tuple[str, int]]:
    """Discretizations where Kino-PAX+ itself found zero solutions across every environment --
    see cost_big_panel.py's module docstring for why that removes the WHOLE discretization from
    the scatter, not just Kino-PAX+'s own point. Returns [(discretization_label, total_runs), ...]."""
    kpp_rows = table[table["Algorithm"] == BASE_DISPLAY[KINOPAX_PLUS]]
    by_disc = kpp_rows.groupby("Discretization")[["N_Success", "N_Total"]].sum()
    zeroed = by_disc[(by_disc["N_Success"] == 0) & (by_disc["N_Total"] > 0)]
    return list(zeroed["N_Total"].items())


def plot_cell(ax, table: pd.DataFrame, model_id: int, metric: str, is_top_row: bool) -> None:
    """Draw one (model, metric) cell: tight-to-data linear axes, y=x reference line, and a plain
    color/shape-coded scatter -- no env-name labels (see module docstring)."""
    plotted = table[
        (table["Algorithm"] != BASE_DISPLAY[KINOPAX_PLUS])
        & table["Mean_Cost"].notna()
        & table["KinoPaxPlus_Mean_Cost"].notna()
    ]

    if plotted.empty:
        ax.text(0.5, 0.5, "No successful\nruns found yet.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666666")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        xs = plotted["Mean_Cost"].to_numpy()
        ys = plotted["KinoPaxPlus_Mean_Cost"].to_numpy()

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

        ax.plot([lo, hi], [lo, hi], linestyle="--", color=BASE_COLORS[KINOPAX_PLUS], linewidth=1.4, zorder=1)

        display_to_token = {v: k for k, v in BASE_DISPLAY.items()}
        for _, row in plotted.iterrows():
            planner_token = display_to_token[row["Algorithm"]]
            ax.scatter(
                row["Mean_Cost"], row["KinoPaxPlus_Mean_Cost"],
                s=110, marker=discretization_marker(row["Discretization"]),
                facecolors=BASE_COLORS[planner_token], edgecolors="black", linewidths=0.7,
                zorder=3,
            )

        style_linear_axis(ax)

    # Flag any discretization Kino-PAX+ never solved at all -- see zero_success_discretizations().
    zeroed = zero_success_discretizations(table)
    # if zeroed:
    #     note = "\n".join(
    #         f"{discretization_display(d)}: Kino-PAX+ 0/{n} successful runs" for d, n in zeroed
    #     )
    #     ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=6.5, color="#aa0000",
    #             ha="left", va="bottom")

    if is_top_row:
        ax.set_title(PANEL_SUBTITLES[model_id], fontsize=12, fontweight="bold")
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

    n_rows, n_cols = len(COST_METRICS), len(MODEL_IDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 9.3),
                              gridspec_kw={"hspace": 0.16, "wspace": 0.16})

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

    all_tables = []
    row_labels = []
    n_total_runs = 0
    drone_col = MODEL_IDS.index(3)
    for row, metric in enumerate(COST_METRICS):
        for col, model_id in enumerate(MODEL_IDS):
            table = build_model_cost_table(model_id, metric, discretization_dirs)
            all_tables.append(table)
            n_total_runs += int(table["N_Total"].sum())
            ax = axes[row][col]
            plot_cell(ax, table, model_id, metric, is_top_row=(row == 0))

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
        axes[row][0].set_ylabel("Final Cost — Kino-PAX+ (linear scale)", fontsize=10)
        pos_left = axes[row][0].get_position()
        y_mid = (pos_left.y0 + pos_left.y1) / 2.0
        row_labels.append(fig.text(0.012, y_mid, COST_METRIC_LABELS[metric], rotation=90,
                                    ha="center", va="center", fontsize=13, fontweight="bold"))

    supxlabel = fig.supxlabel("Final Cost — other algorithms (linear scale)", fontsize=11, x=0.53, y=0.015)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.08)

    base_name = "cost_ratio_big_panel_all_models_linear_short"
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
            "\nNo run CSVs were found anywhere under this dataset -- all six panels are empty "
            "placeholders. Drop the archived per-run CSVs into the environment folders and rerun "
            "this script."
        )


if __name__ == "__main__":
    main()
