"""Final-cost ratio scatter, ALL SIX (model x cost metric) combinations in one figure -- EXACT
DUPLICATE of cost_big_panel.py with the per-row environment-name callouts ("House"/"Narrow"/
"Windows" + leader line) turned off. Kept as a separate file rather than a toggle in the original
so both versions can be generated side by side without re-running twice -- see that file for the
full design writeup (per-cell tight axis fitting, the zero-success red note, row/column labels,
etc.), which all still applies here unchanged. The shared-row dotted horizontal line and
Kino-PAX+'s own marker on it stay (they're not "environment labels", and still convey "these
points share a y-reference"); only the text callout naming which environment that row is goes
away. Without a label to leave room for, each cell is now fit symmetrically tight on both ends --
no LABEL_MARGIN/LEFT_LABEL_CELLS/LABEL_DISTANCE concept needed here.

Edit DATASET_DIR / OUT_DIR below to point at the dataset you want to plot, then run:
    python plots/cost_big_panel_no_env_labels.py
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
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
OUT_DIR = os.path.join(PLOTS_DIR, "output", "cost_ratio")

EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison

INCLUDE_SIMPLECOMBO = False  # <-- TOGGLE. Flip to True to bring SimpleCombo back into this panel.
PLOTTED_OTHER_PLANNERS = OTHER_PLANNERS if INCLUDE_SIMPLECOMBO else [p for p in OTHER_PLANNERS if p != SIMPLECOMBO]

PANEL_SUBTITLES = {
    1: "6D Double Integrator",
    2: "6D Dubins Airplane",
    3: "12D Nonlinear Drone",
}

# Sub-decade "nice" multipliers an axis edge is allowed to snap to -- any integer 1-9, not just
# the coefficients style_log_axis happens to draw a tick mark at (1x major; 2x/4x/6x/8x minor).
# An edge that lands on one of THOSE just gets a label added to the tick that's already there; an
# edge that lands on 3x/5x/7x/9x instead needs a tick added too (see _add_edge_ticks), since
# style_log_axis's own minor locator never draws one there.
NICE_COEFFS = tuple(range(1, 10))


def _nice_log_bound(value: float, round_up: bool) -> float:
    """Nearest of NICE_COEFFS x 10^k to `value`, rounded outward (up for an upper bound, down for
    a lower one) so the result never cuts off `value` itself."""
    exponent = math.floor(math.log10(value))
    candidates = sorted(c * (10.0 ** e) for e in (exponent - 1, exponent, exponent + 1) for c in NICE_COEFFS)
    if round_up:
        return next(v for v in candidates if v >= value * (1 - 1e-9))
    return next(v for v in reversed(candidates) if v <= value * (1 + 1e-9))


def _edge_tick_label(value: float, edges: tuple) -> str:
    """Label for an axis-edge tick -- "$8\\times10^{2}$" style, or just the bare coefficient at
    the 10^0 decade (e.g. "6", not the odd-looking "6x10^0")."""
    for edge in edges:
        if edge > 0 and abs(value - edge) < edge * 1e-6:
            exponent = math.floor(math.log10(value) + 1e-9)
            coeff = int(round(value / (10 ** exponent)))
            if exponent == 0:
                return rf"$\mathdefault{{{coeff}}}$"
            return rf"$\mathdefault{{{coeff}\times10^{{{exponent}}}}}$"
    return ""


def _add_edge_ticks(ax, lo: float, hi: float) -> None:
    """style_log_axis's minor locator only ever draws a tick at 2x/4x/6x/8x each decade -- an
    edge value snapped to 3x/5x/7x/9x (NICE_COEFFS allows any integer 1-9) needs a tick added at
    that exact spot, not just a label with nothing to attach to. Rebuilds each axis's minor ticks
    as "whatever the standard 2/4/6/8 locator would draw in [lo, hi], plus lo and hi themselves" --
    a FixedLocator, but only for THIS subplot, so the shared style_log_axis helper (and every
    other script using it) is untouched."""
    from matplotlib.ticker import FixedLocator, LogLocator

    standard_minor = LogLocator(base=10, subs=(2.0, 4.0, 6.0, 8.0))
    edge_formatter = FuncFormatter(lambda v, _pos, edges=(lo, hi): _edge_tick_label(v, edges))
    for axis in (ax.xaxis, ax.yaxis):
        ticks = sorted({t for t in standard_minor.tick_values(lo, hi) if lo <= t <= hi} | {lo, hi})
        axis.set_minor_locator(FixedLocator(ticks))
        axis.set_minor_formatter(edge_formatter)


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


def zero_success_discretizations(table: pd.DataFrame) -> list:
    """Discretizations where Kino-PAX+ itself found zero solutions across every environment --
    see cost_big_panel.py's module docstring for why that silently removes the WHOLE
    discretization from the scatter, not just Kino-PAX+'s own point. Returns
    [(discretization_label, total_runs), ...]."""
    kpp_rows = table[table["Algorithm"] == BASE_DISPLAY[KINOPAX_PLUS]]
    by_disc = kpp_rows.groupby("Discretization")[["N_Success", "N_Total"]].sum()
    zeroed = by_disc[(by_disc["N_Success"] == 0) & (by_disc["N_Total"] > 0)]
    return list(zeroed["N_Total"].items())


def plot_cell(ax, table: pd.DataFrame, model_id: int, is_top_row: bool) -> None:
    """Draw one (model, metric) cell -- same design as cost_big_panel.py's plot_cell, minus the
    per-row environment-name callout."""
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
        # Tightest "nice" bound enclosing the data on each end -- symmetric on both sides, since
        # there's no label to leave headroom for on either one (contrast cost_big_panel.py, where
        # the label side gets extra margin baked in first).
        lo = _nice_log_bound(min(xs.min(), ys.min()), round_up=False)
        hi = _nice_log_bound(max(xs.max(), ys.max()), round_up=True)

        ax.set_xscale("log")
        ax.set_yscale("log")
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

        # Every algorithm compared within one (environment, discretization) cell shares that
        # cell's Kino-PAX+ mean as its y-value, so they land on one shared horizontal line --
        # draw that line and mark Kino-PAX+'s own position on it (exactly where that row crosses
        # y=x), same as the labeled version, just with no env-name callout attached to it here.
        for (env, disc), group in plotted.groupby(["Environment", "Discretization"]):
            y = group["KinoPaxPlus_Mean_Cost"].iloc[0]
            x_min = group["Mean_Cost"].min()
            x_max = group["Mean_Cost"].max()
            ax.plot([x_min / 1.08, x_max * 1.08], [y, y], color="#999999", linestyle=":",
                     linewidth=1.0, zorder=0)
            ax.scatter(y, y, s=38, marker=discretization_marker(disc),
                       facecolors=BASE_COLORS[KINOPAX_PLUS], edgecolors="black", linewidths=0.7, zorder=6)

        style_log_axis(ax)
        _add_edge_ticks(ax, lo, hi)

    # Flag any discretization Kino-PAX+ never solved at all -- see zero_success_discretizations().
    zeroed = zero_success_discretizations(table)
    if zeroed:
        note = "\n".join(
            f"{discretization_display(d)}: Kino-PAX+ 0/{n} successful runs" for d, n in zeroed
        )
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=6.5, color="#aa0000",
                ha="left", va="bottom")

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
        # fig.text() artists are NOT auto-included by savefig(bbox_inches="tight") -- same
        # clipping trap as fig.suptitle/supxlabel/supylabel elsewhere in this codebase -- so its
        # return value is collected into row_labels and passed via bbox_extra_artists below.
        axes[row][0].set_ylabel("Final Cost — Kino-PAX+ (log scale)", fontsize=10)
        pos_left = axes[row][0].get_position()
        y_mid = (pos_left.y0 + pos_left.y1) / 2.0
        row_labels.append(fig.text(0.012, y_mid, COST_METRIC_LABELS[metric], rotation=90,
                                    ha="center", va="center", fontsize=13, fontweight="bold"))

    supxlabel = fig.supxlabel("Final Cost — other algorithms (log scale)", fontsize=11, x=0.53, y=0.015)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.08)

    base_name = "cost_ratio_big_panel_all_models_no_env_labels"
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
