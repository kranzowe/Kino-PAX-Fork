"""Final-cost ratio scatter, ALL SIX (model x cost metric) combinations in one figure: two rows
(Workspace Path Length, Control Effort) x three columns (one per model, left to right). Exactly
the same data and same per-panel design as cost_ratio_scatter.py (Kino-PAX+ on y, the other three
algorithms on x, color = algorithm, shape = discretization, the shared-row/env-label treatment,
the y=x "as good as Kino-PAX+" line) -- this just lays all six of that script's separate figures
out as one combined panel instead.

AXES ARE NOT SHARED ACROSS COLUMNS, ON PURPOSE: unlike ttfs_ratio_scatter_panel.py (one shared log
range across its three subplots, because time-to-first-solution is the same unit/scale story for
every model), final path cost is NOT comparable across models -- Quad's (m3) costs run roughly two
orders of magnitude above Double Integrator's (m1) and Dubins Airplane's (m2), simply because its
workspace is scaled [0,100]^3 instead of [0,1]^3, not because it's actually doing worse. Forcing
one shared range across all six panels would squash the two smaller models into an unreadable
sliver in the corner of Quad's range. So each of the six panels auto-scales to its own
(model, metric) data instead, snapped to the TIGHTEST enclosing "nice" bound on each end (any
integer coefficient 1-9 times a power of 10, not all the way out to the next full decade) -- e.g.
data spanning 1.3-3.4 gets a 1-4 box, not 1-6 or 1-10. That boundary value is then explicitly
labeled (_edge_tick_label) and, if it doesn't land on one of the ticks style_log_axis's own minor
locator would already draw (2x/4x/6x/8x), a tick is added there too (_add_edge_ticks) so the
label has something to attach to. Column headers name the model (top row only); a
big rotated row label on the left names the metric, since there's no other way to tell which row
is which once cost magnitude alone can't do it (columns) and the model name already lives in the
header (rows). The Kino-PAX+ y-axis label is duplicated once per row (not one shared label for the
whole figure) so it can sit right next to that row's own tick labels instead of a fixed x-position
that a wide row of tick labels (e.g. "10^-2") can end up overlapping.

MISSING DISCRETIZATIONS ARE A REAL FINDING, NOT A BUG: every algorithm in a (environment,
discretization) cell shares Kino-PAX+'s own mean cost as its y-value (see build_model_cost_table),
so if Kino-PAX+ itself never found a solution at some discretization, NOTHING for that
discretization can be plotted in that cell at all -- there's no y to plot against. This is exactly
what happens for Coarse at (m2, Control Effort) and both metrics at (m3, Coarse): Kino-PAX+ has a
flat 0/90 success rate there (checked directly against the data), not a loading/labeling bug in
this script. Each affected cell gets a small red note listing which discretization(s) this hit and
their success rate, rather than just silently having fewer markers than its neighbors.

Edit DATASET_DIR / OUT_DIR below to point at the dataset you want to plot, then run:
    python plots/cost_big_panel.py
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
OUT_DIR = os.path.join(PLOTS_DIR, "output", "cost_ratio")

EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison

INCLUDE_SIMPLECOMBO = False  # <-- TOGGLE. Flip to True to bring SimpleCombo back into this panel.
PLOTTED_OTHER_PLANNERS = OTHER_PLANNERS if INCLUDE_SIMPLECOMBO else [p for p in OTHER_PLANNERS if p != SIMPLECOMBO]

PANEL_SUBTITLES = {
    1: "6D Double Integrator",
    2: "6D Dubins Airplane",
    3: "12D Nonlinear Drone",
}

# (model_id, metric) cells whose environment labels anchor to the LEFT of the data instead of the
# default right -- same idea as ttfs_ratio_scatter_panel.py's per-model label-side toggle, just
# picked by hand per cell here since there's no single rule that predicts which side has the room.
LEFT_LABEL_CELLS = {(1, "effort")}

# Sub-decade "nice" multipliers an axis edge is allowed to snap to -- any integer 1-9, not just
# the coefficients style_log_axis happens to draw a tick mark at (1x major; 2x/4x/6x/8x minor).
# An edge that lands on one of THOSE just gets a label added to the tick that's already there; an
# edge that lands on 3x/5x/7x/9x instead needs a tick added too (see _add_edge_ticks), since
# style_log_axis's own minor locator never draws one there.
NICE_COEFFS = tuple(range(1, 10))

# How far past the data's own extent, on the side the env-name labels sit on, to inflate BEFORE
# rounding out to a "nice" bound -- gives the label + leader line room to sit clear of the data
# point it's labeling instead of being squeezed right up against the axis edge.
LABEL_MARGIN = 1.6

# How far the env-name labels themselves sit from the data they're labeling, as a multiplicative
# offset in data-space past the row's own nearest edge (e.g. 1.35 = 35% further out) -- this is
# the knob to turn if a cell's labels look too close to (crowding the markers) or too far from
# (wasting space / straining the leader lines) their data. EDIT PER-CELL HERE: add or change a
# (model_id, metric) entry to override just that cell; anything not listed uses
# DEFAULT_LABEL_DISTANCE. Larger number = labels sit further away.
DEFAULT_LABEL_DISTANCE = 1.35
LABEL_DISTANCE = {
    (1, "length"): 1.2,
    (3, "length"): 1.2,
    (3, "effort"): 1.2,   # example: push Double Integrator/Control Effort's labels out further
}


def _nice_log_bound(value: float, round_up: bool) -> float:
    """Nearest of NICE_COEFFS x 10^k to `value`, rounded outward (up for an upper bound, down for
    a lower one) so the result never cuts off `value` itself."""
    exponent = math.floor(math.log10(value))
    candidates = sorted(c * (10.0 ** e) for e in (exponent - 1, exponent, exponent + 1) for c in NICE_COEFFS)
    if round_up:
        return next(v for v in candidates if v >= value * (1 - 1e-9))
    return next(v for v in reversed(candidates) if v <= value * (1 + 1e-9))


def _edge_tick_label(value: float, edges: tuple[float, float]) -> str:
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
    edge value snapped to 3x/5x/7x/9x (now that NICE_COEFFS allows any integer 1-9) needs a tick
    added at that exact spot, not just a label with nothing to attach to. Rebuilds each axis's
    minor ticks as "whatever the standard 2/4/6/8 locator would draw in [lo, hi], plus lo and hi
    themselves" -- a FixedLocator, but only for THIS subplot, so the shared style_log_axis helper
    (and every other script using it) is untouched."""
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


def zero_success_discretizations(table: pd.DataFrame) -> list[tuple[str, int]]:
    """Discretizations where Kino-PAX+ itself found zero solutions across every environment --
    see the module docstring for why that silently removes the WHOLE discretization from the
    scatter, not just Kino-PAX+'s own point. Returns [(discretization_label, total_runs), ...]."""
    kpp_rows = table[table["Algorithm"] == BASE_DISPLAY[KINOPAX_PLUS]]
    by_disc = kpp_rows.groupby("Discretization")[["N_Success", "N_Total"]].sum()
    zeroed = by_disc[(by_disc["N_Success"] == 0) & (by_disc["N_Total"] > 0)]
    return list(zeroed["N_Total"].items())


def plot_cell(ax, table: pd.DataFrame, model_id: int, metric: str, is_top_row: bool) -> None:
    """Draw one (model, metric) cell -- same design as cost_ratio_scatter.py's plot_model_cost,
    minus the per-cell legend and title (this figure carries shared legends and uses row/column
    labels instead of six repeated titles)."""
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
        label_left = (model_id, metric) in LEFT_LABEL_CELLS

        # Tightest "nice" bound enclosing the data on each end (see _nice_log_bound) rather than
        # rounding out to the next full decade -- e.g. data up to 4.2 gets an upper bound of 6,
        # not 10. That edge value gets its own explicit label (_edge_tick_label) since it usually
        # lands on what style_log_axis would otherwise leave as an unlabeled minor tick. The side
        # the env-name labels sit on (LEFT_LABEL_CELLS) gets extra headroom baked in BEFORE
        # rounding -- inflating by LABEL_MARGIN first and then rounding out to the next "nice"
        # value -- so there's room for that row's leader line + text without it overlapping the
        # data point it's labeling; the far side, which the labels never reach toward, stays as
        # tight as the data actually needs.
        if label_left:
            lo = _nice_log_bound(min(xs.min(), ys.min()) / LABEL_MARGIN, round_up=False)
            hi = _nice_log_bound(max(xs.max(), ys.max()), round_up=True)
        else:
            lo = _nice_log_bound(min(xs.min(), ys.min()), round_up=False)
            hi = _nice_log_bound(max(xs.max(), ys.max()) * LABEL_MARGIN, round_up=True)

        # Set the final scale/limits/aspect BEFORE anything below measures pixel positions
        # (declutter_label_ys needs ax.transData to already reflect the real rendered layout).
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

        # Same shared-row / declutter-label treatment as cost_ratio_scatter.py, except the label
        # column can sit to either side (LEFT_LABEL_CELLS) and at a per-cell-tunable distance
        # (LABEL_DISTANCE) -- and, now that the axis box is snapped tight to the data (not a
        # generous auto *1.5 margin), that offset can overshoot past the axis edge when the data
        # already sits close to it; clamped to stay a little inside [lo, hi] regardless.
        row_groups = sorted(
            plotted.groupby(["Environment", "Discretization"]),
            key=lambda item: item[1]["KinoPaxPlus_Mean_Cost"].iloc[0],
            reverse=True,
        )
        label_distance = LABEL_DISTANCE.get((model_id, metric), DEFAULT_LABEL_DISTANCE)
        if label_left:
            desired = min(group["Mean_Cost"].min() for _, group in row_groups) / label_distance
            label_x = max(desired, lo * 1.15)
            anchor_side, ha = "min", "right"
        else:
            desired = max(group["Mean_Cost"].max() for _, group in row_groups) * label_distance
            label_x = min(desired, hi / 1.15)
            anchor_side, ha = "max", "left"
        label_ys = declutter_label_ys(
            ax, [(key, group["KinoPaxPlus_Mean_Cost"].iloc[0]) for key, group in row_groups]
        )

        for (env, disc), group in row_groups:
            y = group["KinoPaxPlus_Mean_Cost"].iloc[0]
            x_min = group["Mean_Cost"].min()
            x_max = group["Mean_Cost"].max()
            anchor_x = x_max if anchor_side == "max" else x_min
            ax.plot([x_min / 1.08, x_max * 1.08], [y, y], color="#999999", linestyle=":",
                     linewidth=1.0, zorder=0)
            ax.scatter(y, y, s=38, marker=discretization_marker(disc),
                       facecolors=BASE_COLORS[KINOPAX_PLUS], edgecolors="black", linewidths=0.7, zorder=6)
            ax.annotate(
                env_display_name(env), xy=(anchor_x, y), xycoords="data",
                xytext=(label_x, label_ys[(env, disc)]), textcoords="data",
                ha=ha, va="center", fontsize=7.5, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6, shrinkA=2, shrinkB=2),
            )

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
        # fig.text() artists are NOT auto-included by savefig(bbox_inches="tight") -- same
        # clipping trap as fig.suptitle/supxlabel/supylabel elsewhere in this codebase -- so its
        # return value is collected into row_labels and passed via bbox_extra_artists below.
        # The y-axis label itself is a normal per-Axes ylabel (set below, on this row's leftmost
        # cell), NOT a figure-level artist -- bbox_inches="tight" already accounts for those
        # automatically, and doing it this way lets each row's label sit right next to that row's
        # own tick labels instead of a fixed figure-x position that a wide tick label (e.g.
        # "10^-2") could end up overlapping.
        axes[row][0].set_ylabel("Final Cost — Kino-PAX+ (log scale)", fontsize=10)
        pos_left = axes[row][0].get_position()
        y_mid = (pos_left.y0 + pos_left.y1) / 2.0
        row_labels.append(fig.text(0.012, y_mid, COST_METRIC_LABELS[metric], rotation=90,
                                    ha="center", va="center", fontsize=13, fontweight="bold"))

    supxlabel = fig.supxlabel("Final Cost — other algorithms (log scale)", fontsize=11, x=0.53, y=0.015)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.08)

    base_name = "cost_ratio_big_panel_all_models"
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
