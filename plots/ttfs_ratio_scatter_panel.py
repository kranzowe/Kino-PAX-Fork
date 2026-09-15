"""Time-to-first-solution ratio scatter: REFERENCE_PLANNER (y-axis, currently Kino-PAX) vs every
other planner (x-axis).

Same idea and same per-model data as ttfs_ratio_scatter.py, but all three models are laid out
side by side as one panel of subplots instead of three separate figures, one pair of legends
(Algorithm color, Discretization shape), and one pair of axis labels. No overall title -- meant to
be dropped into LaTeX, which adds its own caption. Each subplot gets the model as its own
subtitle. Color = algorithm, shape = discretization (only "tiny" exists today; "fine"/"coarse" are
picked up automatically once those sweeps are added alongside it). Every algorithm compared
within one (environment, discretization) cell shares REFERENCE_PLANNER's mean as its y-value, so
they land on a shared horizontal line, drawn as a dotted guide connecting them (see the "always
include the reference's own y" note in plot_model_panel -- this used to occasionally look dropped
when only one non-reference algorithm had a data point). The "empty" environment is excluded (not
interesting -- every planner solves it trivially). The y=x line marks "as fast as
REFERENCE_PLANNER", drawn in that planner's own color (currently Kino-PAX's near-black) so it
visually reads as "that algorithm's own line". See plots/zephyr_common.py for the on-disk naming
conventions this ports from scripts/process_paper_benchmark_improvement_scatter.m.

NO ENV-NAME LABELS, BY DEFAULT: this and every future panel plot in this folder default to leaving
off the per-row environment-name text/leader-lines unless specifically asked for -- they added a
lot of layout complexity (side-picking, decluttering, per-model distance tuning) for a payoff that
turned out not to be worth it once the plots had to also carry more rows (Jetson overlays,
multiple discretizations). The color/shape/position encoding still tells the whole story.

AXES ARE NOT SHARED ACROSS SUBPLOTS, in the same manner as cost_big_panel.py: each model's own
data (plus its Jetson point, if any) is fitted to the TIGHTEST enclosing "nice" bound on each end
(any integer coefficient 1-9 times a power of 10 -- see _nice_log_bound), not a decade-rounded or
fixed range. That boundary value gets its own explicit label (_edge_tick_label) since it usually
lands on what style_log_axis would otherwise leave as an unlabeled minor tick. Minor ticks are
drawn at every integer sub-decade coefficient (2-9), not just the even ones, per request -- denser
than style_log_axis's own shared default, so _add_edge_ticks rebuilds this subplot's own minor
locator rather than changing that shared helper (which other scripts still use un-modified).

REFERENCE_PLANNER (below) is the one knob that picks which algorithm the whole panel is plotted
relative to -- swap it to KINOPAX_PLUS/SIMPLECOMBO/KINOPAX_STAR to change which one owns the
y-axis and the y=x line's color; everything else (the "other algorithms" set, axis/legend text,
Jetson overlay) follows automatically.

JETSON OVERLAY (PROOF OF CONCEPT, see INCLUDE_JETSON below): each subplot gets ONE extra point per
environment -- Jetson's KinoPax*-only mean TTFS, plotted at that SAME environment's existing
Zephyr Fine row (JETSON_MATCHING_DISCRETIZATION), not as a row of its own. That is deliberate: the
point is "how does Jetson's hardware compare to Zephyr's GPU at the same discretization", so it
has to share that row's y (REFERENCE_PLANNER's Zephyr mean) rather than getting its own baseline
computed from Jetson's (much smaller, partially-failed for some model/env combinations) run set.
It's drawn with the SAME marker shape as Fine's own (a square) -- only its color (JETSON_COLOR)
sets it apart from a normal Zephyr point, so its legend entry lives with the other color-coded
entries in "Algorithm (color)" rather than "Discretization (shape)". Models 1/2 read Jetson's
discretizationFINE/FINE (tagged by model). Model 3 reads JETSON_20_runs/NEWQUAD_MEDIUM instead --
new hardware data at a genuinely different on-disk discretization ("medium" token) that, per
request, is treated as equivalent to Fine for comparison purposes here (same overlay treatment,
same color, matched onto the same Fine row) even though it isn't literally the same sweep.

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
from matplotlib.ticker import FuncFormatter
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import (  # noqa: E402
    BASE_COLORS,
    BASE_DISPLAY,
    BASE_NAMES,
    KINOPAX_STAR,
    KPAX,
    MODEL_IDS,
    SIMPLECOMBO,
    aggregate_ttfs,
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
JETSON_DIR = os.path.join(PLOTS_DIR, "DATA", "JETSON_20_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "ttfs_ratio")

REFERENCE_PLANNER = KPAX     # <-- Which algorithm the whole panel is plotted relative to (y-axis).
INCLUDE_JETSON = True        # <-- TOGGLE. Flip to False to go back to Zephyr-only, no other changes.
INCLUDE_SIMPLECOMBO = False  # <-- TOGGLE. Flip to True to bring SimpleCombo back into this panel.

EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison
JETSON_COLOR = "#fe6100"  # the only thing that sets a Jetson proof-of-concept point apart

# Which Zephyr discretization each model's Jetson proof-of-concept data matches -- see JETSON
# OVERLAY note in the module docstring. Models with no entry get no Jetson star at all.
JETSON_MATCHING_DISCRETIZATION = {1: "fine", 2: "fine", 3: "fine"}

# Where each model's Jetson KinoPax*-only overlay data actually lives on disk, and the
# discretization token load_runs needs to find it there. Models 1/2 share Zephyr's own Fine
# provenance (tagged pipeline, "fine" token); Model 3's NEWQUAD_MEDIUM sweep is a different
# on-disk discretization ("medium" token) treated as Fine's equivalent for this comparison, per
# request -- see the module docstring's JETSON OVERLAY note.
JETSON_SOURCE_DIR = {
    1: os.path.join(JETSON_DIR, "discretizationFINE", "FINE"),
    2: os.path.join(JETSON_DIR, "discretizationFINE", "FINE"),
    3: os.path.join(JETSON_DIR, "NEWQUAD_MEDIUM"),
}
JETSON_SOURCE_TOKEN = {1: "fine", 2: "fine", 3: "medium"}

# Sub-decade multipliers an axis edge is allowed to snap to -- any integer 1-9, not just the
# coefficients style_log_axis happens to draw a tick mark at. An edge that lands on one of THOSE
# just gets a label added to the tick that's already there; an edge that lands elsewhere needs a
# tick added too (see _add_edge_ticks), since the minor locator built there draws one at every
# integer 2-9 (denser than style_log_axis's own shared default, per request).
NICE_COEFFS = tuple(range(1, 10))

PLOTTED_OTHER_PLANNERS = [
    p for p in BASE_NAMES
    if p != REFERENCE_PLANNER and (INCLUDE_SIMPLECOMBO or p != SIMPLECOMBO)
]

# Exact model subtitles requested -- deliberately spelled out in full rather than reusing the
# shorter MODEL_LABELS from zephyr_common.py (those read e.g. "m2 -- Dubins Airplane 6D").
MODEL_NAMES_LOCAL = {1: "DoubleIntegrator", 2: "DubinsAirplane", 3: "Quad"}
PANEL_SUBTITLES = {
    1: "6D Double Integrator",
    2: "6D Dubins Airplane",
    3: "12D Nonlinear Drone",
}


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
    """Rebuilds each axis's minor ticks as "every integer sub-decade coefficient 2-9 in [lo, hi],
    plus lo and hi themselves" -- denser than style_log_axis's own shared minor locator (which
    only draws 2/4/6/8), and this subplot's own FixedLocator, so the shared helper (and every
    other script using it) is untouched. lo/hi always land on a tick this way even when they snap
    to a coefficient (e.g. 3x, 7x) the plain every-other-integer locator wouldn't have drawn."""
    from matplotlib.ticker import FixedLocator, LogLocator

    standard_minor = LogLocator(base=10, subs=tuple(range(2, 10)))
    edge_formatter = FuncFormatter(lambda v, _pos, edges=(lo, hi): _edge_tick_label(v, edges))
    for axis in (ax.xaxis, ax.yaxis):
        ticks = sorted({t for t in standard_minor.tick_values(lo, hi) if lo <= t <= hi} | {lo, hi})
        axis.set_minor_locator(FixedLocator(ticks))
        axis.set_minor_formatter(edge_formatter)


def jetson_star_x_by_env(model_id: int) -> dict:
    """Jetson's KinoPax*-only mean TTFS per environment -- see the module docstring's JETSON
    OVERLAY note for why only this one algorithm, and only as an overlay on top of the matching
    Zephyr row rather than a row of its own. {} if this model has no matching entry, no Jetson
    data, or the toggle is off."""
    if not INCLUDE_JETSON:
        return {}
    if JETSON_MATCHING_DISCRETIZATION.get(model_id) is None:
        return {}
    base_dir = JETSON_SOURCE_DIR.get(model_id)
    file_token = JETSON_SOURCE_TOKEN.get(model_id)
    if base_dir is None or not os.path.isdir(base_dir):
        return {}

    result = {}
    for env in discover_environments(base_dir):
        if env.lower() in EXCLUDED_ENVIRONMENTS:
            continue
        env_dir = os.path.join(base_dir, env)
        warn_on_unexpected_star_suffixes(env_dir)
        stats = aggregate_ttfs(load_runs(env_dir, env, KINOPAX_STAR, model_id, file_token))
        if not math.isnan(stats.mean):
            result[env] = stats.mean
    return result


def build_model_table(model_id: int, discretization_dirs: list[str]) -> pd.DataFrame:
    rows = []
    for disc_dir in discretization_dirs:
        discretization_label = discretization_label_from_dir(disc_dir)
        environments = [e for e in discover_environments(disc_dir) if e.lower() not in EXCLUDED_ENVIRONMENTS]
        for env in environments:
            env_dir = os.path.join(disc_dir, env)
            warn_on_unexpected_star_suffixes(env_dir)
            ref_stats = aggregate_ttfs(load_runs(env_dir, env, REFERENCE_PLANNER, model_id, discretization_label))
            rows.append({
                "Discretization": discretization_label,
                "Environment": env,
                "Model": MODEL_NAMES_LOCAL[model_id],
                "Algorithm": BASE_DISPLAY[REFERENCE_PLANNER],
                "N_Success": ref_stats.n_success,
                "N_Total": ref_stats.n_total,
                "Mean_TTFS_ms": ref_stats.mean,
                "Std_TTFS_ms": ref_stats.std,
                "Reference_Mean_TTFS_ms": ref_stats.mean,
                "Ratio_to_Reference": 1.0 if not math.isnan(ref_stats.mean) else math.nan,
            })
            for planner in PLOTTED_OTHER_PLANNERS:
                stats = aggregate_ttfs(load_runs(env_dir, env, planner, model_id, discretization_label))
                ratio = math.nan
                if not math.isnan(stats.mean) and not math.isnan(ref_stats.mean) and ref_stats.mean > 0:
                    ratio = stats.mean / ref_stats.mean
                rows.append({
                    "Discretization": discretization_label,
                    "Environment": env,
                    "Model": MODEL_NAMES_LOCAL[model_id],
                    "Algorithm": BASE_DISPLAY[planner],
                    "N_Success": stats.n_success,
                    "N_Total": stats.n_total,
                    "Mean_TTFS_ms": stats.mean,
                    "Std_TTFS_ms": stats.std,
                    "Reference_Mean_TTFS_ms": ref_stats.mean,
                    "Ratio_to_Reference": ratio,
                })
    return pd.DataFrame(rows)


def plot_model_panel(ax, table: pd.DataFrame, subtitle: str, model_id: int, jetson_star_by_env: dict) -> None:
    plotted = table[
        (table["Algorithm"] != BASE_DISPLAY[REFERENCE_PLANNER])
        & table["Mean_TTFS_ms"].notna()
        & table["Reference_Mean_TTFS_ms"].notna()
    ]

    if plotted.empty:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0, 10.0)
        ax.set_ylim(1.0, 10.0)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.5, 0.5, "No successful runs found yet.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666666")
    else:
        jetson_xs = list(jetson_star_by_env.values())
        non_empty = [s for s in [plotted["Mean_TTFS_ms"], plotted["Reference_Mean_TTFS_ms"], pd.Series(jetson_xs)]
                     if not s.empty]
        all_x = pd.concat(non_empty)
        data_min, data_max = all_x.min(), all_x.max()
        # Tightest "nice" bound enclosing the data on each end (see _nice_log_bound) -- no env
        # labels on this plot, so no extra headroom needs baking in on either side.
        lo = _nice_log_bound(data_min, round_up=False)
        hi = _nice_log_bound(data_max, round_up=True)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.plot([lo, hi], [lo, hi], linestyle="--", color=BASE_COLORS[REFERENCE_PLANNER], linewidth=1.4, zorder=1)

        # 5x-faster/5x-slower reference lines: y = FACTOR*x is "that algorithm finished 5x
        # faster than REFERENCE_PLANNER at that same absolute time"; y = x/FACTOR is the mirror
        # image, "REFERENCE_PLANNER finished 5x faster" (that algorithm 5x slower). Red so they
        # read as a matched pair of secondary references straddling the primary y=x line, not a
        # second thing competing for attention. Each is clipped to the visible box -- y=FACTOR*x
        # runs off the top once x > hi/FACTOR, and y = x/FACTOR runs off the left once x < lo*FACTOR.
        FACTOR = 5.0
        faster_end = hi / FACTOR
        if faster_end > lo:
            ax.plot([lo, faster_end], [lo * FACTOR, hi], linestyle=":",
                     color="red", linewidth=1.2, alpha=0.8, zorder=1)
        slower_start = lo * FACTOR
        if slower_start < hi:
            ax.plot([slower_start, hi], [lo, hi / FACTOR], linestyle=":",
                     color="red", linewidth=1.2, alpha=0.8, zorder=1)

        display_to_token = {v: k for k, v in BASE_DISPLAY.items()}
        for _, row in plotted.iterrows():
            planner_token = display_to_token[row["Algorithm"]]
            ax.scatter(
                row["Mean_TTFS_ms"], row["Reference_Mean_TTFS_ms"],
                s=130, marker=discretization_marker(row["Discretization"]),
                facecolors=BASE_COLORS[planner_token], edgecolors="black", linewidths=0.8,
                zorder=3,
            )

        # Every algorithm compared within one (environment, discretization) cell shares that
        # cell's REFERENCE_PLANNER mean as its y-value, so they land on one shared horizontal
        # line -- draw that line explicitly and mark REFERENCE_PLANNER's own position on it
        # (exactly where that row crosses y=x) with a small symbol in its color.
        row_groups = plotted.groupby(["Environment", "Discretization"])

        matching_disc = JETSON_MATCHING_DISCRETIZATION.get(model_id)
        for (env, disc), group in row_groups:
            y = group["Reference_Mean_TTFS_ms"].iloc[0]
            # Always include the reference's own x (which is y, since it sits on the y=x
            # diagonal) in the line's span -- otherwise, when only ONE non-reference algorithm has
            # a data point for this row, the line was just a tiny stub around that single far-off
            # point instead of visibly connecting back to the diagonal reference marker, which
            # could look like the line was simply missing.
            x_min = min(group["Mean_TTFS_ms"].min(), y)
            x_max = max(group["Mean_TTFS_ms"].max(), y)
            # Jetson's KinoPax*-only point lands on THIS row (same y) if this is the
            # discretization Jetson matches for this model -- widen the row's line span to
            # include it so the line visually connects to it, not just the Zephyr cluster. Same
            # marker shape as the discretization it's standing in for; only JETSON_COLOR sets it
            # apart from a normal point.
            jetson_x = jetson_star_by_env.get(env) if disc == matching_disc else None
            if jetson_x is not None:
                x_min, x_max = min(x_min, jetson_x), max(x_max, jetson_x)
            ax.plot([x_min / 1.08, x_max * 1.08], [y, y], color="#999999", linestyle=":",
                     linewidth=1.0, zorder=0)
            ax.scatter(y, y, s=40, marker=discretization_marker(disc),
                       facecolors=BASE_COLORS[REFERENCE_PLANNER], edgecolors="black", linewidths=0.8, zorder=6)
            if jetson_x is not None:
                ax.scatter(jetson_x, y, s=130, marker=discretization_marker(disc),
                           facecolors=JETSON_COLOR, edgecolors="black", linewidths=0.8, zorder=5)

        style_log_axis(ax)
        _add_edge_ticks(ax, lo, hi)

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
    print(f"Reference planner: {BASE_DISPLAY[REFERENCE_PLANNER]}")

    tables = [build_model_table(model_id, discretization_dirs) for model_id in MODEL_IDS]
    n_total_runs = sum(int(t["N_Total"].sum()) for t in tables)
    jetson_stars = {model_id: jetson_star_x_by_env(model_id) for model_id in MODEL_IDS}
    has_jetson = any(jetson_stars.values())

    # Axes are NOT shared across subplots -- each model gets its own tightest-fit range (see
    # plot_model_panel), same as cost_big_panel.py, so no sharex/sharey/label_outer here.
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), gridspec_kw={"wspace": 0.22})

    for ax, model_id, table in zip(axes, MODEL_IDS, tables):
        plot_model_panel(ax, table, PANEL_SUBTITLES[model_id], model_id, jetson_stars[model_id])

    # Y meaning (not numeric range) is identical across all three subplots -- one shared label
    # rather than duplicating it per subplot the way the X label already is. Reference planner's
    # name leads the label (not "... -- Kino-PAX") so it reads immediately as "this axis is
    # Kino-PAX's own time", not a general TTFS axis with the algorithm tacked on as an afterthought.
    ref_display = BASE_DISPLAY[REFERENCE_PLANNER]
    supylabel = fig.supylabel(f"{ref_display} — Time to First Solution (ms, log scale)", fontsize=11, x=0.01)

    algo_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BASE_COLORS[p],
               markeredgecolor="black", markersize=9, label=BASE_DISPLAY[p])
        for p in PLOTTED_OTHER_PLANNERS
    ]
    algo_handles.append(
        Line2D([0], [0], color=BASE_COLORS[REFERENCE_PLANNER], linestyle="--", linewidth=1.4,
               marker="o", markersize=6, markeredgecolor="black", label=f"{ref_display} (y = x)")
    )
    algo_handles.append(
        Line2D([0], [0], color="red", linestyle=":", linewidth=1.2, alpha=0.8,
               label=f"5x faster/slower than {ref_display}")
    )
    if has_jetson:
        # Shape isn't distinctive for Jetson points (they reuse whichever discretization's shape
        # they're standing in for), so this lives with the color-coded entries, not the shape ones.
        algo_handles.append(
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=JETSON_COLOR,
                   markeredgecolor="black", markersize=9, label="Jetson KinoPax* (proof of concept)")
        )
    disc_handles = [
        Line2D([0], [0], marker=discretization_marker(d), linestyle="", markerfacecolor="#888888",
               markeredgecolor="black", markersize=9, label=discretization_display(d))
        for d in discretization_labels
    ]

    # Both legends fit inside the Drone panel's bottom-right corner -- that panel's own data all
    # sits in the upper portion of its own range, leaving that corner empty.
    last_ax = axes[-1]
    legend2 = last_ax.legend(handles=disc_handles, title="Discretization (shape)", loc="lower right",
                              bbox_to_anchor=(0.99, 0.02), fontsize=9, title_fontsize=9, frameon=True)
    last_ax.add_artist(legend2)
    legend1 = last_ax.legend(handles=algo_handles, title="Algorithm (color)", loc="lower right",
                              bbox_to_anchor=(0.99, 0.22), fontsize=9, title_fontsize=9, frameon=True)
    legends = [legend1, legend2, supylabel]

    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.1)

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
