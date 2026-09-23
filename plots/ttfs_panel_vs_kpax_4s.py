"""Time-to-first-solution scatter, 4-SECOND-TRUNCATED, ALL SIX (cost metric x model) combinations
in one figure: two rows (Control Effort, Workspace Path Length) x three columns (one per model,
left to right). Reference planner is Kino-PAX (KPAX) -- every OTHER algorithm not listed in
EXCLUDED_ALGORITHMS (Kino-PAX+, SimpleCombo, Kino-PAX#, by default all included) is plotted
against it, color = algorithm, shape = discretization, log-log axes, y=x "as fast as Kino-PAX"
line, no per-point environment-name labels.

EXCLUDED_ALGORITHMS (below) is the single toggle for which algorithms appear anywhere in this
figure -- the combined 6-panel image, every individual saved cell, and the CSV. Add a planner
token to it to drop that algorithm entirely; REFERENCE_PLANNER (KPAX) can't be added here, since
it's the plot's fixed y-axis baseline, not one of "the other algorithms." Excluding Kino-PAX#
(KINOPAX_STAR) also automatically drops its Jetson-hardware overlay (see jetson_star_x_by_env),
since that overlay is specifically Kino-PAX# run on real hardware.

Design lineage: this is a (metric x model) extension of ttfs_ratio_scatter_panel_no_env_labels.py
(same REFERENCE_PLANNER/no-env-label/5x-faster-slower-line/Jetson-overlay design, grouped exactly
as in ttfs_ratio_scatter_panel.py), with these changes: (1) cost metric becomes a second panel
axis (that file pools length+effort into one set of 3 panels; this one keeps them separate, 2x3),
(2) TTFS is truncated to a 4s budget (see TIMEOUT_MS / zephyr_common.aggregate_ttfs_truncated)
rather than using each run's true, unbounded first-solution time, (3) no per-panel model-name
title, and axis label TEXT (not tick labels, which were already bold via style_log_axis) is bold,
and (4) NO LEGEND ANYWHERE in this figure (combined or individual) -- the figures are now
composed into LaTeX, and the only legend kept across the whole TTFS/Cost pair lives on the two
Cost-vs-Kino-PAX+ Quad panels (see cost_panel_vs_kinopaxplus_4s.py). Every individual saved cell
carries its OWN x AND y axis labels (not just DoubleIntegrator) for the same LaTeX-composition
reason -- each saved image has to stand alone.

TRUNCATION, NOT A DIFFERENT DATASET: uses plots/DATA/ZEPHYR_30_runs (which was not run with a 4s
cap), but treats any run whose real first-solution time exceeds 4000ms as unsolved for THIS
figure's purposes (zephyr_common.aggregate_ttfs_truncated) -- i.e. "how would these runs have
looked under a 4s budget," not each run's own, possibly much longer, true TTFS.

AXES ARE NOT SHARED ACROSS COLUMNS OR ROWS: each (metric, model) panel auto-scales to its own
data, snapped to the tightest enclosing "nice" bound (see _nice_log_bound/_add_edge_ticks) --
same reasoning as every other big-panel script in this suite (Quad's dynamics differ enough from
the other two models that a shared range would squash them).

Edit DATASET_DIR / TIMEOUT_MS / OUT_DIR below, then run:
    python plots/ttfs_panel_vs_kpax_4s.py
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import (  # noqa: E402
    BASE_COLORS,
    BASE_DISPLAY,
    BASE_NAMES,
    COST_METRIC_LABELS,
    KINOPAX_PLUS,
    KINOPAX_STAR,
    KPAX,
    MODEL_IDS,
    MODEL_NAMES,
    SIMPLECOMBO,
    aggregate_ttfs,
    aggregate_ttfs_truncated,
    discover_discretization_dirs,
    discover_environments,
    discretization_label_from_dir,
    discretization_marker,
    load_runs,
    style_log_axis,
    warn_on_unexpected_star_suffixes,
)

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================
# EDIT THESE to point at the dataset/output location you want, or change the truncation budget.
# ================================================================================================
DATASET_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "scatter_panels_4s")
INDIVIDUAL_OUT_DIR = os.path.join(OUT_DIR, "individual")
TIMEOUT_MS = 4000.0

# Row order matches how the combinations were requested: Control Effort first, then Path Length.
COST_METRIC_ROWS = ("effort", "length")

REFERENCE_PLANNER = KPAX    # <-- Kino-PAX. Every OTHER algorithm is plotted relative to it.

# <-- TOGGLE which OTHER algorithms get plotted against REFERENCE_PLANNER. Add any of
# KINOPAX_PLUS / SIMPLECOMBO / KINOPAX_STAR to exclude it from every panel (combined figure,
# individual cells, CSV) -- e.g. {SIMPLECOMBO} or {SIMPLECOMBO, KINOPAX_STAR}. REFERENCE_PLANNER
# itself doesn't belong in this set -- it's the fixed y-axis baseline, not one of "the others."
EXCLUDED_ALGORITHMS = set()

EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison

PLOTTED_OTHER_PLANNERS = [
    p for p in BASE_NAMES
    if p != REFERENCE_PLANNER and p not in EXCLUDED_ALGORITHMS
]

# Jetson-hardware overlay (KinoPax* proof-of-concept), same grouping as
# ttfs_ratio_scatter_panel.py -- see jetson_star_x_by_env() below.
JETSON_DIR = os.path.join(PLOTS_DIR, "DATA", "JETSON_20_runs")
INCLUDE_JETSON_MANUAL = True  # <-- TOGGLE. Independent on/off for the Jetson overlay specifically.
# Also auto-disabled if Kino-PAX# (KINOPAX_STAR) is excluded above, since the overlay IS Kino-PAX#
# run on real hardware -- excluding the algorithm from the comparison shouldn't leave its hardware
# point plotted with nothing to compare it against.
INCLUDE_JETSON = INCLUDE_JETSON_MANUAL and KINOPAX_STAR not in EXCLUDED_ALGORITHMS
JETSON_COLOR = "#fe6100"  # the only thing that sets a Jetson point apart from a normal one

# Which Zephyr discretization Jetson's data actually matches, per model. Models with no entry get
# no Jetson point at all.
JETSON_MATCHING_DISCRETIZATION = {1: "fine", 2: "fine", 3: "fine"}

# Sub-decade "nice" multipliers an axis edge is allowed to snap to -- any integer 1-9, not just
# the coefficients style_log_axis happens to draw a tick mark at (1x major; 2x/4x/6x/8x minor).
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
    that exact spot, not just a label with nothing to attach to."""
    from matplotlib.ticker import FixedLocator, LogLocator

    standard_minor = LogLocator(base=10, subs=(2.0, 4.0, 6.0, 8.0))
    edge_formatter = FuncFormatter(lambda v, _pos, edges=(lo, hi): _edge_tick_label(v, edges))
    for axis in (ax.xaxis, ax.yaxis):
        ticks = sorted({t for t in standard_minor.tick_values(lo, hi) if lo <= t <= hi} | {lo, hi})
        axis.set_minor_locator(FixedLocator(ticks))
        axis.set_minor_formatter(edge_formatter)


def jetson_star_x_by_env(model_id: int) -> dict:
    """Jetson's KinoPax*-only mean TTFS per environment -- see ttfs_ratio_scatter_panel.py's
    module docstring for the full rationale. {} if this model has no Jetson data, no matching
    discretization, or the toggle is off. NOT truncated to TIMEOUT_MS -- this is a small,
    separate proof-of-concept hardware comparison, not part of the ZEPHYR_30_runs 4s-truncation
    story.

    Model 3 (Quad) reads NEWQUAD_MEDIUM ("medium" token, ~42K regions) via the normal tagged
    load_runs pipeline -- this replaced an older, untagged discretizationCOARSE (~8K regions)
    attempt (see success_rate_table.py's module docstring, which made the same switch first)."""
    if not INCLUDE_JETSON:
        return {}
    matching_disc = JETSON_MATCHING_DISCRETIZATION.get(model_id)
    if matching_disc is None:
        return {}
    if model_id in (1, 2):
        base_dir = os.path.join(JETSON_DIR, "discretizationFINE", "FINE")
        token = "fine"
    elif model_id == 3:
        base_dir = os.path.join(JETSON_DIR, "NEWQUAD_MEDIUM")
        token = "medium"
    else:
        return {}
    if not os.path.isdir(base_dir):
        return {}

    result = {}
    for env in discover_environments(base_dir):
        if env.lower() in EXCLUDED_ENVIRONMENTS:
            continue
        env_dir = os.path.join(base_dir, env)
        warn_on_unexpected_star_suffixes(env_dir)
        stats = aggregate_ttfs(load_runs(env_dir, env, KINOPAX_STAR, model_id, token))
        if not math.isnan(stats.mean):
            result[env] = stats.mean
    return result


def build_model_table(model_id: int, metric: str, discretization_dirs: list) -> pd.DataFrame:
    rows = []
    for disc_dir in discretization_dirs:
        discretization_label = discretization_label_from_dir(disc_dir)
        environments = [e for e in discover_environments(disc_dir) if e.lower() not in EXCLUDED_ENVIRONMENTS]
        for env in environments:
            env_dir = os.path.join(disc_dir, env)
            warn_on_unexpected_star_suffixes(env_dir)
            ref_stats = aggregate_ttfs_truncated(
                load_runs(env_dir, env, REFERENCE_PLANNER, model_id, discretization_label, metrics=(metric,)),
                TIMEOUT_MS,
            )
            rows.append({
                "Discretization": discretization_label,
                "Environment": env,
                "Model": MODEL_NAMES[model_id],
                "CostMetric": metric,
                "Algorithm": BASE_DISPLAY[REFERENCE_PLANNER],
                "N_Success": ref_stats.n_success,
                "N_Total": ref_stats.n_total,
                "Mean_TTFS_ms": ref_stats.mean,
                "Std_TTFS_ms": ref_stats.std,
                "Reference_Mean_TTFS_ms": ref_stats.mean,
                "Ratio_to_Reference": 1.0 if not math.isnan(ref_stats.mean) else math.nan,
            })
            for planner in PLOTTED_OTHER_PLANNERS:
                stats = aggregate_ttfs_truncated(
                    load_runs(env_dir, env, planner, model_id, discretization_label, metrics=(metric,)),
                    TIMEOUT_MS,
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
                    "Mean_TTFS_ms": stats.mean,
                    "Std_TTFS_ms": stats.std,
                    "Reference_Mean_TTFS_ms": ref_stats.mean,
                    "Ratio_to_Reference": ratio,
                })
    return pd.DataFrame(rows)


def plot_cell(ax, table: pd.DataFrame, model_id: int, jetson_star_by_env: dict) -> None:
    """Draw one (metric, model) cell -- KPAX-referenced log-log scatter, no env-name labels, no
    per-panel title (dropped -- see module docstring), plus the Jetson-hardware overlay (same
    per-model/per-environment grouping as ttfs_ratio_scatter_panel.py).

    Grouping is driven by ref_rows (REFERENCE_PLANNER's own rows, always present in `table`
    regardless of EXCLUDED_ALGORITHMS -- build_model_table adds them unconditionally), NOT by
    `plotted` (the OTHER algorithms' rows): with everything in EXCLUDED_ALGORITHMS, `plotted` is
    empty even though REFERENCE_PLANNER (and Jetson) still have real data to show -- iterating
    ref_rows instead means the y=x line and REFERENCE_PLANNER's own points (and Jetson's, if
    enabled) still render with no other algorithms plotted, rather than the whole cell going blank."""
    ref_display = BASE_DISPLAY[REFERENCE_PLANNER]
    ref_rows = table[(table["Algorithm"] == ref_display) & table["Mean_TTFS_ms"].notna()]
    plotted = table[
        (table["Algorithm"] != ref_display)
        & table["Mean_TTFS_ms"].notna()
        & table["Reference_Mean_TTFS_ms"].notna()
    ]

    if ref_rows.empty:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0, 10.0)
        ax.set_ylim(1.0, 10.0)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.5, 0.5, "No successful runs found yet.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666666")
        return

    jetson_xs = list(jetson_star_by_env.values())
    other_vals = list(plotted["Mean_TTFS_ms"]) if not plotted.empty else []
    all_x = pd.concat([ref_rows["Mean_TTFS_ms"], pd.Series(other_vals), pd.Series(jetson_xs)])
    lo = _nice_log_bound(all_x.min(), round_up=False)
    hi = _nice_log_bound(all_x.max(), round_up=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.plot([lo, hi], [lo, hi], linestyle="--", color=BASE_COLORS[REFERENCE_PLANNER], linewidth=1.4, zorder=1)

    # 5x-faster/5x-slower reference lines straddling the primary y=x line -- see
    # ttfs_ratio_scatter_panel_no_env_labels.py for the full rationale, unchanged here.
    FACTOR = 5.0
    faster_end = hi / FACTOR
    if faster_end > lo:
        ax.plot([lo, faster_end], [lo * FACTOR, hi], linestyle=":",
                 color="red", linewidth=1.2, alpha=0.8, zorder=1)
    slower_start = lo * FACTOR
    if slower_start < hi:
        ax.plot([slower_start, hi], [lo, hi / FACTOR], linestyle=":",
                 color="red", linewidth=1.2, alpha=0.8, zorder=1)

    if not plotted.empty:
        display_to_token = {v: k for k, v in BASE_DISPLAY.items()}
        for _, row in plotted.iterrows():
            planner_token = display_to_token[row["Algorithm"]]
            ax.scatter(
                row["Mean_TTFS_ms"], row["Reference_Mean_TTFS_ms"],
                s=130, marker=discretization_marker(row["Discretization"]),
                facecolors=BASE_COLORS[planner_token], edgecolors="black", linewidths=0.8,
                zorder=3,
            )

    # Every algorithm compared within one (environment, discretization) cell shares that cell's
    # REFERENCE_PLANNER mean as its y-value -- shared dotted guide line + REFERENCE_PLANNER's own
    # marker on it, exactly where that row crosses y=x. A Jetson point (if this env/disc matches
    # JETSON_MATCHING_DISCRETIZATION for this model) rides the same shared line. With no OTHER
    # algorithms plotted (empty `group` below), the guide line just collapses to a short tick at
    # REFERENCE_PLANNER's (and Jetson's, if present) own value instead of vanishing entirely.
    matching_disc = JETSON_MATCHING_DISCRETIZATION.get(model_id)
    for _, ref_row in ref_rows.iterrows():
        env, disc = ref_row["Environment"], ref_row["Discretization"]
        y = ref_row["Mean_TTFS_ms"]
        group = plotted[(plotted["Environment"] == env) & (plotted["Discretization"] == disc)]
        jetson_x = jetson_star_by_env.get(env) if disc == matching_disc else None
        if not group.empty:
            x_min = group["Mean_TTFS_ms"].min()
            x_max = group["Mean_TTFS_ms"].max()
        else:
            x_min = x_max = y
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
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)


def save_individual_cell(table: pd.DataFrame, metric: str, model_id: int, jetson_star_by_env: dict) -> None:
    """Standalone single-panel version of one (metric, model) cell -- same plot_cell() drawing.
    Every cell gets its own x AND y axis labels now (the figures are composed in LaTeX, so each
    saved image has to stand alone). No legend anywhere in this figure -- the only legend kept
    across the TTFS/Cost pair now lives on the two Cost-vs-Kino-PAX+ Quad panels (see
    cost_panel_vs_kinopaxplus_4s.py)."""
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    plot_cell(ax, table, model_id, jetson_star_by_env)

    ref_display = BASE_DISPLAY[REFERENCE_PLANNER]
    label_artists = [
        ax.set_xlabel("TTFS of All Algorithms (ms, log scale)",
                      fontsize=15, fontweight="bold"),
        ax.set_ylabel("TTFS of Kino-PAX (ms, log scale)",
                      fontsize=15, fontweight="bold"),
    ]

    # Every cell now carries both a long bold x-label AND y-label -- without an explicit layout
    # pass, the aspect-locked axes box (set_aspect("equal") in plot_cell) can leave too little
    # margin in the bottom-left corner for both, and the two labels visually collide there.
    # tight_layout() recomputes the subplot margins to fit everything actually on the figure.
    fig.tight_layout()

    # bbox_inches="tight" alone under-crops large bold axis-label text (clipped a hair short at
    # the edge otherwise) -- passing the label artists explicitly through bbox_extra_artists
    # forces the tight-bbox calculation to include their full extent.
    base = f"ttfs_vs_kpax_4s_{metric}_{MODEL_NAMES[model_id]}"
    fig.savefig(os.path.join(INDIVIDUAL_OUT_DIR, f"{base}.png"), dpi=200, bbox_inches="tight",
                pad_inches=0.08, bbox_extra_artists=label_artists)
    fig.savefig(os.path.join(INDIVIDUAL_OUT_DIR, f"{base}.svg"), bbox_inches="tight",
                pad_inches=0.08, bbox_extra_artists=label_artists)
    plt.close(fig)


def main() -> None:
    if not os.path.isdir(DATASET_DIR):
        raise SystemExit(f"DATASET_DIR does not exist: {DATASET_DIR!r} -- edit it at the top of this file.")
    discretization_dirs = discover_discretization_dirs(DATASET_DIR)
    if not discretization_dirs:
        raise SystemExit(f"No 'discretization<LABEL>' folders found under {DATASET_DIR!r}.")
    discretization_labels = [discretization_label_from_dir(d) for d in discretization_dirs]

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(INDIVIDUAL_OUT_DIR, exist_ok=True)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Discretizations found: {discretization_labels}")
    print(f"Reference planner: {BASE_DISPLAY[REFERENCE_PLANNER]}")
    print(f"Timeout truncation: {TIMEOUT_MS} ms")

    n_rows, n_cols = len(COST_METRIC_ROWS), len(MODEL_IDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.5, 6.0 * n_rows), squeeze=False,
                              gridspec_kw={"hspace": 0.16, "wspace": 0.22})

    jetson_stars = {model_id: jetson_star_x_by_env(model_id) for model_id in MODEL_IDS}

    ref_display = BASE_DISPLAY[REFERENCE_PLANNER]

    all_tables = []
    row_labels = []
    n_total_runs = 0
    for row, metric in enumerate(COST_METRIC_ROWS):
        for col, model_id in enumerate(MODEL_IDS):
            table = build_model_table(model_id, metric, discretization_dirs)
            all_tables.append(table)
            n_total_runs += int(table["N_Total"].sum())
            ax = axes[row][col]
            plot_cell(ax, table, model_id, jetson_stars[model_id])
            save_individual_cell(table, metric, model_id, jetson_stars[model_id])

        # Row label -- the cost metric, rotated in the left margin -- read AFTER this row's three
        # cells are drawn so get_position() reflects their final (post-aspect-lock) layout.
        row_labels.append(axes[row][0].set_ylabel(f"{ref_display} — Time to First Solution (ms, log scale)",
                                                    fontsize=14, fontweight="bold"))
        pos_left = axes[row][0].get_position()
        y_mid = (pos_left.y0 + pos_left.y1) / 2.0
        row_labels.append(fig.text(0.012, y_mid, COST_METRIC_LABELS[metric], rotation=90,
                                    ha="center", va="center", fontsize=16, fontweight="bold"))

    supxlabel = fig.supxlabel("Other Algorithms' Time to First Solution (ms, log scale)",
                               fontsize=15, fontweight="bold", x=0.53, y=0.015)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.97, bottom=0.06)

    base_name = "ttfs_vs_kpax_4s"
    csv_path = os.path.join(OUT_DIR, f"{base_name}.csv")
    pd.concat(all_tables, ignore_index=True).to_csv(csv_path, index=False)
    png_path = os.path.join(OUT_DIR, f"{base_name}.png")
    svg_path = os.path.join(OUT_DIR, f"{base_name}.svg")
    legends = [supxlabel, *row_labels]
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.06, bbox_extra_artists=legends)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.06, bbox_extra_artists=legends)
    plt.close(fig)
    print(f"Wrote {csv_path}, {png_path} + .svg ({n_total_runs} runs across all six panels)")
    print(f"Wrote 6 standalone per-cell PNG+SVG pairs to {INDIVIDUAL_OUT_DIR}")

    if n_total_runs == 0:
        print(
            "\nNo run CSVs were found anywhere under this dataset -- all six panels are empty "
            "placeholders. Drop the archived per-run CSVs into the environment folders and rerun "
            "this script."
        )


if __name__ == "__main__":
    main()
