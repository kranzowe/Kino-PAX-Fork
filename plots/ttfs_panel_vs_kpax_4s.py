"""Time-to-first-solution scatter, 4-SECOND-TRUNCATED, ALL SIX (cost metric x model) combinations
in one figure: two rows (Control Effort, Workspace Path Length) x three columns (one per model,
left to right). Reference planner is Kino-PAX (KPAX) -- every other algorithm (Kino-PAX+,
Kino-PAX#; SimpleCombo excluded by default, see INCLUDE_SIMPLECOMBO) is plotted against it,
color = algorithm, shape = discretization, log-log axes, y=x "as fast as Kino-PAX" line, no
per-point environment-name labels.

Design lineage: this is a (metric x model) extension of ttfs_ratio_scatter_panel_no_env_labels.py
(same REFERENCE_PLANNER/no-env-label/5x-faster-slower-line/legend-on-drone-panel design), with
three changes: (1) cost metric becomes a second panel axis (that file pools length+effort into one
set of 3 panels; this one keeps them separate, 2x3), (2) TTFS is truncated to a 4s budget (see
TIMEOUT_MS / zephyr_common.aggregate_ttfs_truncated) rather than using each run's true,
unbounded first-solution time, and (3) no per-panel model-name title, and axis label TEXT (not
tick labels, which were already bold via style_log_axis) is bold. The Jetson-hardware overlay
from that file is intentionally dropped here -- it's an unrelated proof-of-concept comparison
against a different dataset, not part of this specific request.

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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
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
    aggregate_ttfs_truncated,
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
# EDIT THESE to point at the dataset/output location you want, or change the truncation budget.
# ================================================================================================
DATASET_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "scatter_panels_4s")
INDIVIDUAL_OUT_DIR = os.path.join(OUT_DIR, "individual")
TIMEOUT_MS = 4000.0

# Row order matches how the combinations were requested: Control Effort first, then Path Length.
COST_METRIC_ROWS = ("effort", "length")

REFERENCE_PLANNER = KPAX     # <-- Kino-PAX. Every other algorithm is plotted relative to it.
INCLUDE_SIMPLECOMBO = False  # <-- TOGGLE. Flip to True to bring SimpleCombo back into this panel.

EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison

PLOTTED_OTHER_PLANNERS = [
    p for p in BASE_NAMES
    if p != REFERENCE_PLANNER and (INCLUDE_SIMPLECOMBO or p != SIMPLECOMBO)
]

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


def plot_cell(ax, table: pd.DataFrame) -> None:
    """Draw one (metric, model) cell -- KPAX-referenced log-log scatter, no env-name labels, no
    per-panel title (dropped -- see module docstring)."""
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
        return

    all_x = pd.concat([plotted["Mean_TTFS_ms"], plotted["Reference_Mean_TTFS_ms"]])
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
    # marker on it, exactly where that row crosses y=x.
    for (_env, disc), group in plotted.groupby(["Environment", "Discretization"]):
        y = group["Reference_Mean_TTFS_ms"].iloc[0]
        x_min = group["Mean_TTFS_ms"].min()
        x_max = group["Mean_TTFS_ms"].max()
        ax.plot([x_min / 1.08, x_max * 1.08], [y, y], color="#999999", linestyle=":",
                 linewidth=1.0, zorder=0)
        ax.scatter(y, y, s=40, marker=discretization_marker(disc),
                   facecolors=BASE_COLORS[REFERENCE_PLANNER], edgecolors="black", linewidths=0.8, zorder=6)

    style_log_axis(ax)
    _add_edge_ticks(ax, lo, hi)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)


def save_individual_cell(table: pd.DataFrame, metric: str, model_id: int,
                          algo_handles: list, disc_handles: list) -> None:
    """Standalone single-panel version of one (metric, model) cell -- same plot_cell() drawing,
    but with its own axis labels (bold, matching the combined panel's wording) and its own copy
    of both legends, since a standalone image can't rely on a shared row/figure label or a
    neighboring panel's legend the way a cell inside the combined 2x3 figure can."""
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    plot_cell(ax, table)

    ref_display = BASE_DISPLAY[REFERENCE_PLANNER]
    ax.set_xlabel("Time to First Solution (ms, log scale) — other algorithms",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{ref_display} — Time to First Solution (ms, log scale)",
                  fontsize=11, fontweight="bold")

    legend2 = ax.legend(handles=disc_handles, title="Discretization (shape)", loc="lower right",
                         bbox_to_anchor=(0.99, 0.02), fontsize=8, title_fontsize=8, frameon=True)
    ax.add_artist(legend2)
    ax.legend(handles=algo_handles, title="Algorithm (color)", loc="lower right",
              bbox_to_anchor=(0.99, 0.24), fontsize=8, title_fontsize=8, frameon=True)

    base = f"ttfs_vs_kpax_4s_{metric}_{MODEL_NAMES[model_id]}"
    fig.savefig(os.path.join(INDIVIDUAL_OUT_DIR, f"{base}.png"), dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(INDIVIDUAL_OUT_DIR, f"{base}.svg"), bbox_inches="tight", pad_inches=0.05)
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
    algo_handles.append(
        Line2D([0], [0], color="red", linestyle=":", linewidth=1.2, alpha=0.8,
               label=f"5x faster/slower than {ref_display}")
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
    for row, metric in enumerate(COST_METRIC_ROWS):
        for col, model_id in enumerate(MODEL_IDS):
            table = build_model_table(model_id, metric, discretization_dirs)
            all_tables.append(table)
            n_total_runs += int(table["N_Total"].sum())
            ax = axes[row][col]
            plot_cell(ax, table)
            save_individual_cell(table, metric, model_id, algo_handles, disc_handles)

            if col == drone_col:
                # Both Drone cells get their own copy of both legends, bottom-right -- that
                # corner sits clear of this model's data.
                legend2 = ax.legend(handles=disc_handles, title="Discretization (shape)", loc="lower right",
                                     bbox_to_anchor=(0.99, 0.02), fontsize=8, title_fontsize=8, frameon=True)
                ax.add_artist(legend2)
                ax.legend(handles=algo_handles, title="Algorithm (color)", loc="lower right",
                          bbox_to_anchor=(0.99, 0.22), fontsize=8, title_fontsize=8, frameon=True)

        # Row label -- the cost metric, rotated in the left margin -- read AFTER this row's three
        # cells are drawn so get_position() reflects their final (post-aspect-lock) layout.
        axes[row][0].set_ylabel(f"{ref_display} — Time to First Solution (ms, log scale)",
                                 fontsize=10, fontweight="bold")
        pos_left = axes[row][0].get_position()
        y_mid = (pos_left.y0 + pos_left.y1) / 2.0
        row_labels.append(fig.text(0.012, y_mid, COST_METRIC_LABELS[metric], rotation=90,
                                    ha="center", va="center", fontsize=13, fontweight="bold"))

    supxlabel = fig.supxlabel("Time to First Solution (ms, log scale) — other algorithms",
                               fontsize=11, fontweight="bold", x=0.53, y=0.015)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.97, bottom=0.06)

    base_name = "ttfs_vs_kpax_4s"
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
            "\nNo run CSVs were found anywhere under this dataset -- all six panels are empty "
            "placeholders. Drop the archived per-run CSVs into the environment folders and rerun "
            "this script."
        )


if __name__ == "__main__":
    main()
