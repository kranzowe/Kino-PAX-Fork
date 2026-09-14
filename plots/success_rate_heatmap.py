"""Success rate heatmap: the same data as success_rate_table.py, as one wide combined figure
instead of three separate LaTeX tables. Two rows (Workspace Path Length, Control Effort) x three
columns (one per model) -- same grid layout as cost_big_panel.py -- and each of the six cells is
itself a small heatmap: one row per (Environment, Region) combination, one column per algorithm,
color = success rate (0-100%, red-yellow-green), with the exact percentage annotated in every
cell so color alone never has to carry the precision. All four algorithms are kept (unlike
ttfs_cost_table.py / cost_table_wide.py, there's no specific paper table this needs to match
column-for-column). NOT pooled across cost metrics, for the same reason as success_rate_table.py:
success/failure genuinely differs between the length and effort sweeps in this data (e.g.
Kino-PAX+ goes to 0% at Coarse for some (model, metric) cells but not others), and pooling would
average that away. Regions are Zephyr's Coarse/Fine/Tiny plus the two confirmed Jetson rows -- see
ttfs_cost_table.py's module docstring for the full two-harness provenance story. A region whose
folder doesn't exist yet contributes "--" (grey) cells rather than erroring.

Edit ZEPHYR_DIR / JETSON_DIR / OUT_DIR below if your dataset folders move, then run:
    python plots/success_rate_heatmap.py
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import (  # noqa: E402
    BASE_DISPLAY,
    BASE_NAMES,
    COST_METRIC_LABELS,
    COST_METRICS,
    DEFAULT_MAX_RUNS,
    KPAX,
    MODEL_IDS,
    aggregate_ttfs,
    env_display_name,
    load_runs,
    warn_on_unexpected_star_suffixes,
)

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================
# EDIT THESE if your dataset folders move.
# ================================================================================================
ZEPHYR_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs")
JETSON_DIR = os.path.join(PLOTS_DIR, "DATA", "JETSON_20_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "success_rate")

ENVIRONMENTS = ["house", "narrowPassage", "zigzag"]  # on-disk spelling; "empty" excluded

TABLE_PLANNERS = list(BASE_NAMES)  # all four -- see module docstring
COLUMN_LABELS = {p: BASE_DISPLAY[p] for p in TABLE_PLANNERS}

MODEL_SUBTITLES = {
    1: "6D Double Integrator",
    2: "6D Dubins Airplane",
    3: "12D Nonlinear Drone",
}


def _candidate_filename_no_model_tag(env: str, planner_token: str, delta_tok: str, run: int) -> str:
    """Filename builder for discretizationCOARSE's older, pre-v2 pipeline (no 'm<N>_' model tag)."""
    if planner_token == KPAX:
        return f"{env}_KPAX_delta{delta_tok}_run{run}.csv"
    if planner_token.startswith("CountingStars") or planner_token.startswith("KinoPaxSTAR"):
        return f"{env}_{planner_token}_delta{delta_tok}_run{run}.csv"
    return f"{env}_delta{delta_tok}_run{run}.csv"


def load_runs_no_model_tag(env_dir, env, planner_token, discretization_label, metrics):
    runs = []
    for metric in metrics:
        delta_tok = f"{discretization_label}_{metric}"
        for run in range(DEFAULT_MAX_RUNS):
            fpath = os.path.join(env_dir, _candidate_filename_no_model_tag(env, planner_token, delta_tok, run))
            if not os.path.isfile(fpath):
                continue
            try:
                df = pd.read_csv(fpath, usecols=["best_cost", "elapsed_time_ms"])
                df["best_cost"] = pd.to_numeric(df["best_cost"], errors="coerce")
                runs.append(df)
            except (ValueError, pd.errors.EmptyDataError):
                pass
    return runs


def regions_for_model(model_id: int) -> list:
    """Region rows for one model -- label, discretization folder, on-disk discretization token,
    and whether that filename carries a "m<N>_" model tag. The Jetson row is model-specific -- see
    the module docstring."""
    regions = [{"label": "Coarse", "dir": os.path.join(ZEPHYR_DIR, "discretizationCOARSE"),
                "token": "large", "model_tag": True}]
    if model_id == 3:
        regions.append({"label": "Jetson (Coarse)", "dir": os.path.join(JETSON_DIR, "discretizationCOARSE"),
                         "token": "large", "model_tag": False})
    regions.append({"label": "Fine", "dir": os.path.join(ZEPHYR_DIR, "discretizationFINE"),
                     "token": "fine", "model_tag": True})
    if model_id in (1, 2):
        regions.append({"label": "Jetson (Fine)",
                         "dir": os.path.join(JETSON_DIR, "discretizationFINE", "FINE"),
                         "token": "fine", "model_tag": True})
    regions.append({"label": "Tiny", "dir": os.path.join(ZEPHYR_DIR, "discretizationTINY"),
                     "token": "tiny", "model_tag": True})
    return regions


def success_rate(runs: list) -> float:
    """Percentage of runs that found ANY solution -- see success_rate_table.py for the full
    rationale (aggregate_ttfs's success criterion is exactly "did this run solve at all")."""
    stats = aggregate_ttfs(runs)
    return 100.0 * stats.n_success / stats.n_total if stats.n_total else math.nan


def build_matrix(model_id: int, metric: str, regions: list) -> tuple:
    """(matrix, row_labels) -- matrix[row, col] = success rate % for TABLE_PLANNERS[col] at
    (env, region) row (env-major, so all of one environment's regions stay together), NaN where
    that region's folder doesn't exist for this environment."""
    row_labels = []
    rows = []
    for env in ENVIRONMENTS:
        for region in regions:
            row_labels.append(f"{env_display_name(env)} — {region['label']}")
            env_dir = os.path.join(region["dir"], env)
            row = []
            if not os.path.isdir(env_dir):
                row = [math.nan] * len(TABLE_PLANNERS)
            else:
                warn_on_unexpected_star_suffixes(env_dir)
                for planner in TABLE_PLANNERS:
                    if region["model_tag"]:
                        runs = load_runs(env_dir, env, planner, model_id, region["token"], metrics=(metric,))
                    else:
                        runs = load_runs_no_model_tag(env_dir, env, planner, region["token"], (metric,))
                    row.append(success_rate(runs))
            rows.append(row)
    return np.array(rows, dtype=float), row_labels


def plot_cell(ax, matrix: np.ndarray, row_labels: list, n_envs: int, n_regions_per_env: int,
              is_top_row: bool, subtitle: str, cmap) -> object:
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(TABLE_PLANNERS)))
    ax.set_xticklabels([COLUMN_LABELS[p] for p in TABLE_PLANNERS], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            if math.isnan(value):
                text, color = "--", "#666666"
            else:
                text, color = f"{value:.0f}%", ("black" if value >= 50 else "white")
            ax.text(c, r, text, ha="center", va="center", fontsize=7, color=color)

    # A horizontal line between each environment's block of regions, so the env-major grouping
    # reads clearly instead of looking like one undifferentiated stack of rows.
    for i in range(1, n_envs):
        ax.axhline(i * n_regions_per_env - 0.5, color="white", linewidth=1.5)

    ax.set_xticks(np.arange(-0.5, len(TABLE_PLANNERS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    if is_top_row:
        ax.set_title(subtitle, fontsize=12, fontweight="bold")
    return im


def main() -> None:
    if not os.path.isdir(ZEPHYR_DIR):
        raise SystemExit(f"ZEPHYR_DIR does not exist: {ZEPHYR_DIR!r} -- edit it at the top of this file.")

    os.makedirs(OUT_DIR, exist_ok=True)

    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="#dddddd")

    n_rows, n_cols = len(COST_METRICS), len(MODEL_IDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17.0, 11.5),
                              gridspec_kw={"hspace": 0.55, "wspace": 0.5})

    row_labels_out = []
    im = None
    for row, metric in enumerate(COST_METRICS):
        for col, model_id in enumerate(MODEL_IDS):
            regions = regions_for_model(model_id)
            matrix, row_labels = build_matrix(model_id, metric, regions)
            ax = axes[row][col]
            im = plot_cell(ax, matrix, row_labels, len(ENVIRONMENTS), len(regions),
                            is_top_row=(row == 0), subtitle=MODEL_SUBTITLES[model_id], cmap=cmap)
            row_labels_out.append(pd.DataFrame({
                "Model": MODEL_SUBTITLES[model_id], "CostMetric": metric,
                "Row": row_labels, **{COLUMN_LABELS[p]: matrix[:, i] for i, p in enumerate(TABLE_PLANNERS)},
            }))

        pos_left = axes[row][0].get_position()
        y_mid = (pos_left.y0 + pos_left.y1) / 2.0
        fig.text(0.01, y_mid, COST_METRIC_LABELS[metric], rotation=90, ha="center", va="center",
                  fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Success rate (%)", fontsize=10)

    fig.suptitle("Success Rate — Kino-PAX / Kino-PAX+ / SimpleCombo / KinoPax*", fontsize=14, fontweight="bold")

    csv_path = os.path.join(OUT_DIR, "success_rate_heatmap.csv")
    pd.concat(row_labels_out, ignore_index=True).to_csv(csv_path, index=False)
    png_path = os.path.join(OUT_DIR, "success_rate_heatmap.png")
    svg_path = os.path.join(OUT_DIR, "success_rate_heatmap.svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {csv_path}, {png_path} + .svg")


if __name__ == "__main__":
    main()
