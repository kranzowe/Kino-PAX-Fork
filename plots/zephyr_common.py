"""Shared data-loading utilities for Kino-PAX benchmark plots (ZEPHYR/JETSON sweeps).

Ports the naming/aggregation conventions from
scripts/process_paper_benchmark_improvement_scatter.m and examples/gpu/paper_benchmark_v2.cu
so every plots/*.py script stays consistent with the paper's own MATLAB pipeline.
"""
from __future__ import annotations

import glob
import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

# --- Planner identity: on-disk token -> display name -> color ---------------------------------
# Verbatim from process_paper_benchmark_improvement_scatter.m's baseNames/baseDisplay (colors are
# our own palette, chosen to be distinct and colorblind-friendly -- an Okabe-Ito-derived set).
KPAX = "KPAX"
KINOPAX_PLUS = "KinoPaxPlus"
SIMPLECOMBO = "KinoPaxSTARTrue_cap100_anc1"
KINOPAX_STAR = "CountingStars_bs120_bf40_ef150_cf750_hg1"

BASE_NAMES = [KPAX, KINOPAX_PLUS, SIMPLECOMBO, KINOPAX_STAR]

BASE_DISPLAY = {
    KPAX: "Kino-PAX",
    KINOPAX_PLUS: "Kino-PAX+",
    SIMPLECOMBO: "SimpleCombo",
    KINOPAX_STAR: "KinoPax*",
}

BASE_COLORS = {
    KPAX: "#222222",         # near-black
    KINOPAX_PLUS: "#648fff",  # blue (also the y=x reference line)
    SIMPLECOMBO: "#ffb000",   # teal / bluish-green
    KINOPAX_STAR: "#dc267f",  # muted magenta
}

# Planners plotted on the x-axis of the ratio scatters (everyone but the Kino-PAX+ reference).
OTHER_PLANNERS = [KPAX, SIMPLECOMBO, KINOPAX_STAR]

# --- Model identity ------------------------------------------------------------------------------
MODEL_IDS = [1, 2, 3]
MODEL_NAMES = {1: "DoubleIntegrator", 2: "DubinsAirplane", 3: "Quad"}
MODEL_LABELS = {
    1: "m1 — 6D Double Integrator",
    2: "m2 — Dubins Airplane 6D",
    3: "m3 — 12D Drone",
}
MODEL_MARKERS = {1: "o", 2: "s", 3: "^"}

# --- Discretization identity (marker shape in the per-model ratio scatters) -------------------
# Only "tiny" exists today; "fine"/"coarse" are prepared for so those scripts need no changes
# once those sweeps exist -- anything unrecognized falls back to a diamond.
DISCRETIZATION_MARKERS = {"tiny": "o", "fine": "s", "coarse": "^"}


def discretization_marker(label: str) -> str:
    return DISCRETIZATION_MARKERS.get(label.lower(), "D")


# Line-style analog of DISCRETIZATION_MARKERS, for plots that encode discretization as a line
# (cost-vs-time convergence curves) rather than a scatter-point shape.
DISCRETIZATION_LINESTYLES = {"tiny": "-", "fine": "--", "coarse": (0, (1, 1))}  # solid/dashed/dotted


def discretization_linestyle(label: str) -> object:
    return DISCRETIZATION_LINESTYLES.get(label.lower(), "-.")


def discretization_display(label: str) -> str:
    return label.capitalize()


COST_METRICS = ("length", "effort")
# "time" is a third cost-metric name, not a different kind of column -- ZEPHYR_30_RUNS_3M_4s_TIME
# is a sweep that minimizes elapsed time itself instead of path length or control effort, and its
# per-run CSVs are structurally identical (same best_cost/elapsed_time_ms columns), just tagged
# "_time_" instead of "_length_"/"_effort_" in the filename. Listed here (not in COST_METRICS,
# which stays length/effort-only for the main dataset) so any script pointed at that dataset can
# look up its label the same way as the other two.
COST_METRIC_LABELS = {"length": "Workspace Path Length", "effort": "Control Effort", "time": "Elapsed Time"}
MAX_FLOAT_THRESH = 1e30
DEFAULT_MAX_RUNS = 35  # search cap; harness writes 30 per series -- matches the .m script


def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def discover_discretization_dirs(dataset_dir: str) -> List[str]:
    """Return every 'discretization*' subfolder of one dataset dir (e.g. TINY/FINE/COARSE)."""
    return sorted(d for d in glob.glob(os.path.join(dataset_dir, "discretization*")) if os.path.isdir(d))


def discretization_label_from_dir(discretization_dir: str) -> str:
    name = os.path.basename(discretization_dir.rstrip("/\\"))
    return re.sub("discretization", "", name, flags=re.IGNORECASE).lower()


def discover_environments(discretization_dir: str) -> List[str]:
    """Return the environment subfolder names actually on disk, whatever their casing."""
    return sorted(
        d for d in os.listdir(discretization_dir)
        if os.path.isdir(os.path.join(discretization_dir, d))
    )


def env_display_name(env: str) -> str:
    overrides = {"narrowpassage": "Narrow", "zigzag": "Windows", "house": "House", "empty": "Empty"}
    return overrides.get(env.lower(), env)


def _candidate_filename(env: str, planner_token: str, delta_tok: str, run: int) -> str:
    if planner_token == KPAX:
        return f"{env}_KPAX_delta{delta_tok}_run{run}.csv"
    if planner_token.startswith("CountingStars") or planner_token.startswith("KinoPaxSTAR"):
        return f"{env}_{planner_token}_delta{delta_tok}_run{run}.csv"
    return f"{env}_delta{delta_tok}_run{run}.csv"  # KinoPaxPlus: no planner token in the filename


# The benchmark harness's own coarsest-resolution label is "large" (matching the older pipeline's
# large/fine/tiny terminology), even though the folder is organized/displayed as "coarse"
# everywhere in these scripts to match "coarse/fine/tiny" -- translate right before it's used to
# build a filename, so every discretization-driven script gets this for free.
_DISCRETIZATION_FILE_TOKEN_OVERRIDES = {"coarse": "large"}


def _discretization_file_token(discretization_label: str) -> str:
    return _DISCRETIZATION_FILE_TOKEN_OVERRIDES.get(discretization_label.lower(), discretization_label)


def load_runs(
    env_dir: str,
    env: str,
    planner_token: str,
    model_id: int,
    discretization_label: str,
    metrics: Sequence[str] = COST_METRICS,
    max_runs: int = DEFAULT_MAX_RUNS,
) -> List[pd.DataFrame]:
    """Load every per-run CSV for (planner, model) in env_dir, pooled across `metrics`.

    Mirrors loadRuns() in process_paper_benchmark_improvement_scatter.m: missing files are
    skipped silently (a planner not swept at a given model/env legitimately has 0 runs). Pass a
    single-element `metrics` to load one cost-metric sweep only (required for cost comparisons --
    length and effort runs are not comparable and must never be pooled together).
    """
    runs: List[pd.DataFrame] = []
    file_token = _discretization_file_token(discretization_label)
    for metric in metrics:
        delta_tok = f"m{model_id}_{file_token}_{metric}"
        for run in range(max_runs):
            fname = _candidate_filename(env, planner_token, delta_tok, run)
            fpath = os.path.join(env_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                df = pd.read_csv(fpath, usecols=["best_cost", "elapsed_time_ms"])
                # best_cost's MAX_FLOAT sentinel is written in fixed-point (~1e38 as 39 digits),
                # which overflows pandas' fast C float tokenizer and silently yields a string
                # (object-dtype) column instead of raising -- force it back to numeric here.
                df["best_cost"] = pd.to_numeric(df["best_cost"], errors="coerce")
                runs.append(df)
            except (ValueError, pd.errors.EmptyDataError) as exc:
                warnings.warn(f"Skipping unreadable run file {fpath}: {exc}")
    return runs


def first_sol_time(df: pd.DataFrame, thresh: float = MAX_FLOAT_THRESH) -> float:
    """Elapsed time (ms) of the first row where best_cost < thresh; -1 if never solved."""
    solved = df.index[df["best_cost"] < thresh]
    if len(solved) == 0:
        return -1.0
    return float(df.loc[solved[0], "elapsed_time_ms"])


def final_cost(df: pd.DataFrame, thresh: float = MAX_FLOAT_THRESH) -> float:
    """Final best_cost value in the run; NaN if it never found a solution.

    Verbatim port of finalCost() in process_paper_benchmark_improvement_scatter.m.
    """
    solved = df[df["best_cost"] < thresh]
    if solved.empty:
        return math.nan
    return float(solved["best_cost"].iloc[-1])


def first_sol_cost(df: pd.DataFrame, thresh: float = MAX_FLOAT_THRESH) -> float:
    """best_cost at the first row where best_cost < thresh (the cost of the FIRST solution found,
    before any anytime refinement); NaN if it never found a solution."""
    solved = df.index[df["best_cost"] < thresh]
    if len(solved) == 0:
        return math.nan
    return float(df.loc[solved[0], "best_cost"])


@dataclass
class CellStats:
    mean: float
    std: float
    n_success: int
    n_total: int


def _aggregate(values: Sequence[float]) -> CellStats:
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    mean = float(np.mean(valid)) if len(valid) else math.nan
    std = float(np.std(valid, ddof=0)) if len(valid) > 1 else 0.0
    return CellStats(mean=mean, std=std, n_success=int(len(valid)), n_total=int(len(arr)))


def aggregate_ttfs(runs: Sequence[pd.DataFrame]) -> CellStats:
    """Mean/std time-to-first-solution (ms) across successful runs; unsolved runs excluded."""
    values = [t if t >= 0 else math.nan for t in (first_sol_time(df) for df in runs)]
    return _aggregate(values)


def aggregate_final_cost(runs: Sequence[pd.DataFrame]) -> CellStats:
    """Mean/std final path cost across successful runs; unsolved runs excluded."""
    return _aggregate([final_cost(df) for df in runs])


def aggregate_first_sol_cost(runs: Sequence[pd.DataFrame]) -> CellStats:
    """Mean/std cost-of-first-solution across successful runs; unsolved runs excluded."""
    return _aggregate([first_sol_cost(df) for df in runs])


def style_log_axis(ax) -> None:
    """Shared look for a log-log ratio-scatter axis: bold, denser tick marks, and decade labels in
    exponent form ("10^3", via LogFormatterMathtext) rather than plain numbers ("1000") -- makes
    the log scale itself legible at a glance instead of requiring the reader to notice the
    uneven tick spacing. Sub-decade minor ticks at 2x/4x/6x/8x each decade (not 2x/5x -- that
    left an oddly large, visually irregular gap between the 2x and 5x marks) keep their tick
    marks for visual density but are left unlabeled -- labeling them (even just the 2x one) was
    tried and dropped again: it added clutter without earning its keep once the major decade
    labels alone were legible."""
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        axis.set_minor_locator(LogLocator(base=10, subs=(2.0, 4.0, 6.0, 8.0)))
        axis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.set_minor_formatter(NullFormatter())

    ax.tick_params(which="major", width=1.6, length=6, labelsize=9)
    ax.tick_params(which="minor", width=1.2, length=3.5, labelsize=8)
    for label in (ax.get_xticklabels(minor=False) + ax.get_yticklabels(minor=False)):
        label.set_fontweight("bold")


def style_linear_axis(ax) -> None:
    """Shared look for a LINEAR ratio-scatter axis (fixed, human-picked bounds rather than a
    log range): just bold, slightly heavier tick marks/labels -- plain numbers are already the
    default on a linear axis, no custom locator/formatter needed the way the log case requires."""
    ax.tick_params(which="major", width=1.6, length=6, labelsize=9)
    ax.tick_params(which="minor", width=1.2, length=3.5, labelsize=8)
    for label in (ax.get_xticklabels(minor=False) + ax.get_xticklabels(minor=True)
                  + ax.get_yticklabels(minor=False) + ax.get_yticklabels(minor=True)):
        label.set_fontweight("bold")


def declutter_label_ys(ax, sorted_items: Sequence, min_gap_points: float = 14.0) -> dict:
    """Nudge row-label y-positions apart so they don't print on top of each other.

    `sorted_items` is a list of (key, true_y) pairs already sorted by true_y descending. Returns
    {key: label_y} such that adjacent labels are at least `min_gap_points` apart in the actual
    RENDERED figure (points -- a physical, dpi-independent unit), nudging later (lower) labels
    down as needed. A fixed multiplicative ratio in data-space (the first approach tried here)
    works for a plot spanning two decades but blows up into a chaotic pile of crossing leader
    lines on a plot whose rows are all clustered within a much narrower range (e.g. cost ratios
    near 1) -- measuring in rendered points is scale-invariant, so the same call works either
    way. Call this only after the axes' final xscale/yscale/xlim/ylim/aspect are set, so
    ax.transData reflects the real layout, not matplotlib's original auto-scaled one.

    If the rows are packed into a small fraction of a much taller fixed axis range (e.g. Control
    Effort's data sitting entirely below 0.3 on a fixed 0-1 axis), the naive greedy nudge above
    only ever pushes labels DOWN from the topmost row's own true position -- so what actually
    limits it is the room between that top row and the bottom of the plot, not the axes' full
    height. Get that wrong (e.g. measure against the whole axes) and labels walk straight out
    through the bottom, overlapping the tick labels or vanishing off the edge. So after the
    greedy pass, if the resulting column needs more room than that, it's compressed back to fit
    (preserving order, trading away perfect min-gap spacing for staying on the plot -- a little
    crowded beats invisible).
    """
    fig = ax.figure
    fig.canvas.draw()  # finalize the aspect-adjusted box position before measuring it
    points_per_pixel = 72.0 / fig.dpi

    raw_points = []
    prev_points = None
    for _key, true_y in sorted_items:
        pixel_y = ax.transData.transform((0, true_y))[1]
        label_points = pixel_y * points_per_pixel
        if prev_points is not None and (prev_points - label_points) < min_gap_points:
            label_points = prev_points - min_gap_points
        prev_points = label_points
        raw_points.append(label_points)

    if raw_points:
        bbox = ax.get_window_extent()
        axes_bottom = bbox.y0 * points_per_pixel
        margin = (bbox.y1 - bbox.y0) * points_per_pixel * 0.03
        available_span = raw_points[0] - (axes_bottom + margin)
        used_span = raw_points[0] - raw_points[-1]
        if used_span > available_span > 0:
            scale = available_span / used_span
            raw_points = [raw_points[0] - (raw_points[0] - p) * scale for p in raw_points]

    label_y = {}
    for (key, _true_y), label_points in zip(sorted_items, raw_points):
        label_y[key] = ax.transData.inverted().transform((0, label_points / points_per_pixel))[1]
    return label_y


def warn_on_unexpected_star_suffixes(env_dir: str) -> None:
    """Flag KinoPaxSTAR*/CountingStars* run files whose tuning suffix isn't in BASE_NAMES.

    A tuning-constant change on the C++ side (paper_benchmark_v2.cu's trueLabel()/
    countingStarsLabel()) silently yields 0 matched runs rather than an error -- this surfaces
    that mismatch instead of a script that quietly reports "0 runs" with no clue why.
    """
    if not os.path.isdir(env_dir):
        return
    known = {SIMPLECOMBO, KINOPAX_STAR}
    seen = set()
    pattern = re.compile(r"^[a-zA-Z]+_((?:CountingStars|KinoPaxSTAR)\w*?)_delta(?=m\d)")
    for fname in os.listdir(env_dir):
        m = pattern.match(fname)
        if m:
            seen.add(m.group(1))
    unexpected = seen - known
    if unexpected:
        warnings.warn(
            f"{env_dir}: found KinoPaxSTAR*/CountingStars* run files with unexpected tuning "
            f"suffixes {sorted(unexpected)} -- these won't match BASE_NAMES {sorted(known)} and "
            "will silently contribute 0 runs. Update BASE_NAMES if the tuning constants changed."
        )
