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


def discretization_display(label: str) -> str:
    return label.capitalize()


COST_METRICS = ("length", "effort")
COST_METRIC_LABELS = {"length": "Workspace Path Length", "effort": "Control Effort"}
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
    overrides = {"narrowpassage": "Narrow Passage", "zigzag": "ZigZag", "house": "House", "empty": "Empty"}
    return overrides.get(env.lower(), env)


def _candidate_filename(env: str, planner_token: str, delta_tok: str, run: int) -> str:
    if planner_token == KPAX:
        return f"{env}_KPAX_delta{delta_tok}_run{run}.csv"
    if planner_token.startswith("CountingStars") or planner_token.startswith("KinoPaxSTAR"):
        return f"{env}_{planner_token}_delta{delta_tok}_run{run}.csv"
    return f"{env}_delta{delta_tok}_run{run}.csv"  # KinoPaxPlus: no planner token in the filename


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
    for metric in metrics:
        delta_tok = f"m{model_id}_{discretization_label}_{metric}"
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
