"""Cost vs Time convergence plot: mean best_cost (with a +/-1 std band) against elapsed wall-clock
time, for three algorithms (Kino-PAX, Kino-PAX+, KinoPax* -- SimpleCombo is left out of this plot
specifically, to match the paper's headline comparison) at all three discretizations, one figure
per (model, environment, cost metric) -- length and effort are separate, non-comparable sweeps
(different units) and must never be pooled onto the same y-axis (same rule cost_ratio_scatter.py
follows), so each (model, environment) combination gets 2 figures rather than 1, each with 9
algorithm x discretization lines: color = algorithm, line style = discretization. The time axis is
cut off at PLOT_TIME_LIMIT_MS (5s) -- most runs have long since plateaued by then, and the actual
per-run timeout is far longer, so showing the full range only wastes horizontal space on a flat
tail. Ports plotMeanTime() from
scripts/process_paper_benchmark_v2_and_plot.m: each run's (time, best_cost) series is resampled
onto a shared time grid with a "previous value" step hold (via pandas merge_asof, which also
naturally holds the last value forward past a run's own final sample -- the curve's right edge is
then the mean of every contributing run's OWN final value, not just whichever run happened to
last longest); pre-first-solution samples (best_cost still at the unsolved MAX_FLOAT sentinel) are
NaN'd out so the mean only reflects runs that have actually found a solution by that time.

STOPPED-IMPROVING MARKER: a downward triangle is drawn on a line at the time a MAJORITY of that
line's own runs have already produced their last real sample -- i.e. the mean curve there is now
more than half held-forward/extrapolated rather than live data, which is what "the algorithm ran
out of tree space or ran out of frontier to expand" looks like in an averaged curve. The marker
only appears if that point falls meaningfully before this figure's shared time ceiling (95% of the
longest-running run anywhere in the figure) -- a line that (mostly) runs right up to the shared
time budget gets no marker, since ending because the clock ran out is a different event from
ending because there was nothing left to improve.

The "empty" environment is excluded (trivially solved by everyone -- matches every other script in
this folder). Only the ZEPHYR dataset is used: JETSON's older harness only ever swept one model per
discretization, so it can never supply this figure's "3 discretizations x 1 model" requirement.

Edit DATASET_DIR / OUT_DIR below, then run:
    python plots/cost_vs_time.py
"""
from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import (  # noqa: E402
    BASE_COLORS,
    BASE_DISPLAY,
    BASE_NAMES,
    SIMPLECOMBO,
    COST_METRICS,
    COST_METRIC_LABELS,
    MAX_FLOAT_THRESH,
    MODEL_IDS,
    MODEL_NAMES,
    discover_discretization_dirs,
    discover_environments,
    discretization_display,
    discretization_label_from_dir,
    discretization_linestyle,
    env_display_name,
    load_runs,
    sanitize_name,
    style_linear_axis,
    warn_on_unexpected_star_suffixes,
)

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================
# EDIT THESE to point at the dataset and output location you want.
# ================================================================================================
DATASET_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "cost_vs_time")

EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison
CANONICAL_ENV_ORDER = ["house", "narrowPassage", "zigzag"]

# SimpleCombo left out of this plot specifically -- everything else in plots/*.py still shows it.
PLOTTED_PLANNERS = [p for p in BASE_NAMES if p != SIMPLECOMBO]

PLOT_TIME_LIMIT_MS = 5000.0     # cut the x-axis (and everything computed for it) off at 5s
NUM_TIME_SAMPLES = 400
LIVE_FRACTION_THRESHOLD = 0.5   # a line "plateaus" once fewer than this fraction of its runs are still live
EARLY_STOP_MARGIN = 0.95        # only mark a plateau if it lands before this fraction of the shared ceiling


def resample_run(df: pd.DataFrame, common_time: np.ndarray) -> tuple:
    """Step-hold best_cost onto common_time: previous-value hold, automatically held forward past
    the run's own last sample by merge_asof's 'backward' direction (it always matches the last key
    <= the query time, however far past it that is). NaN before the run's first sample and at any
    point best_cost is still the unsolved sentinel (> MAX_FLOAT_THRESH). Returns (sampled_values,
    last_real_time_ms); last_real_time_ms is 0.0 for a run with no usable rows."""
    run = df[["elapsed_time_ms", "best_cost"]].dropna(subset=["elapsed_time_ms"])
    run = run.sort_values("elapsed_time_ms").drop_duplicates(subset="elapsed_time_ms", keep="last")
    if run.empty:
        return np.full(common_time.shape, np.nan), 0.0
    grid = pd.DataFrame({"elapsed_time_ms": common_time})
    sampled = pd.merge_asof(grid, run, on="elapsed_time_ms", direction="backward")
    values = sampled["best_cost"].to_numpy(dtype=float)
    values[values > MAX_FLOAT_THRESH] = np.nan
    return values, float(run["elapsed_time_ms"].iloc[-1])


def aggregate_line(runs: list, common_time: np.ndarray) -> dict:
    """Mean/std cost curve for one (algorithm, discretization) line, plus the common_time index at
    which a majority of its runs have already produced their last real sample (None if that never
    happens within common_time -- i.e. most runs are still live at the very end). None if no runs."""
    if not runs:
        return None
    sampled, last_times = [], []
    for df in runs:
        values, last_t = resample_run(df, common_time)
        sampled.append(values)
        last_times.append(last_t)
    A = np.vstack(sampled)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice before anyone solves
        mean = np.nanmean(A, axis=0)
        std = np.nanstd(A, axis=0)
    if np.all(np.isnan(mean)):
        return None
    last_times = np.asarray(last_times)
    n_runs = len(runs)
    n_live = np.array([np.sum(last_times >= t) for t in common_time])
    below = n_live < (LIVE_FRACTION_THRESHOLD * n_runs)
    plateau_idx = int(np.argmax(below)) if np.any(below) else None
    return {"mean": mean, "std": std, "n_runs": n_runs, "plateau_idx": plateau_idx}


def plot_cost_vs_time(model_id: int, env: str, metric: str, discretization_dirs: list,
                       discretization_labels: list, out_path: str) -> tuple:
    """Build and save one (model, environment, metric) figure. Returns (n_total_runs, curves_df)."""
    per_line_runs = {}   # (planner, disc_label) -> list[DataFrame]
    global_t_max = 0.0
    n_total_runs = 0
    for disc_dir, disc_label in zip(discretization_dirs, discretization_labels):
        on_disk_envs = [e for e in discover_environments(disc_dir) if e.lower() == env.lower()]
        if not on_disk_envs:
            continue
        env_dir = os.path.join(disc_dir, on_disk_envs[0])
        warn_on_unexpected_star_suffixes(env_dir)
        for planner in PLOTTED_PLANNERS:
            runs = load_runs(env_dir, on_disk_envs[0], planner, model_id, disc_label, metrics=(metric,))
            per_line_runs[(planner, disc_label)] = runs
            n_total_runs += len(runs)
            for df in runs:
                if not df.empty:
                    global_t_max = max(global_t_max, float(df["elapsed_time_ms"].max()))

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    legends = []
    curve_rows = []
    plateau_marker_used = False

    if global_t_max <= 0 or n_total_runs == 0:
        ax.text(0.5, 0.5, "No successful runs found yet.", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#666666")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        plot_t_max = min(global_t_max, PLOT_TIME_LIMIT_MS)
        common_time = np.linspace(0.0, plot_t_max, NUM_TIME_SAMPLES)
        for disc_label in discretization_labels:
            ls = discretization_linestyle(disc_label)
            for planner in PLOTTED_PLANNERS:
                runs = per_line_runs.get((planner, disc_label), [])
                agg = aggregate_line(runs, common_time)
                if agg is None:
                    continue
                valid = ~np.isnan(agg["mean"])
                if not np.any(valid):
                    continue
                color = BASE_COLORS[planner]
                t, mu = common_time[valid], agg["mean"][valid]
                sd = np.nan_to_num(agg["std"][valid])
                ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.10, linewidth=0, zorder=1)
                ax.plot(t, mu, linestyle=ls, color=color, linewidth=2.0, zorder=3)

                curve_rows.append(pd.DataFrame({
                    "Model": MODEL_NAMES[model_id],
                    "Environment": env_display_name(env),
                    "CostMetric": metric,
                    "Algorithm": BASE_DISPLAY[planner],
                    "Discretization": disc_label,
                    "Elapsed_Time_ms": t,
                    "Mean_Cost": mu,
                    "Std_Cost": sd,
                    "N_Runs": agg["n_runs"],
                }))

                p_idx = agg["plateau_idx"]
                if p_idx is not None and not np.isnan(agg["mean"][p_idx]):
                    p_time = common_time[p_idx]
                    if p_time < EARLY_STOP_MARGIN * global_t_max:
                        ax.scatter(p_time, agg["mean"][p_idx], marker="v", s=70,
                                   facecolors=color, edgecolors="black", linewidths=0.8, zorder=5)
                        plateau_marker_used = True

        style_linear_axis(ax)
        ax.set_xlim(0, plot_t_max)

        algo_handles = [
            Line2D([0], [0], color=BASE_COLORS[p], linewidth=2.2, label=BASE_DISPLAY[p])
            for p in PLOTTED_PLANNERS
        ]
        disc_handles = [
            Line2D([0], [0], color="#555555", linewidth=2.0, linestyle=discretization_linestyle(d),
                   label=discretization_display(d))
            for d in discretization_labels
        ]
        if plateau_marker_used:
            disc_handles.append(Line2D(
                [0], [0], marker="v", linestyle="", markerfacecolor="#888888",
                markeredgecolor="black", markersize=8,
                label="Majority of runs stopped\nimproving (tree/frontier\nexhausted)",
            ))
        legend1 = ax.legend(handles=algo_handles, title="Algorithm (color)", loc="upper left",
                             bbox_to_anchor=(1.02, 1.0), fontsize=9, title_fontsize=9, frameon=True)
        ax.add_artist(legend1)
        legend2 = ax.legend(handles=disc_handles, title="Discretization (line style)", loc="upper left",
                             bbox_to_anchor=(1.02, 0.5), fontsize=9, title_fontsize=9, frameon=True)
        legends = [legend1, legend2]

    metric_title = COST_METRIC_LABELS[metric]
    ax.set_xlabel("Elapsed Time (ms)")
    ax.set_ylabel(f"Best Cost So Far ({metric_title})")
    ax.set_title(
        f"Cost vs Time — {env_display_name(env)}, {MODEL_NAMES[model_id]} (m{model_id}), {metric_title}\n"
        "(mean ± 1 std across runs)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", bbox_extra_artists=legends)
    fig.savefig(os.path.splitext(out_path)[0] + ".svg", bbox_inches="tight", bbox_extra_artists=legends)
    plt.close(fig)

    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    return n_total_runs, curves


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

    on_disk_envs = set()
    for disc_dir in discretization_dirs:
        on_disk_envs.update(e for e in discover_environments(disc_dir) if e.lower() not in EXCLUDED_ENVIRONMENTS)
    environments = [e for e in CANONICAL_ENV_ORDER if e in on_disk_envs]
    environments += sorted(e for e in on_disk_envs if e not in environments)
    print(f"Environments found: {environments}")

    any_data = False
    for model_id in MODEL_IDS:
        model_name = MODEL_NAMES[model_id]
        for metric in COST_METRICS:
            for env in environments:
                base_name = f"cost_vs_time_{metric}_m{model_id}_{sanitize_name(model_name)}_{sanitize_name(env)}"
                png_path = os.path.join(OUT_DIR, f"{base_name}.png")
                n_total_runs, curves = plot_cost_vs_time(
                    model_id, env, metric, discretization_dirs, discretization_labels, png_path
                )
                if n_total_runs == 0:
                    print(f"  [{metric} m{model_id} {model_name} {env}] no run files found -- skipping.")
                else:
                    any_data = True
                csv_path = os.path.join(OUT_DIR, f"{base_name}.csv")
                curves.to_csv(csv_path, index=False)
                print(f"  [{metric} m{model_id} {model_name} {env}] wrote {csv_path}, {png_path} + .svg "
                      f"({n_total_runs} runs)")

    if not any_data:
        print(
            "\nNo run CSVs were found anywhere under this dataset -- the plots above are empty "
            "placeholders. Drop the archived per-run CSVs into the environment folders and rerun "
            "this script."
        )


if __name__ == "__main__":
    main()
