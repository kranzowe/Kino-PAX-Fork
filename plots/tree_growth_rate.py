"""Tree growth-rate plot: tree size vs. elapsed wall-clock time, one curve per algorithm, one
panel per discretization -- "how fast does each algorithm's tree grow, and does that change with
discretization coarseness" -- for a single environment and model.

Uses ONE SAMPLE run per (algorithm, discretization) cell (the first run zephyr_common.load_runs
finds), not an aggregate/average across the whole 30-run sweep: different runs' tree_size(t)
curves don't share a common iteration/time grid, so "averaging" them the way a single scalar
(e.g. final cost) gets averaged elsewhere in this suite wouldn't produce a meaningful average
curve -- this shows one concrete, real curve per cell instead of a statistically-blended one.
tree_size and elapsed_time_ms are already columns in every per-run CSV this whole plots/ suite
reads (see zephyr_common.load_runs's new `columns` parameter, added for this) -- no new data
needed.

Environment and model are picked RANDOMLY (random.choice) each run, per request ("just use a
random environment and model") -- printed clearly so it's obvious which one a given PNG shows.
Pin ENV_NAME / MODEL_ID below instead of None to fix a specific one.

Y (tree size) is log-scale -- algorithms commonly differ by orders of magnitude in how many
nodes they've built by a given time. X (elapsed time, seconds) stays linear -- the time ranges
involved are modest enough that log compression isn't needed there, and a linear time axis reads
more directly ("how many seconds until N nodes"). A steeper slope at any point IS that algorithm's
instantaneous growth rate there -- no separate derivative plot needed to see it.

Edit DATASET_DIR / OUT_DIR / ENV_NAME / MODEL_ID / METRIC below, then run:
    python plots/tree_growth_rate.py
"""
from __future__ import annotations

import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import (  # noqa: E402
    BASE_COLORS,
    BASE_DISPLAY,
    BASE_NAMES,
    COST_METRICS,
    MODEL_IDS,
    MODEL_NAMES,
    discover_discretization_dirs,
    discover_environments,
    discretization_display,
    discretization_label_from_dir,
    load_runs,
    warn_on_unexpected_star_suffixes,
)

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================
# EDIT THESE to point at a different dataset/output location, or pin a specific environment,
# model, or cost-metric sweep instead of picking randomly.
# ================================================================================================
DATASET_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "tree_growth_rate")
ENV_NAME = None            # <-- pin a specific on-disk environment name, or leave None for random
MODEL_ID = None            # <-- pin a specific model id (1/2/3), or leave None for random
METRIC = COST_METRICS[0]  # which cost-metric sweep to read the sample run from

EXCLUDED_ENVIRONMENTS = {"empty"}  # trivially solved by everyone -- not an interesting comparison


def pick_environment_and_model(discretization_dirs: list) -> tuple:
    """A random (environment, model_id) pair, unless ENV_NAME/MODEL_ID pin one. The environment
    is drawn from the union of environments across every discovered discretization folder, so
    it's only guaranteed to exist in whichever ones actually swept it -- a discretization missing
    it just renders as "no data" in build_growth_figure below, same convention as elsewhere here."""
    env = ENV_NAME
    if env is None:
        candidates = set()
        for disc_dir in discretization_dirs:
            candidates.update(e for e in discover_environments(disc_dir) if e.lower() not in EXCLUDED_ENVIRONMENTS)
        if not candidates:
            raise SystemExit(f"No environments found under any discretization in {DATASET_DIR!r}.")
        env = random.choice(sorted(candidates))
    model_id = MODEL_ID if MODEL_ID is not None else random.choice(MODEL_IDS)
    return env, model_id


def plot_growth(env: str, model_id: int, discretization_dirs: list) -> plt.Figure:
    disc_labels = [discretization_label_from_dir(d) for d in discretization_dirs]
    fig, axes = plt.subplots(1, len(discretization_dirs), figsize=(5.2 * len(discretization_dirs), 4.6),
                              squeeze=False, sharey=True)
    axes = axes[0]

    for ax, disc_dir, disc_label in zip(axes, discretization_dirs, disc_labels):
        env_dir = os.path.join(disc_dir, env)
        has_data = False
        if os.path.isdir(env_dir):
            warn_on_unexpected_star_suffixes(env_dir)
            for planner in BASE_NAMES:
                runs = load_runs(env_dir, env, planner, model_id, disc_label, metrics=(METRIC,),
                                  columns=("elapsed_time_ms", "tree_size"))
                if not runs:
                    continue
                run = runs[0]
                ax.plot(run["elapsed_time_ms"] / 1000.0, run["tree_size"], "-",
                         color=BASE_COLORS[planner], linewidth=1.8, zorder=3)
                has_data = True
        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                     fontsize=10, color="#666666")
        ax.set_yscale("log")
        ax.set_title(discretization_display(disc_label), fontweight="bold")
        ax.set_xlabel("Elapsed time (s)")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    axes[0].set_ylabel("Tree size (nodes, log scale)")

    handles = [Line2D([0], [0], color=BASE_COLORS[p], linewidth=1.8, label=BASE_DISPLAY[p]) for p in BASE_NAMES]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=8, title="Algorithm", frameon=True)

    fig.suptitle(f"Tree growth rate — {MODEL_NAMES[model_id]}, {env} (metric: {METRIC})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def main() -> None:
    if not os.path.isdir(DATASET_DIR):
        raise SystemExit(f"DATASET_DIR does not exist: {DATASET_DIR!r} -- edit it at the top of this file.")
    discretization_dirs = discover_discretization_dirs(DATASET_DIR)
    if not discretization_dirs:
        raise SystemExit(f"No discretization folders found under {DATASET_DIR!r}.")

    env, model_id = pick_environment_and_model(discretization_dirs)
    print(f"Environment: {env}")
    print(f"Model: {model_id} ({MODEL_NAMES[model_id]})")
    print(f"Cost metric sweep read: {METRIC}")
    print(f"Discretizations: {[discretization_label_from_dir(d) for d in discretization_dirs]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plot_growth(env, model_id, discretization_dirs)
    base_name = f"tree_growth_rate_m{model_id}_{env}"
    png_path = os.path.join(OUT_DIR, f"{base_name}.png")
    svg_path = os.path.join(OUT_DIR, f"{base_name}.svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path} + .svg")


if __name__ == "__main__":
    main()
