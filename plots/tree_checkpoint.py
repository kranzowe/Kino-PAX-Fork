"""Tree checkpoint plot: top-down (X-Y), one panel per planner (Kino-PAX, Kino-PAX+, SimpleCombo),
each showing that planner's tree at a chosen wall-clock checkpoint, with its solution trajectory
overlaid if one existed yet at that checkpoint.

Python port of scripts/plot_tree_checkpoint.m (kept for reference, MATLAB-only) -- renders the
wall-clock-checkpoint tree + solution-trajectory dumps written by
examples/gpu/tree_checkpoint_dump.cu (via scripts/run_tree_checkpoint_dump.sh), on the zigzag
corridor, Model 1, V2's canonical "large"/coarse discretization.

Input (in DATA_DIR):
    {env}_{token}_t{ms}ms_tree.csv   columns idx,x,y,z,vx,vy,vz,parent,cost
    {env}_{token}_t{ms}ms_traj.csv   columns step,x,y,z,vx,vy,vz,cost (0 rows if unsolved yet)
    meta.csv                        workspace bounds + start/goal

UNLIKE THE MATLAB ORIGINAL (pick ONE checkpoint by uncommenting a line, then rerun by hand), this
auto-discovers every checkpoint actually dumped in DATA_DIR (from the tree-file names themselves)
and renders one figure per checkpoint in a single run -- consistent with every other script in
this folder, which all loop over whatever's actually on disk rather than needing hand-editing per
run. A token with no tree file at all for a given checkpoint (e.g. Kino-PAX+ often never reaches
the earliest checkpoints) gets a "no data" placeholder tile instead of erroring, same as the
original.

Node color = insertion order (idx, turbo colormap); both nodes and parent-edges are subsampled
evenly over insertion order (NODE_FRACTION / EDGE_FRACTION below) so early- and late-tree
structure both survive without rendering hundreds of thousands of points.

Edit DATA_DIR / OUT_DIR / OBSTACLES_PATH / ENV_NAME below if your data/output folders or the
environment being plotted move, then run:
    python plots/tree_checkpoint.py
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PLOTS_DIR)

# ================================================================================================
# EDIT THESE if your data/output folders or the environment being plotted move.
# ================================================================================================
DATA_DIR = os.path.join(PLOTS_DIR, "DATA", "tree_checkpoint", "TreeCheckpoints")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "tree_checkpoint")
OBSTACLES_PATH = os.path.join(REPO_ROOT, "include", "config", "obstacles", "zigzag", "obstacles.csv")

ENV_NAME = "zigzag"
# On-disk token -> display label. These are tree_checkpoint_dump.cu's OWN token spellings
# (e.g. "KinoPaxSTARTrue", no tuning-constant suffix) -- a different, simpler convention than
# zephyr_common's BASE_NAMES (e.g. "KinoPaxSTARTrue_cap100_anc1"), so this script stays
# self-contained rather than trying to force a match against those.
TOKEN_LABELS = {"KPAX": "Kino-PAX", "KinoPaxPlus": "Kino-PAX+", "KinoPaxSTARTrue": "SimpleCombo"}
TOKENS = list(TOKEN_LABELS)

NODE_FRACTION = 0.15    # fraction of nodes drawn per panel
EDGE_FRACTION = 0.15    # fraction of parent edges drawn per panel
OBSTACLE_ALPHA = 0.35   # top-down = obstacles don't stack along the view axis like 3D does
NODE_SIZE = 6
TRAJ_LINEWIDTH = 2.2
TRAJ_COLOR = "#1a59f2"
SOLVED_COLOR = "#0d7a1a"
UNSOLVED_COLOR = "#991414"
NODATA_COLOR = "#808080"


def discover_checkpoints(data_dir: str, env: str, tokens: list) -> list:
    """Every checkpoint (in ms) with at least one token's tree file dumped, sorted ascending --
    read from the filenames themselves rather than a hand-picked constant."""
    pattern = re.compile(rf"^{re.escape(env)}_(?:{'|'.join(re.escape(t) for t in tokens)})_t(\d+)ms_tree\.csv$")
    found = set()
    for fname in os.listdir(data_dir):
        m = pattern.match(fname)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def load_meta(data_dir: str) -> dict:
    return pd.read_csv(os.path.join(data_dir, "meta.csv")).iloc[0].to_dict()


def load_obstacles(path: str):
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, header=None).to_numpy()


def draw_obstacles(ax, obstacles: np.ndarray) -> None:
    """Top-down X-Y footprint of every box, ignoring z-extent entirely (a projection, not a
    slice) -- same as the .m original's drawObstacles2D."""
    for xmin, ymin, _zmin, xmax, ymax, _zmax in obstacles:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                facecolor="#738296", edgecolor="#4d4d4d",
                                alpha=OBSTACLE_ALPHA, linewidth=0.5, zorder=1))


def draw_workspace_square(ax, meta: dict) -> None:
    lo, hi = meta["W_MIN"], meta["W_MAX"]
    ax.add_patch(Rectangle((lo, lo), hi - lo, hi - lo, edgecolor="#d9d9d9",
                            facecolor="none", linewidth=0.6, zorder=1))


def draw_tree(ax, tree: pd.DataFrame) -> None:
    """Top-down (X-Y) parent edges (NaN-separated segments) + nodes colored by insertion order
    (idx), both subsampled evenly over insertion order -- direct port of the .m original's
    drawTree2D."""
    n = len(tree)
    if n == 0:
        return

    parent = tree["parent"].to_numpy()
    tree_x, tree_y = tree["x"].to_numpy(), tree["y"].to_numpy()
    child_rows = np.flatnonzero(parent >= 0)
    if EDGE_FRACTION > 0 and len(child_rows) > 0:
        n_edge = max(1, round(EDGE_FRACTION * len(child_rows)))
        if n_edge < len(child_rows):
            pick = np.unique(np.round(np.linspace(0, len(child_rows) - 1, n_edge)).astype(int))
            child_rows = child_rows[pick]
        parent_rows = parent[child_rows]  # 0-indexed; rows are already in idx order
        nan_col = np.full(len(child_rows), np.nan)
        xs = np.stack([tree_x[child_rows], tree_x[parent_rows], nan_col], axis=1).ravel()
        ys = np.stack([tree_y[child_rows], tree_y[parent_rows], nan_col], axis=1).ravel()
        ax.plot(xs, ys, "-", color="#b8b8bf", linewidth=0.3, zorder=2)

    n_node = max(1, round(NODE_FRACTION * n))
    ridx = np.unique(np.round(np.linspace(0, n - 1, min(n_node, n))).astype(int))
    ax.scatter(tree_x[ridx], tree_y[ridx], s=NODE_SIZE, c=tree["idx"].to_numpy()[ridx],
               cmap="turbo", alpha=0.8, zorder=3)


def draw_trajectory(ax, traj: pd.DataFrame) -> None:
    ax.plot(traj["x"], traj["y"], "-", color=TRAJ_COLOR, linewidth=TRAJ_LINEWIDTH, zorder=4)


def draw_start_goal(ax, meta: dict) -> None:
    ax.scatter([meta["start_x"]], [meta["start_y"]], s=55, marker="o",
               facecolors="#26b33f", edgecolors="black", linewidths=0.8, zorder=5)
    ax.scatter([meta["goal_x"]], [meta["goal_y"]], s=130, marker="*",
               facecolors="#d92626", edgecolors="black", linewidths=0.8, zorder=5)


def finish_axes(ax, meta: dict) -> None:
    ax.set_xlim(meta["W_MIN"], meta["W_MAX"])
    ax.set_ylim(meta["W_MIN"], meta["W_MAX"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_checkpoint(checkpoint_ms: int, meta: dict, obstacles) -> plt.Figure:
    fig, axes = plt.subplots(1, len(TOKENS), figsize=(15.0, 5.6))
    for ax, token in zip(axes, TOKENS):
        tree_path = os.path.join(DATA_DIR, f"{ENV_NAME}_{token}_t{checkpoint_ms}ms_tree.csv")
        traj_path = os.path.join(DATA_DIR, f"{ENV_NAME}_{token}_t{checkpoint_ms}ms_traj.csv")

        if obstacles is not None:
            draw_obstacles(ax, obstacles)
        draw_workspace_square(ax, meta)

        if not os.path.isfile(tree_path):
            print(f"  [warn] missing tree file: {tree_path}")
            title_text, title_color = "no data", NODATA_COLOR
        else:
            tree = pd.read_csv(tree_path)
            draw_tree(ax, tree)
            if os.path.isfile(traj_path):
                traj = pd.read_csv(traj_path)
            else:
                print(f"  [warn] missing trajectory file: {traj_path}")
                traj = pd.DataFrame()
            if len(traj) > 0:
                draw_trajectory(ax, traj)
                title_text = f"N={len(tree)} nodes | SOLVED, cost={traj['cost'].iloc[-1]:.4f}"
                title_color = SOLVED_COLOR
            else:
                title_text = f"N={len(tree)} nodes | no solution yet"
                title_color = UNSOLVED_COLOR

        draw_start_goal(ax, meta)
        finish_axes(ax, meta)
        ax.set_title(f"{TOKEN_LABELS.get(token, token)}\n{title_text}", fontsize=9, color=title_color)

    fig.suptitle(f"Tree + solution trajectory @ {checkpoint_ms}ms — {ENV_NAME} (Model 1, coarse/large)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def main() -> None:
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"DATA_DIR does not exist: {DATA_DIR!r} -- edit it at the top of this file.")
    meta = load_meta(DATA_DIR)
    obstacles = load_obstacles(OBSTACLES_PATH)
    if obstacles is None:
        print(f"[warn] no obstacle CSV found at {OBSTACLES_PATH!r}; drawing trees without the environment.")
    else:
        print(f"Obstacles: {OBSTACLES_PATH} ({len(obstacles)} boxes)")

    checkpoints = discover_checkpoints(DATA_DIR, ENV_NAME, TOKENS)
    if not checkpoints:
        raise SystemExit(f"No checkpoint tree files found under {DATA_DIR!r} for env={ENV_NAME!r}.")
    print(f"Checkpoints found: {checkpoints} ms")

    os.makedirs(OUT_DIR, exist_ok=True)
    for checkpoint_ms in checkpoints:
        print(f"\nRendering checkpoint t={checkpoint_ms}ms...")
        fig = plot_checkpoint(checkpoint_ms, meta, obstacles)
        base_name = f"tree_checkpoint_{ENV_NAME}_t{checkpoint_ms}ms"
        png_path = os.path.join(OUT_DIR, f"{base_name}.png")
        svg_path = os.path.join(OUT_DIR, f"{base_name}.svg")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {png_path} + .svg")


if __name__ == "__main__":
    main()
