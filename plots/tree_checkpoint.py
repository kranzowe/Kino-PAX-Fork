"""Tree checkpoint plot: top-down (X-Y), one panel per planner (Kino-PAX, Kino-PAX+, KinoPax*),
each showing that planner's tree at a chosen wall-clock checkpoint, with its solution trajectory
overlaid if one existed yet at that checkpoint.

Python port of scripts/plot_tree_checkpoint.m (kept for reference, MATLAB-only, and kept in sync
with it -- re-check this file whenever that one changes) -- renders the wall-clock-checkpoint
tree + solution-trajectory dumps written by examples/gpu/tree_checkpoint_dump.cu (via
scripts/run_tree_checkpoint_dump.sh), on the zigzag corridor, Model 1, V2's canonical
"large"/coarse discretization.

Input (in DATA_DIR):
    {env}_{token}_t{ms}ms_tree.csv   columns idx,x,y,z,vx,vy,vz,parent,cost -- already dumped at
                                     only 10% of leaf trajectories (tree_checkpoint_dump.cu's own
                                     doing); NODE_FRACTION/EDGE_FRACTION below are a further,
                                     purely cosmetic render-time subsample on top of that.
    {env}_{token}_t{ms}ms_traj.csv   columns step,x,y,z,vx,vy,vz,cost (0 rows if unsolved yet)
    meta.csv                        workspace bounds + start/goal

UNLIKE THE MATLAB ORIGINAL (pick ONE checkpoint by uncommenting a line, then rerun by hand), this
auto-discovers every checkpoint actually dumped in DATA_DIR (from the tree-file names themselves)
and renders one figure per checkpoint in a single run -- consistent with every other script in
this folder, which all loop over whatever's actually on disk rather than needing hand-editing per
run. A token with no tree file at all for a given checkpoint (e.g. Kino-PAX+ often never reaches
the earliest checkpoints) gets a "no data" placeholder tile instead of erroring, same as the
original.

Node color = insertion order (idx, turbo colormap), subsampled evenly over insertion order
(NODE_FRACTION below) so early- and late-tree structure both survive without rendering hundreds
of thousands of points. Edges are drawn for that SAME subsampled node set (never an independently
chosen domain) -- an earlier version of this script picked the rendered nodes and the rendered
edges as two separate evenly-spaced subsamples, which could each land on different rows, so a
displayed node's one edge back to its parent often just wasn't in the edge subsample -- it looked
exactly like a bug where nodes have no parent, but checking the dumped tree CSVs directly (every
`parent` value is either -1 or a valid, in-range row) confirms every node genuinely does have one;
examples/gpu/tree_checkpoint_dump.cu's leaf-sampling + full-ancestor-chain-preservation logic is
correct as written. EDGE_FRACTION now only trims which of the shown nodes' own edges get drawn
(a subset of the node subsample itself), not a second independent domain.

FRAME_CHECKPOINTS_MS additionally exports each panel of those checkpoints as its own standalone
square image (no title/axis labels/ticks/grid -- just the workspace square, obstacles, tree,
trajectory, and start/goal) into OUT_DIR/frames/, alongside the normal combined multi-panel figure
every checkpoint still gets.

Edit DATA_DIR / OUT_DIR / OBSTACLES_PATH / ENV_NAME / FRAME_CHECKPOINTS_MS below if your
data/output folders, the environment being plotted, or which checkpoints get frame exports move,
then run:
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
# (e.g. "CountingStars", no tuning-constant suffix) -- a different, simpler convention than
# zephyr_common's BASE_NAMES (e.g. "CountingStars_bs120_bf40_ef150_cf750_hg1"), so this script
# stays self-contained rather than trying to force a match against those.
TOKEN_LABELS = {"KPAX": "Kino-PAX", "KinoPaxPlus": "Kino-PAX+", "CountingStars": "KinoPax*"}
TOKENS = list(TOKEN_LABELS)

NODE_FRACTION = 1.0    # fraction of nodes drawn per panel -- 0.15 was fine for KPAX/CountingStars
                        # (already dense enough to look like a solid mass either way) but made
                        # Kino-PAX+ look poorly connected: at some checkpoints its ~300k nodes are
                        # heavily concentrated in tight clusters rather than spread evenly (a real
                        # search-behavior difference, confirmed directly against the dumped CSVs --
                        # not a rendering or data bug), so a fixed 15% sample left its already-thin
                        # open-corridor coverage looking like scattered, disconnected-looking long
                        # strands. Bumping this doesn't meaningfully change how the already-dense
                        # trees look, but fills in enough of Kino-PAX+'s sparser regions to read as
                        # connected structure instead.
EDGE_FRACTION = 1.0     # fraction of the SHOWN nodes' own edges drawn -- see module docstring;
                        # 1.0 means every shown node's edge to its parent is drawn, so nothing
                        # ever appears to float disconnected.
OBSTACLE_ALPHA = 0.35   # top-down = obstacles don't stack along the view axis like 3D does
NODE_SIZE = 6
TRAJ_LINEWIDTH = 2.2
TRAJ_COLOR = "#1a59f2"
SOLVED_COLOR = "#0d7a1a"
UNSOLVED_COLOR = "#991414"
NODATA_COLOR = "#808080"
GOAL_COLOR = "#26b33f"

FRAME_CHECKPOINTS_MS = (100, 1000)  # which checkpoints also get standalone per-panel frame exports


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
    """Top-down (X-Y) nodes colored by insertion order (idx), subsampled evenly over insertion
    order (NODE_FRACTION), plus parent edges for that SAME shown-node set (never an independently
    subsampled domain -- see module docstring for why that used to make connected nodes look
    like they were floating with no parent)."""
    n = len(tree)
    if n == 0:
        return

    parent = tree["parent"].to_numpy()
    tree_x, tree_y = tree["x"].to_numpy(), tree["y"].to_numpy()

    n_node = max(1, round(NODE_FRACTION * n))
    ridx = np.unique(np.round(np.linspace(0, n - 1, min(n_node, n))).astype(int))

    edge_src = ridx[parent[ridx] >= 0]
    if EDGE_FRACTION < 1.0 and len(edge_src) > 0:
        n_edge = max(1, round(EDGE_FRACTION * len(edge_src)))
        if n_edge < len(edge_src):
            pick = np.unique(np.round(np.linspace(0, len(edge_src) - 1, n_edge)).astype(int))
            edge_src = edge_src[pick]
    if len(edge_src) > 0:
        parent_rows = parent[edge_src]
        nan_col = np.full(len(edge_src), np.nan)
        xs = np.stack([tree_x[edge_src], tree_x[parent_rows], nan_col], axis=1).ravel()
        ys = np.stack([tree_y[edge_src], tree_y[parent_rows], nan_col], axis=1).ravel()
        ax.plot(xs, ys, "-", color="#b8b8bf", linewidth=0.3, zorder=2)

    ax.scatter(tree_x[ridx], tree_y[ridx], s=NODE_SIZE, c=tree["idx"].to_numpy()[ridx],
               cmap="turbo", alpha=0.8, zorder=3)


def draw_trajectory(ax, traj: pd.DataFrame) -> None:
    ax.plot(traj["x"], traj["y"], "-", color=TRAJ_COLOR, linewidth=TRAJ_LINEWIDTH, zorder=4)


def draw_start_goal(ax, meta: dict) -> None:
    ax.scatter([meta["start_x"]], [meta["start_y"]], s=55, marker="o",
               facecolors=GOAL_COLOR, edgecolors="black", linewidths=0.8, zorder=5)
    ax.scatter([meta["goal_x"]], [meta["goal_y"]], s=220, marker="o",
               facecolors=GOAL_COLOR, edgecolors="black", linewidths=0.8, zorder=5)


def finish_axes(ax, meta: dict) -> None:
    ax.set_xlim(meta["W_MIN"], meta["W_MAX"])
    ax.set_ylim(meta["W_MIN"], meta["W_MAX"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)


def draw_panel(ax, token: str, checkpoint_ms: int, meta: dict, obstacles) -> tuple:
    """Draws obstacles/workspace/tree/trajectory/start-goal for one token at one checkpoint onto
    `ax`. Returns (title_text, title_color) describing solve status, for callers that want a
    title -- the standalone frame exporter doesn't."""
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
    return title_text, title_color


def plot_checkpoint(checkpoint_ms: int, meta: dict, obstacles) -> plt.Figure:
    fig, axes = plt.subplots(1, len(TOKENS), figsize=(15.0, 5.6))
    for ax, token in zip(axes, TOKENS):
        title_text, title_color = draw_panel(ax, token, checkpoint_ms, meta, obstacles)
        finish_axes(ax, meta)
        ax.set_title(f"{TOKEN_LABELS.get(token, token)}\n{title_text}", fontsize=9, color=title_color)

    fig.suptitle(f"Tree + solution trajectory @ {checkpoint_ms}ms — {ENV_NAME} (Model 1, coarse/large)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def save_frame(token: str, checkpoint_ms: int, meta: dict, obstacles, frames_dir: str) -> str:
    """One panel, standalone -- no title, axis labels, ticks, or grid, just the workspace square
    with its content, cropped tight so the saved image is exactly that square."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    draw_panel(ax, token, checkpoint_ms, meta, obstacles)
    ax.set_xlim(meta["W_MIN"], meta["W_MAX"])
    ax.set_ylim(meta["W_MIN"], meta["W_MAX"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    path = os.path.join(frames_dir, f"{ENV_NAME}_{token}_t{checkpoint_ms}ms.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path


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
    frames_dir = os.path.join(OUT_DIR, "frames")
    frame_checkpoints = [c for c in FRAME_CHECKPOINTS_MS if c in checkpoints]
    missing_frame_checkpoints = [c for c in FRAME_CHECKPOINTS_MS if c not in checkpoints]
    if missing_frame_checkpoints:
        print(f"[warn] FRAME_CHECKPOINTS_MS {missing_frame_checkpoints} not found on disk -- skipping those.")
    if frame_checkpoints:
        os.makedirs(frames_dir, exist_ok=True)

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

        if checkpoint_ms in frame_checkpoints:
            for token in TOKENS:
                frame_path = save_frame(token, checkpoint_ms, meta, obstacles, frames_dir)
                print(f"  Wrote frame {frame_path}")


if __name__ == "__main__":
    main()
