"""Tree growth ANIMATION: the same three-panel (Kino-PAX / Kino-PAX+ / KinoPax*) tree +
solution-trajectory render as plots/tree_checkpoint.py, played back as a movie. VIEW_MODE below
toggles between plots/tree_checkpoint.py's own top-down X-Y view (default) and a true isometric 3D
view (same fixed ISO_ELEV/ISO_AZIM/orthographic convention as plots/view_environment.py) that uses
the z-column the top-down view discards. Node/edge/trajectory/obstacle DATA and styling (turbo-by-
insertion-order coloring, sparsification handling, "stopped early" labeling, start/goal marker
colors) come from tree_checkpoint.py's own data-loading helpers either way (load_meta,
load_checkpoint_stats, discover_checkpoints, compute_color_vmax, _node_count_text,
_last_reached_checkpoint, TOKEN_LABELS/TOKENS/color constants) -- for VIEW_MODE="top_down" this
file also reuses tree_checkpoint.py's own 2D drawing functions (draw_panel, finish_axes,
draw_obstacles, draw_workspace_square, draw_start_goal) directly; for VIEW_MODE="isometric" it
uses this file's own 3D counterparts instead (draw_obstacles_3d, draw_tree_3d, draw_trajectory_3d,
draw_start_goal_3d, finish_axes_3d, draw_panel_3d), since tree_checkpoint.py's own drawing
functions are hardwired to 2D (Rectangle patches, plain ax.plot/ax.scatter -- no z at all).

Input (in DATA_DIR, written by examples/gpu/tree_growth_animation_dump.cu via
scripts/run_tree_growth_animation_dump.sh -- same file formats as tree_checkpoint.py's own input,
just many more checkpoints):
    {env}_{token}_t{ms}ms_tree.csv   one file per (token, checkpoint_ms), every 30ms out to 3000ms
    {env}_{token}_t{ms}ms_traj.csv
    {env}_checkpoint_stats.csv
    meta.csv

The animation is prepended with one synthetic t=0ms frame (empty tree, no data file needed -- the
dump's first real checkpoint is 30ms) showing just the environment and start/goal, so the movie
visibly starts from nothing before the first real checkpoint appears, then steps through every
checkpoint actually found on disk in order, three panels per frame.

Renders one animation: MP4 if ffmpeg is available (directly, or via the imageio-ffmpeg pip
package's bundled binary), otherwise falls back to an animated GIF.

Edit DATA_DIR / OUT_DIR / OBSTACLES_PATH / ENV_NAME / FPS / ISO_ELEV / ISO_AZIM below if your
data/output folders, the environment being plotted, the animation's playback speed, or the camera
angle need to change, then run:
    python plots/tree_growth_animation.py
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys


def _ensure_working_mpl_toolkits() -> None:
    """Same fix as plots/view_environment.py's own helper (copied verbatim, see its comment for
    the full explanation): on at least one dev machine this repo is used on, a second, broken
    mpl_toolkits install shadows the real one bundled with the matplotlib actually in use, which
    silently disables the "3d" projection ("Unable to import Axes3D" warning) the first time
    matplotlib.pyplot is imported -- and that failure is cached, so it must be fixed before that
    first import, not after. This animation is the first script in plots/ that actually needs
    mplot3d (tree_checkpoint.py's own top-down view never did), so it needs this guard too."""
    if "mpl_toolkits" in sys.modules:
        try:
            import mpl_toolkits.mplot3d  # noqa: F401
            return  # already fine
        except Exception:
            del sys.modules["mpl_toolkits"]
    import matplotlib
    matplotlib_site_dir = os.path.dirname(os.path.dirname(matplotlib.__file__))
    spec = importlib.machinery.PathFinder.find_spec("mpl_toolkits", [matplotlib_site_dir])
    if spec is None:
        return  # fall through; the normal import below will surface whatever error applies
    module = importlib.util.module_from_spec(spec)
    sys.modules["mpl_toolkits"] = module
    if spec.loader is not None:  # None for a plain namespace package -- nothing to execute
        spec.loader.exec_module(module)


_ensure_working_mpl_toolkits()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import pandas as pd

try:
    # imageio-ffmpeg bundles a static ffmpeg binary -- pip install imageio-ffmpeg gives MP4 output
    # with no system ffmpeg install/PATH setup needed. Silently no-op if it isn't installed; the
    # save step below still falls back to an animated GIF in that case.
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

import tree_checkpoint as tc

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================
# EDIT THESE if your data/output folders, the environment being plotted, playback speed, or the
# camera angle move.
# ================================================================================================
DATA_DIR = os.path.join(PLOTS_DIR, "DATA", "tree_animation_30_2", "TreeGrowthAnimation")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "tree_growth_animation")
ENV_NAME = "zigzag"
FPS = 20  # 100 checkpoints @ 20fps = 5s animation; raise/lower to change playback speed only
SAVE_DPI = 100  # lower = faster encode; 150+ only matters if you're zooming in on a still frame

# "top_down" = tree_checkpoint.py's own flat X-Y view (default). "isometric" = the 3D view added
# for this animation (see module docstring) -- also generally slower to render (3D depth-sorting
# adds real cost on top of the point/edge count, see RENDER_NODE_FRACTION below).
VIEW_MODE = "top_down"

# True isometric projection: elev = arctan(1/sqrt(2)) so all three axes foreshorten equally --
# identical fixed angle to plots/view_environment.py's own ISO_ELEV/ISO_AZIM, so this animation's
# camera matches that static figure's convention. Only used when VIEW_MODE == "isometric".
ISO_ELEV = 35.264
ISO_AZIM = -45

# Additional, RENDER-TIME-ONLY subsample on top of whatever fraction of the tree got saved to
# disk (see tree_growth_animation_dump.cu's TREE_LEAF_SAMPLE_FRAC) -- a free lever for animation
# speed since it needs no GPU rerun, just a faster Python pass. 1.0 draws every node/edge that
# was saved; lower it (e.g. 0.3-0.5) if rendering is still too slow after a leaner dump.
RENDER_NODE_FRACTION = 1.0

# Point tree_checkpoint.py's own module-level constants at THIS data/output/env, and its
# NODE_FRACTION at the render-time knob above -- the 3D drawing functions below read
# tc.NODE_FRACTION/tc.EDGE_FRACTION/tc.NODE_SIZE/tc.TRAJ_LINEWIDTH and the color constants
# directly off `tc` so this file never duplicates those values.
tc.DATA_DIR = DATA_DIR
tc.ENV_NAME = ENV_NAME
tc.NODE_FRACTION = RENDER_NODE_FRACTION

OBSTACLE_ALPHA_3D = 0.18  # lower than tree_checkpoint's top-down OBSTACLE_ALPHA (0.5) -- a 3D
                          # translucent box stacks visually with whatever tree/edges are behind
                          # it, so a lighter alpha keeps the tree readable through it.


def _box_faces(lo: tuple, hi: tuple) -> list:
    """Same 6-quad box decomposition as view_environment.py's own _box_faces."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # bottom
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # front (y=y0)
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # back (y=y1)
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # left (x=x0)
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # right (x=x1)
    ]


def draw_obstacles_3d(ax, obstacles: np.ndarray) -> None:
    for xmin, ymin, zmin, xmax, ymax, zmax in obstacles:
        poly = Poly3DCollection(_box_faces((xmin, ymin, zmin), (xmax, ymax, zmax)),
                                 facecolor="#000000", edgecolor="#0e0000",
                                 linewidths=0.3, alpha=OBSTACLE_ALPHA_3D)
        ax.add_collection3d(poly)


def draw_tree_3d(ax, tree: pd.DataFrame, color_vmax: float) -> None:
    """3D counterpart of tree_checkpoint.draw_tree: same NODE_FRACTION/EDGE_FRACTION subsampling
    (read live off `tc`, so RENDER_NODE_FRACTION above still applies) and the same turbo-by-
    insertion-order node coloring, just plotted with z instead of a top-down projection that threw
    it away. Edges are a single Line3DCollection (one artist for every edge, like the 2D version's
    single NaN-separated Line2D) rather than one ax.plot call per edge, for the same reason
    view_environment.py's own trajectory fix used Line3DCollection: a single 3D line artist with
    computed_zorder=True still gets real per-segment depth sorting against the obstacle boxes."""
    n = len(tree)
    if n == 0:
        return

    parent = tree["parent"].to_numpy()
    x, y, z = tree["x"].to_numpy(), tree["y"].to_numpy(), tree["z"].to_numpy()

    n_node = max(1, round(tc.NODE_FRACTION * n))
    ridx = np.unique(np.round(np.linspace(0, n - 1, min(n_node, n))).astype(int))

    edge_src = ridx[parent[ridx] >= 0]
    if tc.EDGE_FRACTION < 1.0 and len(edge_src) > 0:
        n_edge = max(1, round(tc.EDGE_FRACTION * len(edge_src)))
        if n_edge < len(edge_src):
            pick = np.unique(np.round(np.linspace(0, len(edge_src) - 1, n_edge)).astype(int))
            edge_src = edge_src[pick]
    if len(edge_src) > 0:
        parent_rows = parent[edge_src]
        segments = np.stack([
            np.stack([x[edge_src], y[edge_src], z[edge_src]], axis=1),
            np.stack([x[parent_rows], y[parent_rows], z[parent_rows]], axis=1),
        ], axis=1)
        ax.add_collection3d(Line3DCollection(segments, colors="#b8b8bf", linewidths=0.3, alpha=0.5))

    ax.scatter(x[ridx], y[ridx], z[ridx], s=tc.NODE_SIZE, c=tree["idx"].to_numpy()[ridx],
               cmap="turbo", vmin=0, vmax=color_vmax, alpha=0.8, depthshade=False)


def draw_trajectory_3d(ax, traj: pd.DataFrame) -> None:
    """3D counterpart of tree_checkpoint.draw_trajectory -- same dark-outline-plus-bright-green
    styling, built as per-segment Line3DCollections (like view_environment.py's own occlusion-
    correct trajectory) so the path occludes/gets occluded by obstacle boxes correctly instead of
    getting one coarse depth value for its whole length."""
    points = traj[["x", "y", "z"]].to_numpy()
    if len(points) < 2:
        return
    segments = np.stack([points[:-1], points[1:]], axis=1)
    ax.add_collection3d(Line3DCollection(segments, colors=tc.TRAJ_COLOR, linewidths=tc.TRAJ_LINEWIDTH + 6))
    ax.add_collection3d(Line3DCollection(segments, colors="#4cff46", linewidths=tc.TRAJ_LINEWIDTH))


def draw_start_goal_3d(ax, meta: dict) -> None:
    ax.scatter([meta["start_x"]], [meta["start_y"]], [meta["start_z"]], s=80, marker="o",
               facecolors="#f716ff", edgecolors="black", linewidths=0.8, depthshade=False, zorder=6)
    ax.scatter([meta["goal_x"]], [meta["goal_y"]], [meta["goal_z"]], s=400, marker="o",
               facecolors=tc.GOAL_COLOR, edgecolors="black", linewidths=0.8, depthshade=False, zorder=6)


def finish_axes_3d(ax, meta: dict) -> None:
    lo, hi = meta["W_MIN"], meta["W_MAX"]
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_zlim(lo, hi)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ISO_ELEV, azim=ISO_AZIM)


def draw_panel_3d(ax, token: str, checkpoint_ms: int, meta: dict, obstacles, color_vmax_by_token: dict,
                   stats: dict) -> tuple:
    """3D counterpart of tree_checkpoint.draw_panel -- same file paths/missing-file/"stopped
    early" logic (reusing tc._node_count_text/tc._last_reached_checkpoint directly, since those
    are pure data lookups with nothing 2D-specific about them), just calling the _3d drawing
    functions above instead of tree_checkpoint's own 2D ones."""
    tree_path = os.path.join(tc.DATA_DIR, f"{tc.ENV_NAME}_{token}_t{checkpoint_ms}ms_tree.csv")
    traj_path = os.path.join(tc.DATA_DIR, f"{tc.ENV_NAME}_{token}_t{checkpoint_ms}ms_traj.csv")

    if obstacles is not None:
        draw_obstacles_3d(ax, obstacles)

    if not os.path.isfile(tree_path):
        print(f"  [warn] missing tree file: {tree_path}")
        last_reached = tc._last_reached_checkpoint(stats, token)
        if last_reached >= 0 and last_reached < checkpoint_ms:
            title_text = f"no data (stopped early, after t={last_reached}ms)"
        else:
            title_text = "no data"
        title_color = tc.NODATA_COLOR
    else:
        tree = pd.read_csv(tree_path)
        draw_tree_3d(ax, tree, color_vmax_by_token[token])
        node_count_text = tc._node_count_text(stats, token, checkpoint_ms, tree)
        if os.path.isfile(traj_path):
            traj = pd.read_csv(traj_path)
        else:
            print(f"  [warn] missing trajectory file: {traj_path}")
            traj = pd.DataFrame()
        if len(traj) > 0:
            draw_trajectory_3d(ax, traj)
            title_text = f"{node_count_text} | SOLVED, cost={traj['cost'].iloc[-1]:.4f}"
            title_color = tc.SOLVED_COLOR
        else:
            title_text = f"{node_count_text} | no solution yet"
            title_color = tc.UNSOLVED_COLOR

    draw_start_goal_3d(ax, meta)
    return title_text, title_color


def draw_empty_panel_3d(ax, meta: dict, obstacles) -> tuple:
    """The synthetic t=0ms frame: nothing has been planned yet, so just the environment and
    start/goal -- no tree file exists at 0ms (the dump's first real checkpoint is 30ms), so this
    is drawn directly rather than routed through draw_panel_3d's missing-file handling, which
    would otherwise misreport this expected, always-true state as a "no data"/warning."""
    if obstacles is not None:
        draw_obstacles_3d(ax, obstacles)
    draw_start_goal_3d(ax, meta)
    return "N=0 nodes | not started", tc.NODATA_COLOR


def draw_empty_panel_2d(ax, meta: dict, obstacles) -> tuple:
    """Top-down counterpart of draw_empty_panel_3d, using tree_checkpoint.py's own 2D obstacle/
    workspace/start-goal drawing functions directly."""
    if obstacles is not None:
        tc.draw_obstacles(ax, obstacles)
    tc.draw_workspace_square(ax, meta)
    tc.draw_start_goal(ax, meta)
    return "N=0 nodes | not started", tc.NODATA_COLOR


def main() -> None:
    if not os.path.isdir(tc.DATA_DIR):
        raise SystemExit(
            f"DATA_DIR does not exist: {tc.DATA_DIR!r} -- run scripts/run_tree_growth_animation_dump.sh "
            "first, or edit DATA_DIR at the top of this file."
        )
    meta = tc.load_meta(tc.DATA_DIR)
    obstacles = tc.load_obstacles(tc.OBSTACLES_PATH)
    if obstacles is None:
        print(f"[warn] no obstacle CSV found at {tc.OBSTACLES_PATH!r}; drawing trees without the environment.")
    else:
        print(f"Obstacles: {tc.OBSTACLES_PATH} ({len(obstacles)} boxes)")

    stats = tc.load_checkpoint_stats(tc.DATA_DIR, tc.ENV_NAME)
    if not stats:
        print(f"[warn] no {tc.ENV_NAME}_checkpoint_stats.csv found in {tc.DATA_DIR!r} -- true tree "
              "size/% of capacity won't be shown. Falling back to each sparsified tree CSV's own row count.")

    real_checkpoints = tc.discover_checkpoints(tc.DATA_DIR, tc.ENV_NAME, tc.TOKENS)
    if not real_checkpoints:
        raise SystemExit(f"No checkpoint tree files found under {tc.DATA_DIR!r} for env={tc.ENV_NAME!r}.")
    print(f"Checkpoints found: {len(real_checkpoints)} (from {real_checkpoints[0]}ms to {real_checkpoints[-1]}ms)")

    # 0 prepended as a synthetic first frame -- see draw_empty_panel_3d. NOT included in
    # real_checkpoints (that list drives compute_color_vmax/discovery, which are about what's
    # actually on disk), only in the animation's own playback order.
    animation_checkpoints = [0] + real_checkpoints

    # Anchor each token's node-color scale (see tc.compute_color_vmax) to the LAST checkpoint
    # found rather than tc.COLOR_SCALE_CHECKPOINT_MS's fixed 1000ms -- for a full-length animation
    # we want a token's color gradient to span its entire growth, ending at its true final color
    # on the last frame, not saturate partway through. Same function, same per-token-ceiling
    # rationale as tree_checkpoint.py -- just anchored at the end of the animation instead of a
    # fixed mid-run point.
    color_vmax_by_token = tc.compute_color_vmax(tc.DATA_DIR, tc.ENV_NAME, tc.TOKENS, real_checkpoints[-1])
    print("Node color scale (each token's own idx 0 -> its size at the final checkpoint):")
    for token in tc.TOKENS:
        print(f"  {tc.TOKEN_LABELS.get(token, token)}: 0 -> {color_vmax_by_token[token]:.0f}")

    if VIEW_MODE not in ("top_down", "isometric"):
        raise SystemExit(f"VIEW_MODE must be 'top_down' or 'isometric', got {VIEW_MODE!r}.")
    print(f"View mode: {VIEW_MODE}")

    os.makedirs(OUT_DIR, exist_ok=True)

    is_3d = VIEW_MODE == "isometric"
    fig, axes = plt.subplots(1, len(tc.TOKENS), figsize=(15.0, 5.6),
                              subplot_kw={"projection": "3d"} if is_3d else {})

    def render_frame(checkpoint_ms: int) -> None:
        for ax, token in zip(axes, tc.TOKENS):
            ax.clear()
            if is_3d:
                if checkpoint_ms == 0:
                    title_text, title_color = draw_empty_panel_3d(ax, meta, obstacles)
                else:
                    title_text, title_color = draw_panel_3d(
                        ax, token, checkpoint_ms, meta, obstacles, color_vmax_by_token, stats
                    )
                finish_axes_3d(ax, meta)
            else:
                if checkpoint_ms == 0:
                    title_text, title_color = draw_empty_panel_2d(ax, meta, obstacles)
                else:
                    title_text, title_color = tc.draw_panel(
                        ax, token, checkpoint_ms, meta, obstacles, color_vmax_by_token, stats
                    )
                tc.finish_axes(ax, meta)
            ax.set_title(f"{tc.TOKEN_LABELS.get(token, token)}\n{title_text}", fontsize=9, color=title_color)
        fig.suptitle(
            f"Tree + solution trajectory @ {checkpoint_ms}ms — {tc.ENV_NAME} (Model 1, coarse/large)",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(frame_idx: int):
        checkpoint_ms = animation_checkpoints[frame_idx]
        if frame_idx % 20 == 0:
            print(f"  Rendering frame {frame_idx + 1}/{len(animation_checkpoints)} (t={checkpoint_ms}ms)...")
        render_frame(checkpoint_ms)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(animation_checkpoints), blit=False)

    base_name = f"tree_growth_animation_{tc.ENV_NAME}"
    mp4_path = os.path.join(OUT_DIR, f"{base_name}.mp4")
    try:
        writer = animation.FFMpegWriter(fps=FPS, bitrate=4000)
        anim.save(mp4_path, writer=writer, dpi=SAVE_DPI)
        print(f"\nWrote {mp4_path}")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[warn] ffmpeg unavailable ({exc}); falling back to an animated GIF (larger file, slower to open).")
        gif_path = os.path.join(OUT_DIR, f"{base_name}.gif")
        writer = animation.PillowWriter(fps=FPS)
        anim.save(gif_path, writer=writer, dpi=SAVE_DPI)
        print(f"\nWrote {gif_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
