"""Tree growth ANIMATION: the same three-panel (Kino-PAX / Kino-PAX+ / KinoPax*) top-down tree +
solution-trajectory render as plots/tree_checkpoint.py, just played back as a movie instead of a
handful of static frames -- reuses that script's own drawing functions directly (draw_panel,
finish_axes, draw_start_goal, compute_color_vmax, etc.), imported and pointed at this script's own
data directory, so the actual rendering logic (which nodes/edges get drawn, obstacle/tree/
trajectory styling, sparsification handling, "stopped early" labeling) never diverges between the
static and animated versions.

Input (in DATA_DIR, written by examples/gpu/tree_growth_animation_dump.cu via
scripts/run_tree_growth_animation_dump.sh -- same file formats as tree_checkpoint.py's own input,
just many more checkpoints):
    {env}_{token}_t{ms}ms_tree.csv   one file per (token, checkpoint_ms), every 30ms out to 3000ms
    {env}_{token}_t{ms}ms_traj.csv
    {env}_checkpoint_stats.csv
    meta.csv

Renders one animation (MP4 if ffmpeg is available, otherwise falls back to an animated GIF) that
steps through every checkpoint actually found on disk, in order, three panels per frame exactly as
plot_checkpoint() in tree_checkpoint.py draws them.

Edit DATA_DIR / OUT_DIR / OBSTACLES_PATH / ENV_NAME / FPS below if your data/output folders, the
environment being plotted, or the animation's playback speed need to change, then run:
    python plots/tree_growth_animation.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
# EDIT THESE if your data/output folders, the environment being plotted, or playback speed move.
# ================================================================================================
DATA_DIR = os.path.join(PLOTS_DIR, "DATA", "tree_animation", "TreeGrowthAnimation")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "tree_growth_animation")
ENV_NAME = "zigzag"
FPS = 20  # 100 checkpoints @ 20fps = 5s animation; raise/lower to change playback speed only

# Point tree_checkpoint.py's own module-level constants at THIS data/output/env -- every drawing
# function it defines (draw_panel, finish_axes, draw_obstacles, draw_tree, draw_trajectory,
# draw_start_goal, compute_color_vmax, load_checkpoint_stats, discover_checkpoints, ...) reads
# these as module globals, so this is enough to reuse them completely unchanged rather than
# duplicating any plotting logic here.
tc.DATA_DIR = DATA_DIR
tc.ENV_NAME = ENV_NAME


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

    checkpoints = tc.discover_checkpoints(tc.DATA_DIR, tc.ENV_NAME, tc.TOKENS)
    if not checkpoints:
        raise SystemExit(f"No checkpoint tree files found under {tc.DATA_DIR!r} for env={tc.ENV_NAME!r}.")
    print(f"Checkpoints found: {len(checkpoints)} (from {checkpoints[0]}ms to {checkpoints[-1]}ms)")

    # Anchor each token's node-color scale (see tc.compute_color_vmax) to the LAST checkpoint
    # found rather than tc.COLOR_SCALE_CHECKPOINT_MS's fixed 1000ms -- for a full-length animation
    # we want a token's color gradient to span its entire growth, ending at its true final color
    # on the last frame, not saturate partway through. Same function, same per-token-ceiling
    # rationale as tree_checkpoint.py -- just anchored at the end of the animation instead of a
    # fixed mid-run point.
    color_vmax_by_token = tc.compute_color_vmax(tc.DATA_DIR, tc.ENV_NAME, tc.TOKENS, checkpoints[-1])
    print("Node color scale (each token's own idx 0 -> its size at the final checkpoint):")
    for token in tc.TOKENS:
        print(f"  {tc.TOKEN_LABELS.get(token, token)}: 0 -> {color_vmax_by_token[token]:.0f}")

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, len(tc.TOKENS), figsize=(15.0, 5.6))

    def render_frame(checkpoint_ms: int) -> None:
        for ax, token in zip(axes, tc.TOKENS):
            ax.clear()
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
        checkpoint_ms = checkpoints[frame_idx]
        if frame_idx % 20 == 0:
            print(f"  Rendering frame {frame_idx + 1}/{len(checkpoints)} (t={checkpoint_ms}ms)...")
        render_frame(checkpoint_ms)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(checkpoints), blit=False)

    base_name = f"tree_growth_animation_{tc.ENV_NAME}"
    mp4_path = os.path.join(OUT_DIR, f"{base_name}.mp4")
    try:
        writer = animation.FFMpegWriter(fps=FPS, bitrate=4000)
        anim.save(mp4_path, writer=writer, dpi=150)
        print(f"\nWrote {mp4_path}")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[warn] ffmpeg unavailable ({exc}); falling back to an animated GIF (larger file, slower to open).")
        gif_path = os.path.join(OUT_DIR, f"{base_name}.gif")
        writer = animation.PillowWriter(fps=FPS)
        anim.save(gif_path, writer=writer, dpi=150)
        print(f"\nWrote {gif_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
