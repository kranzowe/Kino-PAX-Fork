"""Tree-growth visualization for Chain Reaction and Fission Reaction vs CountingStars (KinoPax#): the existing
tree_growth_animation.py / tree_checkpoint.py pipeline, pointed at the three-planner dump written by
scripts/paper/run_chain_reaction_benchmark.sh --viz
(TreeGrowthAnimationDump --planners=CountingStars,ChainReaction,FissionReaction).
All drawing and data loading is reused unchanged; this wrapper only swaps the planner tokens/labels,
the data and output folders, the figure-title note and the playback speed.

The dump covers only the FIRST 100 ms of planning, one checkpoint every 10 ms (plus this animation's
synthetic empty t=0 frame), with a small fraction of each tree saved. Fission Reaction is shown at ONE
branching factor, FISSION_VIZ_K (must equal VIZ_FISSION_K in the runner).

Input (copy build/Data/Viz/ChainReaction/ here):
    plots/DATA/tree_animation_chain_reaction/TreeGrowthAnimation/
        {env}_{CountingStars|ChainReaction|FissionReaction_k<K>}_t{ms}ms_{tree,traj}.csv
    ... plus meta.csv and {env}_checkpoint_stats.csv

    python plots/paper/chain_reaction_growth_animation.py            # animation (MP4, or GIF without ffmpeg)
    python plots/paper/chain_reaction_growth_animation.py --static   # per-checkpoint PNG/SVG panels instead

A planner whose run ended early (Chain/Fission Reaction fill the tree, or the frontier empties) simply has
no file for the later checkpoints; the panels label that "stopped early" from the stats file. Fission
Reaction grows exponentially, so it usually stops inside the 100 ms window.
"""
from __future__ import annotations

import os
import sys

import tree_checkpoint as tc
import tree_growth_animation as tga

PLOTS_DIR = tga.PLOTS_DIR
DATA_DIR = os.path.join(PLOTS_DIR, "DATA", "tree_animation_chain_reaction", "TreeGrowthAnimation")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "chain_reaction_animation")
FISSION_VIZ_K = 4   # must equal VIZ_FISSION_K in scripts/paper/run_chain_reaction_benchmark.sh
TOKEN_LABELS = {
    "CountingStars": "KinoPax#",
    "ChainReaction": "Chain Reaction",
    f"FissionReaction_k{FISSION_VIZ_K}": f"Fission Reaction (k={FISSION_VIZ_K})",
}
MODEL_NOTE = "Model 2 Dubins, coarse/large"

# 11 frames (t = 0, 10, ..., 100 ms) would play in ~0.5 s at tree_growth_animation's 20 fps.
FPS = 4

# --static only: checkpoints that exist on the 10 ms grid, and the one the node color scale anchors to.
STATIC_FRAME_CHECKPOINTS_MS = (20, 50, 100)
STATIC_COLOR_SCALE_MS = 100


def main() -> None:
    tc.TOKEN_LABELS = dict(TOKEN_LABELS)
    tc.TOKENS = list(TOKEN_LABELS)
    tc.DATA_DIR = DATA_DIR
    tc.MODEL_NOTE = MODEL_NOTE
    tga.DATA_DIR = DATA_DIR
    tga.OUT_DIR = OUT_DIR
    tga.FPS = FPS

    if "--static" in sys.argv:
        tc.OUT_DIR = os.path.join(OUT_DIR, "static")
        tc.FRAME_CHECKPOINTS_MS = STATIC_FRAME_CHECKPOINTS_MS
        tc.COLOR_SCALE_CHECKPOINT_MS = STATIC_COLOR_SCALE_MS
        tc.main()
    else:
        tga.main()


if __name__ == "__main__":
    main()
