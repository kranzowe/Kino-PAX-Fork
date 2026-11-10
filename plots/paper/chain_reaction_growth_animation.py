"""Tree-growth visualization for Chain Reaction vs CountingStars (KinoPax#): the existing
tree_growth_animation.py / tree_checkpoint.py pipeline, pointed at the two-planner dump written by
scripts/paper/run_chain_reaction_benchmark.sh --viz (TreeGrowthAnimationDump --planners=CountingStars,ChainReaction).
All drawing and data loading is reused unchanged; this wrapper only swaps the planner tokens/labels,
the data and output folders, and the figure-title note.

Input (copy build/Data/Viz/ChainReaction/ here):
    plots/DATA/tree_animation_chain_reaction/TreeGrowthAnimation/{env}_{ChainReaction|CountingStars}_t{ms}ms_{tree,traj}.csv
    ... plus meta.csv and {env}_checkpoint_stats.csv

    python plots/paper/chain_reaction_growth_animation.py            # animation (MP4, or GIF without ffmpeg)
    python plots/paper/chain_reaction_growth_animation.py --static   # per-checkpoint PNG/SVG panels instead

A planner whose run ended early (Chain Reaction fills the tree, or its frontier empties) simply has no
file for the later checkpoints; the panels label that "stopped early" from the stats file.
"""
from __future__ import annotations

import os
import sys

import tree_checkpoint as tc
import tree_growth_animation as tga

PLOTS_DIR = tga.PLOTS_DIR
DATA_DIR = os.path.join(PLOTS_DIR, "DATA", "tree_animation_chain_reaction", "TreeGrowthAnimation")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "chain_reaction_animation")
TOKEN_LABELS = {"CountingStars": "KinoPax#", "ChainReaction": "Chain Reaction"}
MODEL_NOTE = "Model 2 Dubins, coarse/large"

# --static only: checkpoints that exist on the 30ms grid, and the one the node color scale anchors to.
STATIC_FRAME_CHECKPOINTS_MS = (300, 990, 3000)
STATIC_COLOR_SCALE_MS = 990


def main() -> None:
    tc.TOKEN_LABELS = dict(TOKEN_LABELS)
    tc.TOKENS = list(TOKEN_LABELS)
    tc.DATA_DIR = DATA_DIR
    tc.MODEL_NOTE = MODEL_NOTE
    tga.DATA_DIR = DATA_DIR
    tga.OUT_DIR = OUT_DIR

    if "--static" in sys.argv:
        tc.OUT_DIR = os.path.join(OUT_DIR, "static")
        tc.FRAME_CHECKPOINTS_MS = STATIC_FRAME_CHECKPOINTS_MS
        tc.COLOR_SCALE_CHECKPOINT_MS = STATIC_COLOR_SCALE_MS
        tc.main()
    else:
        tga.main()


if __name__ == "__main__":
    main()
