"""Environment viewer: isometric 3D view of an obstacle set, with the Model 1 start/goal points
marked and (if one has been dropped in) a sample solved trajectory overlaid.

Python port of scripts/view_environment.m (kept for reference, MATLAB-only) -- renders House,
Narrow(Passage), and Windows (zigzag)'s obstacle CSV as translucent 3D boxes, one color per
"wall" (grouped by y-extent, so distinct walls read as distinct colors instead of all obstacles
blurring into one tint), true isometric projection (orthographic, elev/azim below), with a dashed
straight start->goal line for scale. The .m original's second figure (per-wall 2D face slices,
for checking exactly where a narrow passage's opening is) is NOT ported here -- this script's
whole job, per request, is the isometric view; that diagnostic is still MATLAB-only if needed.

Start/goal marker styling matches plots/tree_checkpoint.py's own convention (start = magenta/pink
circle, goal = a larger green circle) instead of the .m original's green circle / red star --
kept in sync so the two scripts read as one consistent visual language across this whole folder.

TRAJECTORY OVERLAY: if TRAJECTORY_DIR (below) has a "<env>_trajectory.csv" for a given
environment (columns must include x,y,z -- extra columns are ignored), that path is drawn through
the boxes in the same dark-outline-plus-bright-green style as tree_checkpoint.py's own solved
trajectories. TRAJECTORY_DIR is created (empty) on first run if it doesn't exist yet -- drop CSVs
in by hand, or generate one per environment with scripts/run_sample_trajectories.sh (builds and
runs the existing TreeCheckpointDump tool once per environment and pulls out CountingStars's own
solved path). An environment with no matching CSV yet just renders without one -- not an error.

Edit ENVIRONMENTS / OBSTACLES_DIR / TRAJECTORY_DIR / OUT_DIR below, then run:
    python plots/view_environment.py
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys

import matplotlib


def _ensure_working_mpl_toolkits() -> None:
    """On at least one dev machine this was written on, a SECOND, broken mpl_toolkits install
    (missing its matplotlib.libs DLL dir, so its own __init__.py raises on import) sits on
    sys.path ahead of the real one bundled with the matplotlib actually in use -- a system-Python
    packaging quirk unrelated to this repo, not something to fix by editing files outside it.
    Rather than let that break every script here that needs 3D (mplot3d), explicitly resolve
    "mpl_toolkits" from the SAME site-packages directory matplotlib itself loaded from, and
    register that as the "mpl_toolkits" module before anything imports a submodule of it.

    MUST run before `matplotlib.pyplot` (or anything else that imports `matplotlib.projections`)
    is imported: that module's own top-level code does `from mpl_toolkits.mplot3d import Axes3D`
    exactly once, in a try/except that silently sets Axes3D = None and skips registering the "3d"
    projection at all if that import fails -- and it's cached in sys.modules after that first
    (successful-or-not) attempt, so fixing mpl_toolkits AFTER the fact doesn't retroactively
    register anything; matplotlib.pyplot must see a working mpl_toolkits on its ONE attempt."""
    if "mpl_toolkits" in sys.modules:
        try:
            import mpl_toolkits.mplot3d  # noqa: F401
            return  # already fine
        except Exception:
            del sys.modules["mpl_toolkits"]
    matplotlib_site_dir = os.path.dirname(os.path.dirname(matplotlib.__file__))
    spec = importlib.machinery.PathFinder.find_spec("mpl_toolkits", [matplotlib_site_dir])
    if spec is None:
        return  # fall through; the normal import below will surface whatever error applies
    module = importlib.util.module_from_spec(spec)
    sys.modules["mpl_toolkits"] = module
    if spec.loader is not None:  # None for a plain namespace package -- nothing to execute
        spec.loader.exec_module(module)


_ensure_working_mpl_toolkits()

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import env_display_name  # noqa: E402

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PLOTS_DIR)

# ================================================================================================
# EDIT THESE if your obstacle/trajectory/output folders or which environments to render move.
# ================================================================================================
OBSTACLES_DIR = os.path.join(REPO_ROOT, "include", "config", "obstacles")
TRAJECTORY_DIR = os.path.join(PLOTS_DIR, "DATA", "environment_trajectories")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "environment_views")
ENVIRONMENTS = ["house", "narrowPassage", "zigzag"]  # on-disk folder names under OBSTACLES_DIR

# Model 1 start/goal, workspace [0,1]^3 -- identical across every environment (see e.g.
# examples/gpu/paper_benchmark_v2.cu's h_initial/h_goal setup); NOT environment-specific.
START_PT = (0.10, 0.08, 0.05)
GOAL_PT = (0.80, 0.95, 0.90)
WS_MIN, WS_MAX = 0.0, 1.0

FACE_ALPHA = 0.28
START_COLOR = "#f716ff"   # matches tree_checkpoint.py's own start marker
GOAL_COLOR = "#00a51e"    # matches tree_checkpoint.py's own goal marker
TRAJ_OUTLINE_COLOR = "#008533"
TRAJ_COLOR = "#16ff16"
TRAJ_LINEWIDTH = 2.6

# True isometric projection: elev = arctan(1/sqrt(2)) so all three axes foreshorten equally.
ISO_ELEV = 35.264
ISO_AZIM = -45


def load_obstacles(path: str) -> np.ndarray:
    obstacles = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if obstacles.shape[1] != 6:
        raise ValueError(f"{path}: expected 6 columns (xmin,ymin,zmin,xmax,ymax,zmax), "
                          f"got {obstacles.shape[1]}")
    return obstacles


def group_walls(obstacles: np.ndarray) -> np.ndarray:
    """Group index per obstacle box, by its (ymin, ymax) extent -- each distinct extent is one
    "wall" (direct port of the .m original's `unique(O(:,[2 5]),'rows')`)."""
    y_extent = obstacles[:, [1, 4]]
    _uniq, inverse = np.unique(y_extent, axis=0, return_inverse=True)
    return inverse.ravel()


def _box_faces(lo: tuple, hi: tuple) -> list:
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


def draw_box(ax, lo: tuple, hi: tuple, color, alpha: float) -> None:
    poly = Poly3DCollection(_box_faces(lo, hi), facecolor=color, edgecolor="#333344",
                             linewidths=0.4, alpha=alpha)
    ax.add_collection3d(poly)


def draw_trajectory(ax, traj: pd.DataFrame) -> None:
    ax.plot3D(traj["x"], traj["y"], traj["z"], "-", color=TRAJ_OUTLINE_COLOR,
               linewidth=TRAJ_LINEWIDTH + 2, zorder=4)
    ax.plot3D(traj["x"], traj["y"], traj["z"], "-", color=TRAJ_COLOR,
               linewidth=TRAJ_LINEWIDTH, zorder=5)


def draw_start_goal(ax) -> None:
    ax.scatter([START_PT[0]], [START_PT[1]], [START_PT[2]], s=80, marker="o",
               facecolors=START_COLOR, edgecolors="black", linewidths=0.8, depthshade=False, zorder=6)
    ax.scatter([GOAL_PT[0]], [GOAL_PT[1]], [GOAL_PT[2]], s=400, marker="o",
               facecolors=GOAL_COLOR, edgecolors="black", linewidths=0.8, depthshade=False, zorder=6)
    ax.plot3D([START_PT[0], GOAL_PT[0]], [START_PT[1], GOAL_PT[1]], [START_PT[2], GOAL_PT[2]],
               "k--", linewidth=1.0, zorder=3)


def plot_environment(env: str, obstacles: np.ndarray, traj: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(9.0, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")

    groups = group_walls(obstacles)
    n_wall = int(groups.max()) + 1 if len(groups) else 0
    wall_cmap = plt.get_cmap("tab10" if n_wall <= 10 else "tab20")
    for i, row in enumerate(obstacles):
        color = wall_cmap(groups[i] / max(n_wall - 1, 1))
        draw_box(ax, tuple(row[:3]), tuple(row[3:]), color, FACE_ALPHA)

    if traj is not None:
        draw_trajectory(ax, traj)
    draw_start_goal(ax)

    ax.set_xlim(WS_MIN, WS_MAX)
    ax.set_ylim(WS_MIN, WS_MAX)
    ax.set_zlim(WS_MIN, WS_MAX)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=ISO_ELEV, azim=ISO_AZIM)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=START_COLOR,
               markeredgecolor="black", markersize=8, label="start"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=GOAL_COLOR,
               markeredgecolor="black", markersize=13, label="goal"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=1.0, label="straight line"),
    ]
    if traj is not None:
        handles.append(Line2D([0], [0], color=TRAJ_COLOR, linewidth=TRAJ_LINEWIDTH,
                               label="sample trajectory (CountingStars)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    ax.set_title(f"{env_display_name(env)} — {len(obstacles)} obstacles, {n_wall} walls",
                 fontweight="bold")
    return fig


def main() -> None:
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    readme_path = os.path.join(TRAJECTORY_DIR, "README.txt")
    if not os.path.isfile(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                "Drop a per-environment sample trajectory CSV here to have it overlaid on that\n"
                "environment's isometric view (plots/view_environment.py picks it up automatically\n"
                "-- no code changes needed).\n\n"
                "Expected filename: <env>_trajectory.csv, one of:\n"
                "  house_trajectory.csv\n"
                "  narrowPassage_trajectory.csv\n"
                "  zigzag_trajectory.csv\n\n"
                "Required columns: x,y,z (extra columns, e.g. vx,vy,vz,cost, are ignored).\n\n"
                "Generate these with scripts/run_sample_trajectories.sh, which builds and runs the\n"
                "existing TreeCheckpointDump tool once per environment and pulls out CountingStars's\n"
                "own solved trajectory -- or drop in any CSV with those columns by hand.\n"
            )
        print(f"Wrote {readme_path}")

    os.makedirs(OUT_DIR, exist_ok=True)

    for env in ENVIRONMENTS:
        obstacles_path = os.path.join(OBSTACLES_DIR, env, "obstacles.csv")
        if not os.path.isfile(obstacles_path):
            print(f"[warn] no obstacle file for {env!r} at {obstacles_path!r} -- skipping.")
            continue
        obstacles = load_obstacles(obstacles_path)
        print(f"{env_display_name(env)}: loaded {len(obstacles)} obstacles from {obstacles_path}")

        traj_path = os.path.join(TRAJECTORY_DIR, f"{env}_trajectory.csv")
        traj = None
        if os.path.isfile(traj_path):
            traj = pd.read_csv(traj_path)
            missing = {"x", "y", "z"} - set(traj.columns)
            if missing:
                print(f"  [warn] {traj_path} is missing column(s) {missing} -- skipping trajectory overlay.")
                traj = None
            else:
                print(f"  Overlaying trajectory from {traj_path} ({len(traj)} points)")
        else:
            print(f"  No trajectory found at {traj_path} -- drop one in to have it drawn here.")

        fig = plot_environment(env, obstacles, traj)
        base_name = f"environment_{env}"
        png_path = os.path.join(OUT_DIR, f"{base_name}.png")
        svg_path = os.path.join(OUT_DIR, f"{base_name}.svg")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {png_path} + .svg")


if __name__ == "__main__":
    main()
