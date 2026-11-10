"""Environment viewer using Plotly/WebGL.

Renders House, Narrow Passage, and Zigzag obstacle CSVs as translucent 3D boxes,
with Model 1 start/goal points and an optional solved trajectory overlay.

Compared with Matplotlib mplot3d, Plotly renders the 3D scene through WebGL and
uses a real depth buffer, which makes front/behind relationships substantially
more reliable for trajectories crossing translucent obstacle geometry.

Edit ENVIRONMENTS / OBSTACLES_DIR / TRAJECTORY_DIR / OUT_DIR / INTERACTIVE below,
then run:

    python plots/view_environment.py

Static PNG/SVG export requires Kaleido:

    pip install -U kaleido

Without Kaleido the interactive HTML output still works.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import env_display_name


PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PLOTS_DIR)

# ================================================================================================
# EDIT THESE if your obstacle/trajectory/output folders or which environments to render move.
# ================================================================================================
OBSTACLES_DIR = os.path.join(REPO_ROOT, "include", "config", "obstacles")
TRAJECTORY_DIR = os.path.join(PLOTS_DIR, "DATA", "environment_trajectories")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "environment_views")
ENVIRONMENTS = ["house", "narrowPassage", "zigzag"]

# Opens each environment in a browser window/tab after writing outputs.
INTERACTIVE = False

# Model 1 start/goal, workspace [0,1]^3.
START_PT = (0.10, 0.08, 0.05)
GOAL_PT = (0.80, 0.95, 0.90)
WS_MIN, WS_MAX = 0.0, 1.0

FACE_ALPHA = 0.28
START_COLOR = "#f716ff"
GOAL_COLOR = "#00a51e"
TRAJ_OUTLINE_COLOR = "#000000"
TRAJ_COLOR = "#16ff16"
TRAJ_LINEWIDTH = 5.0
TRAJ_OUTLINE_WIDTH = 14.0

# Camera. Equal-magnitude x/y/z components give an isometric-style orthographic view.
# Signs match the old Matplotlib azim=-45 view reasonably closely.
CAMERA_EYE = dict(x=1.55, y=-1.55, z=1.55)

# Triangle indices for a rectangular box with 8 vertices.
_BOX_I = [0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 6, 4]
_BOX_J = [1, 2, 4, 2, 5, 3, 7, 5, 6, 6, 7, 7]
_BOX_K = [2, 4, 1, 5, 2, 7, 2, 6, 5, 7, 3, 3]


def load_obstacles(path: str) -> np.ndarray:
    obstacles = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if obstacles.shape[1] != 6:
        raise ValueError(
            f"{path}: expected 6 columns (xmin,ymin,zmin,xmax,ymax,zmax), "
            f"got {obstacles.shape[1]}"
        )
    return obstacles


def group_walls(obstacles: np.ndarray) -> np.ndarray:
    """Group obstacles by their (ymin, ymax) extent, matching the old renderer."""
    y_extent = obstacles[:, [1, 4]]
    _uniq, inverse = np.unique(y_extent, axis=0, return_inverse=True)
    return inverse.ravel()


def box_vertices(lo: tuple[float, float, float], hi: tuple[float, float, float]):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return (
        [x0, x1, x1, x0, x0, x1, x1, x0],
        [y0, y0, y1, y1, y0, y0, y1, y1],
        [z0, z0, z0, z0, z1, z1, z1, z1],
    )


def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def wall_colors(n_wall: int) -> list[str]:
    palette = qualitative.Plotly + qualitative.D3 + qualitative.G10 + qualitative.T10
    if n_wall <= len(palette):
        return palette[:n_wall]
    return [palette[i % len(palette)] for i in range(n_wall)]


def draw_box(fig: go.Figure, lo: tuple, hi: tuple, color: str) -> None:
    """Draw an axis-aligned box as six independently rendered flat faces.

    A single Mesh3d box is internally triangulated by Plotly. With transparency and
    lighting, the two triangles forming a rectangular face can become visibly different.
    Rendering each rectangle as its own two-triangle mesh with ambient-only lighting
    removes that diagonal/triangle shading artifact.
    """
    x0, y0, z0 = lo
    x1, y1, z1 = hi

    faces = [
        # bottom / top
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        # y-min / y-max
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        # x-min / x-max
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]

    for face in faces:
        xyz = np.asarray(face, dtype=float)
        fig.add_trace(
            go.Mesh3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=color,
                opacity=FACE_ALPHA,
                flatshading=True,
                # Ambient-only lighting makes every point on a face exactly the same color.
                lighting=dict(
                    ambient=1.0,
                    diffuse=0.0,
                    specular=0.0,
                    roughness=1.0,
                    fresnel=0.0,
                ),
                hoverinfo="skip",
                showscale=False,
                name="obstacle",
                showlegend=False,
            )
        )

    # Draw the 12 box edges explicitly.
    vertices = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xe, ye, ze = [], [], []
    for a, b in edges:
        xe += [vertices[a, 0], vertices[b, 0], None]
        ye += [vertices[a, 1], vertices[b, 1], None]
        ze += [vertices[a, 2], vertices[b, 2], None]

    fig.add_trace(
        go.Scatter3d(
            x=xe,
            y=ye,
            z=ze,
            mode="lines",
            line=dict(color="rgba(51,51,68,0.65)", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )


def draw_trajectory(fig: go.Figure, traj: pd.DataFrame) -> None:
    points = traj[["x", "y", "z"]]

    # Draw the outline first, then the bright center. Both are true 3D WebGL traces,
    # so depth testing happens against the obstacle meshes rather than via mplot3d's
    # painter-style artist sorting.
    fig.add_trace(
        go.Scatter3d(
            x=points["x"],
            y=points["y"],
            z=points["z"],
            mode="lines",
            line=dict(color=TRAJ_OUTLINE_COLOR, width=TRAJ_OUTLINE_WIDTH),
            hoverinfo="skip",
            showlegend=False,
            name="trajectory outline",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=points["x"],
            y=points["y"],
            z=points["z"],
            mode="lines",
            line=dict(color=TRAJ_COLOR, width=TRAJ_LINEWIDTH),
            hoverinfo="skip",
            showlegend=False,
            name="trajectory",
        )
    )


def draw_start_goal(fig: go.Figure) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=[START_PT[0]],
            y=[START_PT[1]],
            z=[START_PT[2]],
            mode="markers",
            marker=dict(size=7, color=START_COLOR, line=dict(color="black", width=2)),
            hoverinfo="skip",
            showlegend=False,
            name="start",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[GOAL_PT[0]],
            y=[GOAL_PT[1]],
            z=[GOAL_PT[2]],
            mode="markers",
            marker=dict(size=20, color=GOAL_COLOR, line=dict(color="black", width=2)),
            hoverinfo="skip",
            showlegend=False,
            name="goal",
        )
    )


def plot_environment(env: str, obstacles: np.ndarray, traj: pd.DataFrame | None) -> go.Figure:
    groups = group_walls(obstacles)
    n_wall = int(groups.max()) + 1 if len(groups) else 0
    colors = wall_colors(n_wall)

    fig = go.Figure()

    for i, row in enumerate(obstacles):
        draw_box(fig, tuple(row[:3]), tuple(row[3:]), colors[groups[i]])

    if traj is not None:
        draw_trajectory(fig, traj)

    draw_start_goal(fig)

    axis_common = dict(
        range=[WS_MIN, WS_MAX],
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
        title="",
        showspikes=False,
        visible=False,
    )

    fig.update_layout(
        title=dict(
            text=f"{env_display_name(env)} — {len(obstacles)} obstacles, {n_wall} walls",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        width=900,
        height=750,
        margin=dict(l=0, r=0, b=0, t=55),
        paper_bgcolor="white",
        scene=dict(
            xaxis=axis_common,
            yaxis=axis_common,
            zaxis=axis_common,
            aspectmode="cube",
            camera=dict(
                eye=CAMERA_EYE,
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
                projection=dict(type="orthographic"),
            ),
            bgcolor="white",
        ),
        showlegend=False,
    )

    return fig


def write_static_images(fig: go.Figure, png_path: str, svg_path: str) -> None:
    try:
        fig.write_image(png_path, width=900, height=750, scale=2)
        fig.write_image(svg_path, width=900, height=750)
        print(f"  Wrote {png_path} + .svg")
    except Exception as exc:
        print(
            "  [warn] Static image export failed. Plotly uses Kaleido for PNG/SVG export.\n"
            f"         Install/update it with: pip install -U kaleido\n"
            f"         Details: {exc}"
        )


def main() -> None:
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    readme_path = os.path.join(TRAJECTORY_DIR, "README.txt")
    if not os.path.isfile(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                "Drop a per-environment sample trajectory CSV here to have it overlaid.\n\n"
                "Expected filenames:\n"
                "  house_trajectory.csv\n"
                "  narrowPassage_trajectory.csv\n"
                "  zigzag_trajectory.csv\n\n"
                "Required columns: x,y,z; extra columns are ignored.\n"
            )
        print(f"Wrote {readme_path}")

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
            print(f"  No trajectory found at {traj_path} -- rendering obstacles only.")

        fig = plot_environment(env, obstacles, traj)

        base_name = f"environment_{env}"
        html_path = os.path.join(OUT_DIR, f"{base_name}.html")
        png_path = os.path.join(OUT_DIR, f"{base_name}.png")
        svg_path = os.path.join(OUT_DIR, f"{base_name}.svg")

        fig.write_html(html_path, include_plotlyjs="cdn")
        # print(f"  Wrote {html_path}")
        # write_static_images(fig, png_path, svg_path)

        if INTERACTIVE:
            fig.show()


if __name__ == "__main__":
    main()
