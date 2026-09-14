"""Kino-PAX / Kino-PAX+ / KinoPax* cost comparison table: first-solution cost AND final cost, side
by side, for both cost metrics (workspace path length, control effort), broken down by region
(discretization level) and environment. Companion to ttfs_table.py (via ttfs_cost_table.py), kept
as its own wider table on purpose -- fitting both cost stats per algorithm needs 6 data columns
instead of 3, which reads better as a full-page-width table* than crammed into the narrow TTFS
table. One table set per model. SimpleCombo is intentionally excluded (per request, to match a
specific paper table). Cost metrics are never pooled (length and effort are different units, from
separate sweeps). The "empty" environment is excluded (trivially solved by everyone). Within each
row, the best (lowest) First value and, separately, the best Final value across the three
algorithms are bolded (success rate is never bolded).

Model 3 (12D Nonlinear Drone) gets two special treatments per request: its cost values are
divided by 100 (still to 1 decimal place -- otherwise this model's costs are ~2 orders of
magnitude larger than the other two models', which is just their inherent path/effort scale, not
a real difference worth showing at full precision here), and each algorithm gets a third
Success Rate column (successful runs / total runs, as a percentage) alongside First and Final --
useful specifically for this model since (per plots/ttfs_ratio_scatter.py's Coarse-discretization
finding) Kino-PAX+ can fail outright at some regions for this model, which a bare cost number
can't distinguish from "solved but expensive."

Regions are just Zephyr's Coarse/Fine/Tiny -- see ttfs_cost_table.py's module docstring for why
the Jetson dataset's model-specific row was dropped from both tables (its older harness never
records which vehicle model produced a given run, and the git history of the script that built it
rules out the model attribution these tables used to assume). A region whose folder doesn't exist
yet prints as "--" rather than erroring.

For each model, writes a CSV (plots/output/tables/) and prints + saves the equivalent LaTeX
table* source (as a .txt file, ready to paste into the paper).

Edit ZEPHYR_DIR / OUT_DIR below if your dataset folders move.
"""
from __future__ import annotations

import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import (  # noqa: E402
    KINOPAX_PLUS,
    KINOPAX_STAR,
    KPAX,
    MODEL_IDS,
    aggregate_final_cost,
    aggregate_first_sol_cost,
    env_display_name,
    load_runs,
    sanitize_name,
    warn_on_unexpected_star_suffixes,
)

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================
# EDIT THESE if your dataset folders move.
# ================================================================================================
ZEPHYR_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "tables")

def regions_for_model(model_id: int) -> list:
    """Region rows for one model's table -- label, discretization folder, and the on-disk
    discretization token used inside run filenames. Same three regions for every model -- see the
    module docstring for why the Jetson dataset's model-specific row was dropped. Zephyr's
    "Coarse" folder's on-disk token is "large" (the harness's own coarsest-resolution label), not
    "coarse"."""
    return [
        {"label": "Coarse", "dir": os.path.join(ZEPHYR_DIR, "discretizationCOARSE"), "token": "large"},
        {"label": "Fine", "dir": os.path.join(ZEPHYR_DIR, "discretizationFINE"), "token": "fine"},
        {"label": "Tiny", "dir": os.path.join(ZEPHYR_DIR, "discretizationTINY"), "token": "tiny"},
    ]

ENVIRONMENTS = ["house", "narrowPassage", "zigzag"]  # on-disk spelling; "empty" excluded

# Exactly the three columns requested -- SimpleCombo intentionally omitted.
TABLE_PLANNERS = [KPAX, KINOPAX_PLUS, KINOPAX_STAR]
COLUMN_LABELS = {KPAX: "Kino-PAX", KINOPAX_PLUS: "Kino-PAX+", KINOPAX_STAR: "KinoPax*"}

MODEL_NAMES_LOCAL = {1: "DoubleIntegrator", 2: "DubinsAirplane", 3: "Quad"}
MODEL_SUBTITLES = {
    1: "6D Double Integrator",
    2: "6D Dubins Airplane",
    3: "12D Nonlinear Drone",
}

def sections_for_model(model_id: int) -> list:
    """(section title, metric, decimal places, value scale) -- Model 3's costs are divided by
    100 (scale=0.01); the other two models are unscaled."""
    if model_id == 3:
        return [
            ("Cost (Workspace Path Length)", "length", 2, 0.01),
            ("Cost (Control Effort)", "effort", 2, 0.01),
        ]
    return [
        ("Cost (Workspace Path Length)", "length", 4, 1.0),
        ("Cost (Control Effort)", "effort", 2, 1.0),
    ]


def stats_for_model(model_id: int) -> list:
    return ["First", "Final", "SuccessRate"] #if model_id == 3 else ["First", "Final"]


def region_env_values(region: dict, env: str, model_id: int, metric: str, scale: float) -> dict:
    """{planner: {"First": value, "Final": value, "SuccessRate": pct}} -- First/Final already
    have `scale` applied; SuccessRate is always a 0-100 percentage, unaffected by scale."""
    empty = {p: {"First": math.nan, "Final": math.nan, "SuccessRate": math.nan} for p in TABLE_PLANNERS}
    env_dir = os.path.join(region["dir"], env)
    if not os.path.isdir(env_dir):
        return empty
    warn_on_unexpected_star_suffixes(env_dir)
    values = {}
    for planner in TABLE_PLANNERS:
        runs = load_runs(env_dir, env, planner, model_id, region["token"], metrics=(metric,))
        first_stats = aggregate_first_sol_cost(runs)
        final_stats = aggregate_final_cost(runs)
        success_rate = 100.0 * first_stats.n_success / first_stats.n_total if first_stats.n_total else math.nan
        values[planner] = {
            "First": first_stats.mean * scale,
            "Final": final_stats.mean * scale,
            "SuccessRate": success_rate,
        }
    return values


def build_model_sections(model_id: int, regions: list) -> dict:
    """{section_title: (decimals, {env: {region_label: {planner: {stat: value}}}})}"""
    sections = {}
    for section_title, metric, decimals, scale in sections_for_model(model_id):
        env_data = {}
        for env in ENVIRONMENTS:
            region_rows = {}
            for region in regions:
                region_rows[region["label"]] = region_env_values(region, env, model_id, metric, scale)
            env_data[env] = region_rows
        sections[section_title] = (decimals, env_data)
    return sections


def fmt(value: float, decimals: int, bold: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    text = f"{value:.{decimals}f}"
    return rf"\textbf{{{text}}}" if bold else text


def fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.0f}\\%"


def best_planner(vals: dict, stat_name: str) -> object:
    """Planner key with the lowest (best) `stat_name` value, ignoring NaN/missing; None if all
    missing."""
    valid = {p: v[stat_name] for p, v in vals.items()
             if not (isinstance(v[stat_name], float) and math.isnan(v[stat_name]))}
    return min(valid, key=valid.get) if valid else None


def sections_to_dataframe(sections: dict, regions: list) -> pd.DataFrame:
    rows = []
    for section_title, (decimals, env_data) in sections.items():
        for env in ENVIRONMENTS:
            for region in regions:
                region_label = region["label"]
                vals = env_data[env][region_label]
                row = {
                    "Section": section_title,
                    "Environment": env_display_name(env),
                    "Region": region_label,
                }
                for planner in TABLE_PLANNERS:
                    for stat_name in ("First", "Final", "SuccessRate"):
                        row[f"{COLUMN_LABELS[planner]} ({stat_name})"] = vals[planner][stat_name]
                rows.append(row)
    return pd.DataFrame(rows)


def render_latex(subtitle: str, sections: dict, regions: list, model_id: int) -> str:
    """NOTE: the Environment column uses \\multirow, so the LaTeX preamble needs
    \\usepackage{multirow} for this to compile."""
    stat_names = stats_for_model(model_id)
    n_stats = len(stat_names)
    n_cols = 2 + len(TABLE_PLANNERS) * n_stats  # Environment + Region + (planner x stat)
    col_spec = "ll" + "r" * (n_cols - 2)

    group_header = ["", ""]
    for planner in TABLE_PLANNERS:
        group_header.append(rf"\multicolumn{{{n_stats}}}{{c}}{{{COLUMN_LABELS[planner]}}}")
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{3 + i * n_stats}-{2 + (i + 1) * n_stats}}}"
        for i in range(len(TABLE_PLANNERS))
    )
    stat_header = ["Environment", "Region"] + [
        ("Success Rate" if s == "SuccessRate" else s) for _ in TABLE_PLANNERS for s in stat_names
    ]

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        rf"\caption{{Cost Comparison --- {subtitle}}}",
        rf"\label{{tab:cost_comparison_{sanitize_name(subtitle).lower()}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(group_header) + r" \\",
        cmidrules,
        " & ".join(stat_header) + r" \\",
    ]
    for section_title, (decimals, env_data) in sections.items():
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{section_title}}}}} \\")
        for env in ENVIRONMENTS:
            lines.append(r"\midrule")
            # Environment name sits in its own column (via \multirow, spanning this
            # environment's region rows) instead of a separate header row above them --
            # saves a line per environment; the room for the extra column is there
            # horizontally since this table is already a full-page-width table*.
            for i, region in enumerate(regions):
                region_label = region["label"]
                vals = env_data[env][region_label]
                best_first = best_planner(vals, "First")
                best_final = best_planner(vals, "Final")
                env_cell = rf"\multirow{{{len(regions)}}}{{*}}{{{env_display_name(env)}}}" if i == 0 else ""
                cells = [env_cell, region_label]
                for planner in TABLE_PLANNERS:
                    for stat_name in stat_names:
                        if stat_name == "SuccessRate":
                            cells.append(fmt_pct(vals[planner][stat_name]))
                        else:
                            is_best = planner == (best_first if stat_name == "First" else best_final)
                            cells.append(fmt(vals[planner][stat_name], decimals, bold=is_best))
                lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    for model_id in MODEL_IDS:
        subtitle = MODEL_SUBTITLES[model_id]
        regions = regions_for_model(model_id)
        missing_regions = [r["label"] for r in regions if not os.path.isdir(r["dir"])]
        if missing_regions:
            print(f"[{subtitle}] Note: no data folder yet, will print as '--': {missing_regions}")

        sections = build_model_sections(model_id, regions)

        base_name = f"cost_table_wide_m{model_id}_{sanitize_name(MODEL_NAMES_LOCAL[model_id])}"
        csv_path = os.path.join(OUT_DIR, f"{base_name}.csv")
        sections_to_dataframe(sections, regions).to_csv(csv_path, index=False)

        latex = render_latex(subtitle, sections, regions, model_id)
        txt_path = os.path.join(OUT_DIR, f"{base_name}.tex.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(latex + "\n")

        print(f"% ===== {subtitle} =====")
        print(latex)
        print(f"\nWrote {csv_path} and {txt_path}\n")


if __name__ == "__main__":
    main()
