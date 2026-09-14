"""Kino-PAX / Kino-PAX+ / KinoPax* comparison table: Time to First Solution, broken down by region
(discretization level) and environment. One table set per model. SimpleCombo is intentionally
excluded (per request, to match a specific 3-column paper table); all four algorithms are still
available in ttfs_ratio_scatter.py / cost_ratio_scatter.py if a fuller comparison is ever needed.
TTFS is pooled across the length and effort cost-metric sweeps (cost metric doesn't affect solve
timing). The "empty" environment is excluded (trivially solved by everyone). Within each row, the
best (lowest) value across the three algorithms is bolded. See plots/cost_table_wide.py for the
companion cost table (first-solution cost and final cost, both cost metrics) -- kept as a
separate, wider table on purpose rather than crammed into this one.

Regions are just Zephyr's Coarse/Fine/Tiny. A JETSON_20_runs dataset also exists on disk (the
OLDER, pre-v2 benchmark pipeline, examples/gpu/paper_benchmark.cu) and used to contribute one
model-specific row per model here -- deliberately dropped: that pipeline hardcodes exactly one
vehicle model per compiled binary and never records which one in the run's filename or CSV
columns, and a check of scripts/run_paper_benchmark.sh's full git history turned up two problems
with the row-per-model attribution this table used to rely on -- (1) that script has only ever
been configured for Model 1 or Model 2, never Model 3, so the "Jetson Coarse = Model 3" mapping
this table used is provably wrong, and (2) within any one script version, ALL THREE deltas
(large/fine/tiny) always ran under the SAME hardcoded model, so "Jetson Fine = Model 2" is only
right if that folder's data was actually captured while the script was in its Model-2 state --
unverifiable from the repo alone. Re-add Jetson rows once that provenance is actually confirmed
(e.g. from lab notes on what was checked out on the Jetson device at capture time), rather than
guessing again. A region whose folder doesn't exist yet prints as "--" rather than erroring.

For each model, writes a CSV (plots/output/tables/) and prints + saves the equivalent LaTeX table
source (as a .txt file, ready to paste into the paper).

Edit ZEPHYR_DIR / OUT_DIR below if your dataset folders move.
"""
from __future__ import annotations

import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import (  # noqa: E402
    COST_METRICS,
    KINOPAX_PLUS,
    KINOPAX_STAR,
    KPAX,
    MODEL_IDS,
    aggregate_ttfs,
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
    module docstring for why the Jetson dataset's model-specific row was dropped."""
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

# (section title, metrics to load, aggregator, decimal places to print)
SECTIONS = [
    ("TTFS (ms)", COST_METRICS, aggregate_ttfs, 1),
]


def region_env_values(region: dict, env: str, model_id: int, metrics, aggregator) -> dict:
    env_dir = os.path.join(region["dir"], env)
    if not os.path.isdir(env_dir):
        return {p: math.nan for p in TABLE_PLANNERS}
    warn_on_unexpected_star_suffixes(env_dir)
    values = {}
    for planner in TABLE_PLANNERS:
        runs = load_runs(env_dir, env, planner, model_id, region["token"], metrics=metrics)
        values[planner] = aggregator(runs).mean
    return values


def build_model_sections(model_id: int, regions: list) -> dict:
    """{section_title: (decimals, {env: {region_label: {planner: value}}})}"""
    sections = {}
    for section_title, metrics, aggregator, decimals in SECTIONS:
        env_data = {}
        for env in ENVIRONMENTS:
            region_rows = {}
            for region in regions:
                region_rows[region["label"]] = region_env_values(region, env, model_id, metrics, aggregator)
            env_data[env] = region_rows
        sections[section_title] = (decimals, env_data)
    return sections


def fmt(value: float, decimals: int, bold: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    text = f"{value:.{decimals}f}"
    return rf"\textbf{{{text}}}" if bold else text


def best_planner(vals: dict) -> object:
    """Planner key with the lowest (best) value in `vals`, ignoring NaN/missing; None if all
    missing."""
    valid = {p: v for p, v in vals.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    return min(valid, key=valid.get) if valid else None


def sections_to_dataframe(sections: dict, regions: list) -> pd.DataFrame:
    rows = []
    for section_title, (decimals, env_data) in sections.items():
        for env in ENVIRONMENTS:
            for region in regions:
                region_label = region["label"]
                vals = env_data[env][region_label]
                rows.append({
                    "Section": section_title,
                    "Environment": env_display_name(env),
                    "Region": region_label,
                    "Kino-PAX": vals[KPAX],
                    "Kino-PAX+": vals[KINOPAX_PLUS],
                    "KinoPax*": vals[KINOPAX_STAR],
                })
    return pd.DataFrame(rows)


def render_latex(subtitle: str, sections: dict, regions: list) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Time to First Solution --- {subtitle}}}",
        rf"\label{{tab:ttfs_comparison_{sanitize_name(subtitle).lower()}}}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Region & " + " & ".join(COLUMN_LABELS[p] for p in TABLE_PLANNERS) + r" \\",
    ]
    for section_title, (decimals, env_data) in sections.items():
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textbf{{{section_title}}}}} \\")
        for env in ENVIRONMENTS:
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{env_display_name(env)}}}}} \\")
            for region in regions:
                region_label = region["label"]
                vals = env_data[env][region_label]
                best = best_planner(vals)
                cells = " & ".join(fmt(vals[p], decimals, bold=(p == best)) for p in TABLE_PLANNERS)
                lines.append(f"{region_label} & {cells} " + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
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

        base_name = f"ttfs_table_m{model_id}_{sanitize_name(MODEL_NAMES_LOCAL[model_id])}"
        csv_path = os.path.join(OUT_DIR, f"{base_name}.csv")
        sections_to_dataframe(sections, regions).to_csv(csv_path, index=False)

        latex = render_latex(subtitle, sections, regions)
        txt_path = os.path.join(OUT_DIR, f"{base_name}.tex.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(latex + "\n")

        print(f"% ===== {subtitle} =====")
        print(latex)
        print(f"\nWrote {csv_path} and {txt_path}\n")


if __name__ == "__main__":
    main()
