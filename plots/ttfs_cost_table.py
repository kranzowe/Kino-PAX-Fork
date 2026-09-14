"""Kino-PAX / Kino-PAX+ / KinoPax* comparison table: Time to First Solution, broken down by region
(discretization level) and environment. One table set per model. SimpleCombo is intentionally
excluded (per request, to match a specific 3-column paper table); all four algorithms are still
available in ttfs_ratio_scatter.py / cost_ratio_scatter.py if a fuller comparison is ever needed.
TTFS is pooled across the length and effort cost-metric sweeps (cost metric doesn't affect solve
timing). The "empty" environment is excluded (trivially solved by everyone). Within each row, the
best (lowest) value across the three algorithms is bolded. See plots/cost_table_wide.py for the
companion cost table (first-solution cost and final cost, both cost metrics) -- kept as a
separate, wider table on purpose rather than crammed into this one.

Regions are Zephyr's Coarse/Fine/Tiny plus, model-dependent, a single Jetson row: the Jetson
dataset (plots/DATA/JETSON_20_runs) comes from the OLDER, pre-v2 benchmark pipeline
(examples/gpu/paper_benchmark.cu, not paper_benchmark_v2.cu), which only ever built one hardcoded
model per binary rather than sweeping all three, so each Jetson sweep is a *different single
model*: discretizationFINE was run at Model 2 (Dubins Airplane), discretizationCOARSE at Model 3
(Quad) -- confirmed from that harness's own header comments, which is also why COARSE's on-disk
delta token is "large" (that pipeline's own coarsest label) even though the folder is named
"discretizationCOARSE" to match Zephyr's convention. So Model 1's table has no Jetson row at all,
Model 2's table's Jetson row reads discretizationFINE, and Model 3's Jetson row reads
discretizationCOARSE -- each with its own model-tag-free filename pattern (e.g.
"house_KPAX_deltafine_length_run0.csv", no "m<N>_" prefix). A region whose folder doesn't exist
yet prints as "--" rather than erroring.

For each model, writes a CSV (plots/output/tables/) and prints + saves the equivalent LaTeX table
source (as a .txt file, ready to paste into the paper).

Edit ZEPHYR_DIR / JETSON_DIR / OUT_DIR below if your dataset folders move.
"""
from __future__ import annotations

import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import (  # noqa: E402
    COST_METRICS,
    DEFAULT_MAX_RUNS,
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
JETSON_DIR = os.path.join(PLOTS_DIR, "DATA", "JETSON_20_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "tables")

def regions_for_model(model_id: int) -> list:
    """Region rows for one model's table -- label, discretization folder, on-disk discretization
    token used inside run filenames, and whether that filename carries a "m<N>_" model tag. The
    Jetson row (if any) is model-specific -- see the module docstring."""
    regions = [{"label": "Coarse", "dir": os.path.join(ZEPHYR_DIR, "discretizationCOARSE"),
                "token": "large", "model_tag": True}]
    if model_id == 3:
        regions.append({"label": "Jetson (Coarse)", "dir": os.path.join(JETSON_DIR, "discretizationCOARSE"),
                         "token": "large", "model_tag": False})
    regions.append({"label": "Fine", "dir": os.path.join(ZEPHYR_DIR, "discretizationFINE"),
                     "token": "fine", "model_tag": True})
    if model_id == 2:
        regions.append({"label": "Jetson (Fine)", "dir": os.path.join(JETSON_DIR, "discretizationFINE"),
                         "token": "fine", "model_tag": False})
    regions.append({"label": "Tiny", "dir": os.path.join(ZEPHYR_DIR, "discretizationTINY"),
                     "token": "tiny", "model_tag": True})
    return regions

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


def _candidate_filename_no_model_tag(env: str, planner_token: str, delta_tok: str, run: int) -> str:
    """Filename builder for the older, pre-v2 pipeline (no 'm<N>_' model tag) -- see Jetson note."""
    if planner_token == KPAX:
        return f"{env}_KPAX_delta{delta_tok}_run{run}.csv"
    if planner_token.startswith("CountingStars") or planner_token.startswith("KinoPaxSTAR"):
        return f"{env}_{planner_token}_delta{delta_tok}_run{run}.csv"
    return f"{env}_delta{delta_tok}_run{run}.csv"


def load_runs_no_model_tag(env_dir, env, planner_token, discretization_label, metrics):
    runs = []
    for metric in metrics:
        delta_tok = f"{discretization_label}_{metric}"
        for run in range(DEFAULT_MAX_RUNS):
            fpath = os.path.join(env_dir, _candidate_filename_no_model_tag(env, planner_token, delta_tok, run))
            if not os.path.isfile(fpath):
                continue
            try:
                df = pd.read_csv(fpath, usecols=["best_cost", "elapsed_time_ms"])
                df["best_cost"] = pd.to_numeric(df["best_cost"], errors="coerce")
                runs.append(df)
            except (ValueError, pd.errors.EmptyDataError):
                pass
    return runs


def region_env_values(region: dict, env: str, model_id: int, metrics, aggregator) -> dict:
    env_dir = os.path.join(region["dir"], env)
    if not os.path.isdir(env_dir):
        return {p: math.nan for p in TABLE_PLANNERS}
    warn_on_unexpected_star_suffixes(env_dir)
    values = {}
    for planner in TABLE_PLANNERS:
        if region["model_tag"]:
            runs = load_runs(env_dir, env, planner, model_id, region["token"], metrics=metrics)
        else:
            runs = load_runs_no_model_tag(env_dir, env, planner, region["token"], metrics=metrics)
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
