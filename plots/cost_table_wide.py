"""Kino-PAX / Kino-PAX+ / KinoPax* cost comparison table: first-solution cost AND final cost, side
by side, for both cost metrics (workspace path length, control effort), broken down by region
(discretization level -- coarse/fine/tiny, plus the separate Jetson "fine" sweep) and environment.
Companion to ttfs_table.py (via ttfs_cost_table.py), kept as its own wider table on purpose --
fitting both cost stats per algorithm needs 6 data columns instead of 3, which reads better as a
full-page-width table* than crammed into the narrow TTFS table. One table set per model.
SimpleCombo is intentionally excluded (per request, to match a specific paper table). Cost metrics
are never pooled (length and effort are different units, from separate sweeps). The "empty" and
"zigzag" environments are excluded.

"Fine (Jetson)" reads from a completely separate dataset folder (plots/DATA/JETSON_20_runs), which
comes from the OLDER, pre-v2 benchmark pipeline (examples/gpu/paper_benchmark.cu): its filenames
have no "m<N>_" model tag, because that harness only ever built one hardcoded model per binary
(#define MODEL 2, i.e. Dubins Airplane) rather than sweeping all three -- so this region is only
ever populated for Model 2, and only for the "length" metric (no "effort" runs exist there at all).
A region whose folder doesn't exist yet (coarse/fine haven't been swept on Zephyr as of this
writing) prints as "--" rather than erroring.

For each model, writes a CSV (plots/output/tables/) and prints + saves the equivalent LaTeX
table* source (as a .txt file, ready to paste into the paper).

Edit ZEPHYR_DIR / JETSON_DIR / OUT_DIR below if your dataset folders move.
"""
from __future__ import annotations

import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zephyr_common import (  # noqa: E402
    DEFAULT_MAX_RUNS,
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
JETSON_DIR = os.path.join(PLOTS_DIR, "DATA", "JETSON_20_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "tables")

# label, discretization folder, on-disk discretization token used inside run filenames,
# whether that filename carries a "m<N>_" model tag, and (if not) which single model the data
# actually is -- see the Jetson explanation above.
REGIONS = [
    {"label": "Coarse", "dir": os.path.join(ZEPHYR_DIR, "discretizationCOARSE"),
     "token": "coarse", "model_tag": True, "only_model": None},
    {"label": "Fine", "dir": os.path.join(ZEPHYR_DIR, "discretizationFINE"),
     "token": "fine", "model_tag": True, "only_model": None},
    {"label": "Fine (Jetson)", "dir": os.path.join(JETSON_DIR, "discretizationFINE"),
     "token": "fine", "model_tag": False, "only_model": 2},
    {"label": "Tiny", "dir": os.path.join(ZEPHYR_DIR, "discretizationTINY"),
     "token": "tiny", "model_tag": True, "only_model": None},
]

ENVIRONMENTS = ["house", "narrowPassage"]  # on-disk spelling; "empty" and "zigzag" excluded

# Exactly the three columns requested -- SimpleCombo intentionally omitted.
TABLE_PLANNERS = [KPAX, KINOPAX_PLUS, KINOPAX_STAR]
COLUMN_LABELS = {KPAX: "Kino-PAX", KINOPAX_PLUS: "Kino-PAX+", KINOPAX_STAR: "KinoPax*"}

MODEL_NAMES_LOCAL = {1: "DoubleIntegrator", 2: "DubinsAirplane", 3: "Quad"}
MODEL_SUBTITLES = {
    1: "6D Double Integrator",
    2: "6D Dubins Airplane",
    3: "12D Nonlinear Drone",
}

# (section title, metric, decimal places to print)
SECTIONS = [
    ("Cost (Workspace Path Length)", "length", 3),
    ("Cost (Control Effort)", "effort", 1),
]

# The two cost stats each algorithm column is split into.
STATS = [("First", aggregate_first_sol_cost), ("Final", aggregate_final_cost)]


def _candidate_filename_no_model_tag(env: str, planner_token: str, delta_tok: str, run: int) -> str:
    """Filename builder for the older, pre-v2 pipeline (no 'm<N>_' model tag) -- see Jetson note."""
    if planner_token == KPAX:
        return f"{env}_KPAX_delta{delta_tok}_run{run}.csv"
    if planner_token.startswith("CountingStars") or planner_token.startswith("KinoPaxSTAR"):
        return f"{env}_{planner_token}_delta{delta_tok}_run{run}.csv"
    return f"{env}_delta{delta_tok}_run{run}.csv"


def load_runs_no_model_tag(env_dir, env, planner_token, discretization_label, metric):
    runs = []
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


def region_env_values(region: dict, env: str, model_id: int, metric: str) -> dict:
    """{planner: {"First": value, "Final": value}}"""
    empty = {p: {stat_name: math.nan for stat_name, _ in STATS} for p in TABLE_PLANNERS}
    if region["only_model"] is not None and region["only_model"] != model_id:
        return empty
    env_dir = os.path.join(region["dir"], env)
    if not os.path.isdir(env_dir):
        return empty
    warn_on_unexpected_star_suffixes(env_dir)
    values = {}
    for planner in TABLE_PLANNERS:
        if region["model_tag"]:
            runs = load_runs(env_dir, env, planner, model_id, region["token"], metrics=(metric,))
        else:
            runs = load_runs_no_model_tag(env_dir, env, planner, region["token"], metric)
        values[planner] = {stat_name: aggregator(runs).mean for stat_name, aggregator in STATS}
    return values


def build_model_sections(model_id: int) -> dict:
    """{section_title: (decimals, {env: {region_label: {planner: {stat: value}}}})}"""
    sections = {}
    for section_title, metric, decimals in SECTIONS:
        env_data = {}
        for env in ENVIRONMENTS:
            region_rows = {}
            for region in REGIONS:
                region_rows[region["label"]] = region_env_values(region, env, model_id, metric)
            env_data[env] = region_rows
        sections[section_title] = (decimals, env_data)
    return sections


def fmt(value: float, decimals: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{decimals}f}"


def sections_to_dataframe(sections: dict) -> pd.DataFrame:
    rows = []
    for section_title, (decimals, env_data) in sections.items():
        for env in ENVIRONMENTS:
            for region in REGIONS:
                region_label = region["label"]
                vals = env_data[env][region_label]
                row = {
                    "Section": section_title,
                    "Environment": env_display_name(env),
                    "Region": region_label,
                }
                for planner in TABLE_PLANNERS:
                    for stat_name, _ in STATS:
                        row[f"{COLUMN_LABELS[planner]} ({stat_name})"] = vals[planner][stat_name]
                rows.append(row)
    return pd.DataFrame(rows)


def render_latex(subtitle: str, sections: dict) -> str:
    n_cols = 1 + len(TABLE_PLANNERS) * len(STATS)  # Region + (planner x stat)
    col_spec = "l" + "r" * (n_cols - 1)

    group_header = [""]
    for planner in TABLE_PLANNERS:
        group_header.append(rf"\multicolumn{{{len(STATS)}}}{{c}}{{{COLUMN_LABELS[planner]}}}")
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + i * len(STATS)}-{1 + (i + 1) * len(STATS)}}}"
        for i in range(len(TABLE_PLANNERS))
    )
    stat_header = ["Region"] + [stat_name for _ in TABLE_PLANNERS for stat_name, _ in STATS]

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
            lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{env_display_name(env)}}}}} \\")
            for region in REGIONS:
                region_label = region["label"]
                vals = env_data[env][region_label]
                cells = [region_label]
                for planner in TABLE_PLANNERS:
                    for stat_name, _ in STATS:
                        cells.append(fmt(vals[planner][stat_name], decimals))
                lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    missing_regions = [r["label"] for r in REGIONS if not os.path.isdir(r["dir"])]
    if missing_regions:
        print(f"Note: these regions have no data folder yet and will print as '--': {missing_regions}\n")

    for model_id in MODEL_IDS:
        subtitle = MODEL_SUBTITLES[model_id]
        sections = build_model_sections(model_id)

        base_name = f"cost_table_wide_m{model_id}_{sanitize_name(MODEL_NAMES_LOCAL[model_id])}"
        csv_path = os.path.join(OUT_DIR, f"{base_name}.csv")
        sections_to_dataframe(sections).to_csv(csv_path, index=False)

        latex = render_latex(subtitle, sections)
        txt_path = os.path.join(OUT_DIR, f"{base_name}.tex.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(latex + "\n")

        print(f"% ===== {subtitle} =====")
        print(latex)
        print(f"\nWrote {csv_path} and {txt_path}\n")


if __name__ == "__main__":
    main()
