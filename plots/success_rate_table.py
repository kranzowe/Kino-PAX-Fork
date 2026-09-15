"""Success rate comparison table: percentage of runs that found ANY solution (best_cost ever
dropped below the unsolved sentinel), broken down by region (discretization level) and
environment, for all four algorithms (Kino-PAX, Kino-PAX+, SimpleCombo, KinoPax*) -- unlike
ttfs_cost_table.py / cost_table_wide.py, this one keeps SimpleCombo, since there's no specific
paper table it needs to match column-for-column here. One table set per model.

NOT POOLED ACROSS COST METRICS, on purpose (unlike ttfs_cost_table.py's TTFS, which pools length
and effort since solve timing doesn't depend on which cost is being minimized): success/failure
CAN differ between the two sweeps -- they're separate run batches, and this project has already
found real cases where one metric succeeds cleanly while the other fails outright for the same
(model, discretization) cell (e.g. Kino-PAX+ at Coarse for the Dubins Airplane: 0/90 successful
runs on Control Effort specifically, per cost_big_panel.py's own red-flagged finding). Pooling
would average that away into a misleadingly middling number instead of surfacing it. So each
model's table has two sections, one per cost metric, each showing that metric's own success rate.

Within each row, the BEST (highest, not lowest -- success rate is the one metric in this whole
table suite where bigger is better) value across the four algorithms is bolded.

Regions are Zephyr's Coarse/Fine/Tiny plus two confirmed Jetson rows -- see ttfs_cost_table.py's
module docstring for the full provenance story: discretizationFINE/FINE (tagged pipeline) for
Models 1 and 2 only, and a "Jetson (42K)" row for Model 3 from JETSON_DIR/NEWQUAD_MEDIUM (new
hardware data at a genuinely new discretization, replacing this table's earlier, much weaker
"Jetson (Coarse)" attempt). A region whose folder doesn't exist yet prints as "--" rather than
erroring.

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
    BASE_DISPLAY,
    BASE_NAMES,
    COST_METRIC_LABELS,
    COST_METRICS,
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
    """Region rows for one model's table -- label, discretization folder, and on-disk
    discretization token. The Jetson row (if any) is model-specific -- see the module docstring.
    Every region here uses the tagged pipeline (load_runs) now that Model 3's Jetson row is
    NEWQUAD_MEDIUM instead of the old untagged discretizationCOARSE attempt."""
    regions = [{"label": "Coarse", "dir": os.path.join(ZEPHYR_DIR, "discretizationCOARSE"), "token": "large"}]
    if model_id == 3:
        regions.append({"label": "Jetson (42K)", "dir": os.path.join(JETSON_DIR, "NEWQUAD_MEDIUM"),
                         "token": "medium"})
    regions.append({"label": "Fine", "dir": os.path.join(ZEPHYR_DIR, "discretizationFINE"), "token": "fine"})
    if model_id in (1, 2):
        regions.append({"label": "Jetson (Fine)",
                         "dir": os.path.join(JETSON_DIR, "discretizationFINE", "FINE"), "token": "fine"})
    regions.append({"label": "Tiny", "dir": os.path.join(ZEPHYR_DIR, "discretizationTINY"), "token": "tiny"})
    return regions

ENVIRONMENTS = ["house", "narrowPassage", "zigzag"]  # on-disk spelling; "empty" excluded

# All four algorithms -- see module docstring for why this table doesn't drop SimpleCombo the way
# ttfs_cost_table.py / cost_table_wide.py do.
TABLE_PLANNERS = list(BASE_NAMES)
COLUMN_LABELS = {p: BASE_DISPLAY[p] for p in TABLE_PLANNERS}

MODEL_NAMES_LOCAL = {1: "DoubleIntegrator", 2: "DubinsAirplane", 3: "Quad"}
MODEL_SUBTITLES = {
    1: "6D Double Integrator",
    2: "6D Dubins Airplane",
    3: "12D Nonlinear Drone",
}

# One section per cost metric, deliberately NOT pooled -- see module docstring.
SECTIONS = [(COST_METRIC_LABELS[m], (m,)) for m in COST_METRICS]


def success_rate(runs: list) -> float:
    """Percentage of runs that found ANY solution -- n_success/n_total from aggregate_ttfs()
    (whose success criterion, "did best_cost ever drop below the unsolved sentinel", is exactly
    "did this run solve at all", independent of which stat you aggregate). NaN if there were no
    runs to begin with (region/environment combination doesn't exist), same "--" convention as
    every other table in this folder."""
    stats = aggregate_ttfs(runs)
    return 100.0 * stats.n_success / stats.n_total if stats.n_total else math.nan


def region_env_values(region: dict, env: str, model_id: int, metrics) -> dict:
    env_dir = os.path.join(region["dir"], env)
    if not os.path.isdir(env_dir):
        return {p: math.nan for p in TABLE_PLANNERS}
    warn_on_unexpected_star_suffixes(env_dir)
    values = {}
    for planner in TABLE_PLANNERS:
        runs = load_runs(env_dir, env, planner, model_id, region["token"], metrics=metrics)
        values[planner] = success_rate(runs)
    return values


def build_model_sections(model_id: int, regions: list) -> dict:
    """{section_title: {env: {region_label: {planner: success_rate_pct}}}}"""
    sections = {}
    for section_title, metrics in SECTIONS:
        env_data = {}
        for env in ENVIRONMENTS:
            region_rows = {}
            for region in regions:
                region_rows[region["label"]] = region_env_values(region, env, model_id, metrics)
            env_data[env] = region_rows
        sections[section_title] = env_data
    return sections


def fmt_pct(value: float, bold: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    text = f"{value:.0f}\\%"
    return rf"\textbf{{{text}}}" if bold else text


def best_planner(vals: dict) -> object:
    """Planner key with the HIGHEST (best) value in `vals`, ignoring NaN/missing; None if all
    missing -- success rate is the one stat in this table suite where bigger is better."""
    valid = {p: v for p, v in vals.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    return max(valid, key=valid.get) if valid else None


def sections_to_dataframe(sections: dict, regions: list) -> pd.DataFrame:
    rows = []
    for section_title, env_data in sections.items():
        for env in ENVIRONMENTS:
            for region in regions:
                region_label = region["label"]
                vals = env_data[env][region_label]
                row = {"Section": section_title, "Environment": env_display_name(env), "Region": region_label}
                for planner in TABLE_PLANNERS:
                    row[COLUMN_LABELS[planner]] = vals[planner]
                rows.append(row)
    return pd.DataFrame(rows)


def render_latex(subtitle: str, sections: dict, regions: list) -> str:
    n_cols = 1 + len(TABLE_PLANNERS)
    col_spec = "l" + "r" * len(TABLE_PLANNERS)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Success Rate --- {subtitle}}}",
        rf"\label{{tab:success_rate_{sanitize_name(subtitle).lower()}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"Region & " + " & ".join(COLUMN_LABELS[p] for p in TABLE_PLANNERS) + r" \\",
    ]
    for section_title, env_data in sections.items():
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{section_title}}}}} \\")
        for env in ENVIRONMENTS:
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{env_display_name(env)}}}}} \\")
            for region in regions:
                region_label = region["label"]
                vals = env_data[env][region_label]
                best = best_planner(vals)
                cells = " & ".join(fmt_pct(vals[p], bold=(p == best)) for p in TABLE_PLANNERS)
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

        base_name = f"success_rate_table_m{model_id}_{sanitize_name(MODEL_NAMES_LOCAL[model_id])}"
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
