"""Kino-PAX / Kino-PAX+ / KinoPax* comparison table: Time to First Solution AND success rate,
side by side per algorithm, broken down by region (discretization level) and environment. One
table set per model. SimpleCombo is intentionally excluded (per request, to match a specific
3-column-of-algorithms paper table); all four algorithms are still available in
ttfs_ratio_scatter.py / cost_ratio_scatter.py if a fuller comparison is ever needed. TTFS (and its
success rate) is pooled across the length and effort cost-metric sweeps (cost metric doesn't
affect solve timing, and a run either solved or it didn't regardless of which cost it was
minimizing). The "empty" environment is excluded (trivially solved by everyone). Within each row,
the best (lowest) TTFS across the three algorithms is bolded; success rate is never bolded (it's
informational context for the TTFS next to it, not itself a competition column) -- same convention
as cost_table_wide.py's own SuccessRate column. See plots/cost_table_wide.py for the companion cost
table (first-solution cost and final cost, both cost metrics) -- kept as a separate, wider table on
purpose rather than crammed into this one.

Success rate sits in its OWN column per algorithm (2 columns each: TTFS (ms), Success) rather
than a parenthetical or footnote, on purpose -- it deserves the same horizontal weight as the TTFS
value next to it, not a squeezed-in afterthought.

Regions are Zephyr's Coarse/Fine/Tiny plus two confirmed Jetson rows (provenance confirmed by
hand this time, not inferred from git history -- see the earlier, wrong guess this docstring used
to describe): Jetson's discretizationCOARSE is the 12D Nonlinear Drone (Model 3) ONLY, so it adds
a "Jetson (Coarse)" row to Model 3's table right after "Coarse"; discretizationFINE is the two 6D
systems (Models 1 and 2), so it adds a "Jetson (Fine)" row to BOTH of their tables right after
"Fine". Unlike the Jetson data this table showed once before, Coarse and Fine now come from two
DIFFERENT harnesses with different filename conventions:
  - discretizationCOARSE/<env>/ is still the OLDER, pre-v2 pipeline (examples/gpu/
    paper_benchmark.cu) -- one hardcoded model per binary, NO "m<N>_" tag in the filename (e.g.
    "house_KPAX_deltalarge_length_run0.csv"), loaded with load_runs_no_model_tag() below.
  - discretizationFINE/FINE/<env>/ (note the doubled "FINE" -- the run harness's own output
    layout, not a mistake on this table's part) is the NEWER paper_benchmark_v3 pipeline, which
    swept all three models in one pass and DOES tag every filename with "m<N>_" (e.g.
    "..._deltam1_fine_length_run0.csv") -- loaded with the same load_runs() the Zephyr regions
    use, no special-casing needed, and Model 3's own (mostly-failed, per that run's
    failures.log -- Kino-PAX+ got 0/20 successful runs in two of three environments) fine-
    resolution attempt is deliberately NOT surfaced here, since Model 3's real fine-resolution
    story is Zephyr's own Fine row, not this troubled Jetson attempt at it.
A region whose folder doesn't exist yet prints as "--" rather than erroring.

For each model, writes a CSV (plots/output/tables/) and prints + saves the equivalent LaTeX table
source (as a .txt file, ready to paste into the paper) -- NOTE this table now uses \\multicolumn
grouped headers and \\cmidrule, so the LaTeX preamble needs \\usepackage{booktabs} (already
implied by \\toprule/\\midrule/\\bottomrule, used here already). \\tabcolsep is tightened to 2pt
(default 6pt) inside a \\begingroup/\\endgroup around just this table's tabular, since six wide
columns otherwise pad themselves out with more whitespace than the numbers need -- scoped locally
so it doesn't leak into any other table sharing the same document.

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


def _candidate_filename_no_model_tag(env: str, planner_token: str, delta_tok: str, run: int) -> str:
    """Filename builder for discretizationCOARSE's older, pre-v2 pipeline (no 'm<N>_' model tag)
    -- see the module docstring's Jetson note."""
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


def regions_for_model(model_id: int) -> list:
    """Region rows for one model's table -- label, discretization folder, on-disk discretization
    token, and whether that filename carries a "m<N>_" model tag. The Jetson row (if any) is
    model-specific -- see the module docstring."""
    regions = [{"label": "Coarse", "dir": os.path.join(ZEPHYR_DIR, "discretizationCOARSE"),
                "token": "large", "model_tag": True}]
    if model_id == 3:
        regions.append({"label": "Jetson (Coarse)", "dir": os.path.join(JETSON_DIR, "discretizationCOARSE"),
                         "token": "large", "model_tag": False})
    regions.append({"label": "Fine", "dir": os.path.join(ZEPHYR_DIR, "discretizationFINE"),
                     "token": "fine", "model_tag": True})
    if model_id in (1, 2):
        regions.append({"label": "Jetson (Fine)",
                         "dir": os.path.join(JETSON_DIR, "discretizationFINE", "FINE"),
                         "token": "fine", "model_tag": True})
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

# (section title, metrics to pool, decimal places for the TTFS value)
SECTIONS = [
    ("TTFS (ms)", COST_METRICS, 1),
]


def region_env_values(region: dict, env: str, model_id: int, metrics) -> dict:
    """{planner: {"TTFS": mean_ms, "SuccessRate": pct}} -- success rate reuses aggregate_ttfs's
    own n_success/n_total (its criterion, "did best_cost ever drop below the unsolved sentinel",
    is exactly "did this run solve at all")."""
    empty = {p: {"TTFS": math.nan, "SuccessRate": math.nan} for p in TABLE_PLANNERS}
    env_dir = os.path.join(region["dir"], env)
    if not os.path.isdir(env_dir):
        return empty
    warn_on_unexpected_star_suffixes(env_dir)
    values = {}
    for planner in TABLE_PLANNERS:
        if region["model_tag"]:
            runs = load_runs(env_dir, env, planner, model_id, region["token"], metrics=metrics)
        else:
            runs = load_runs_no_model_tag(env_dir, env, planner, region["token"], metrics)
        stats = aggregate_ttfs(runs)
        success_rate = 100.0 * stats.n_success / stats.n_total if stats.n_total else math.nan
        values[planner] = {"TTFS": stats.mean, "SuccessRate": success_rate}
    return values


def build_model_sections(model_id: int, regions: list) -> dict:
    """{section_title: (decimals, {env: {region_label: {planner: {"TTFS":..., "SuccessRate":...}}}})}"""
    sections = {}
    for section_title, metrics, decimals in SECTIONS:
        env_data = {}
        for env in ENVIRONMENTS:
            region_rows = {}
            for region in regions:
                region_rows[region["label"]] = region_env_values(region, env, model_id, metrics)
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


def best_planner(ttfs_vals: dict) -> object:
    """Planner key with the lowest (best) TTFS, ignoring NaN/missing; None if all missing."""
    valid = {p: v for p, v in ttfs_vals.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    return min(valid, key=valid.get) if valid else None


def sections_to_dataframe(sections: dict, regions: list) -> pd.DataFrame:
    rows = []
    for section_title, (decimals, env_data) in sections.items():
        for env in ENVIRONMENTS:
            for region in regions:
                region_label = region["label"]
                vals = env_data[env][region_label]
                row = {"Section": section_title, "Environment": env_display_name(env), "Region": region_label}
                for planner in TABLE_PLANNERS:
                    row[f"{COLUMN_LABELS[planner]} (TTFS ms)"] = vals[planner]["TTFS"]
                    row[f"{COLUMN_LABELS[planner]} (Success Rate)"] = vals[planner]["SuccessRate"]
                rows.append(row)
    return pd.DataFrame(rows)


def render_latex(subtitle: str, sections: dict, regions: list) -> str:
    """NOTE: grouped \\multicolumn headers + \\cmidrule, so the usual booktabs preamble
    (\\usepackage{booktabs}) needs to be in place -- already implied by \\toprule/\\midrule."""
    n_cols = 1 + 2 * len(TABLE_PLANNERS)  # Region + (planner x [TTFS, Success])
    col_spec = "l" + "rr" * len(TABLE_PLANNERS)

    group_header = [""] + [rf"\multicolumn{{2}}{{c}}{{{COLUMN_LABELS[p]}}}" for p in TABLE_PLANNERS]
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + i * 2}-{3 + i * 2}}}" for i in range(len(TABLE_PLANNERS))
    )
    stat_header = ["Region"]
    for _ in TABLE_PLANNERS:
        stat_header += ["TTFS (ms)", "Success"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Time to First Solution --- {subtitle}}}",
        rf"\label{{tab:ttfs_comparison_{sanitize_name(subtitle).lower()}}}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{2pt}",
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
            for region in regions:
                region_label = region["label"]
                vals = env_data[env][region_label]
                best = best_planner({p: vals[p]["TTFS"] for p in TABLE_PLANNERS})
                cells = [region_label]
                for p in TABLE_PLANNERS:
                    cells.append(fmt(vals[p]["TTFS"], decimals, bold=(p == best)))
                    cells.append(fmt_pct(vals[p]["SuccessRate"]))
                lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\endgroup")
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
