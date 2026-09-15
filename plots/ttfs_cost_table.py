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

Regions are Zephyr's Coarse/Fine/Tiny. A region whose folder doesn't exist yet prints as "--"
rather than erroring.

SHORT-TIMEOUT BRANCH: pointed at ZEPHYR_30_runs_SHORT (1M-node / 3s-timeout sweep) instead of the
main ZEPHYR_30_runs dataset -- own output folder/filenames so the two never overwrite each other.
Jetson rows are OMITTED here (the main-dataset version of this table adds "Jetson (Coarse)" /
"Jetson (Fine)" rows from JETSON_20_runs): there's no short-timeout Jetson sweep, and mixing the
old 20-run Jetson data -- a different config entirely -- into a table that's otherwise all
1M-node/3s numbers would be misleading. ZEPHYR_30_runs_SHORT also spells its discretization
folders WITHOUT the "discretization" prefix ("COARSE" not "discretizationCOARSE") --
resolve_discretization_dir (zephyr_common) tries both spellings so regions_for_model below needs
no further changes for that.

For each model, writes a CSV (plots/output/tables_short/) and prints + saves the equivalent LaTeX
table source (as a .txt file, ready to paste into the paper) -- NOTE this table now uses
\\multicolumn grouped headers and \\cmidrule, so the LaTeX preamble needs \\usepackage{booktabs}
(already implied by \\toprule/\\midrule/\\bottomrule, used here already). \\tabcolsep is tightened
to 2pt (default 6pt) inside a \\begingroup/\\endgroup around just this table's tabular, since six
wide columns otherwise pad themselves out with more whitespace than the numbers need -- scoped
locally so it doesn't leak into any other table sharing the same document.

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
    resolve_discretization_dir,
    sanitize_name,
    warn_on_unexpected_star_suffixes,
)

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================
# EDIT THESE if your dataset folders move.
# SHORT-TIMEOUT BRANCH: ZEPHYR_DIR points at ZEPHYR_30_runs_SHORT, own OUT_DIR -- see module
# docstring.
# ================================================================================================
ZEPHYR_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs_SHORT")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "tables_short")


def regions_for_model(model_id: int) -> list:
    """Region rows for one model's table -- label, discretization folder, and on-disk
    discretization token. No Jetson rows on this branch -- see the module docstring."""
    return [
        {"label": "Coarse", "dir": resolve_discretization_dir(ZEPHYR_DIR, "COARSE"), "token": "large"},
        {"label": "Fine", "dir": resolve_discretization_dir(ZEPHYR_DIR, "FINE"), "token": "fine"},
        {"label": "Tiny", "dir": resolve_discretization_dir(ZEPHYR_DIR, "TINY"), "token": "tiny"},
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
        runs = load_runs(env_dir, env, planner, model_id, region["token"], metrics=metrics)
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
        rf"\caption{{Time to First Solution --- {subtitle} (Short Timeout: 1M-node / 3s)}}",
        rf"\label{{tab:ttfs_comparison_{sanitize_name(subtitle).lower()}_short}}",
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

        base_name = f"ttfs_table_m{model_id}_{sanitize_name(MODEL_NAMES_LOCAL[model_id])}_short"
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
