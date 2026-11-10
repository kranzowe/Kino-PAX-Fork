"""Concise, single-column success-rate summary: one row per algorithm within each discretization
level, pooled across every environment AND every model -- "success rate of each algorithm at each
discretization," nothing else, for a quick top-line table rather than the full (region x
environment x model) breakdown that success_rate_table.py / success_rate_heatmap.py already cover.

TRUNCATED TO A 4-SECOND BUDGET, NOT "DID IT EVER SOLVE": plots/DATA/ZEPHYR_30_runs was not run
with a 4s cap, so a naive success check ("did best_cost ever drop below the unsolved sentinel by
the run's own end") would wrongly count a run that only found its first solution at, say, 9
seconds as a "success" -- not what a 4s-budget success rate means. Every rate here instead uses
zephyr_common.success_rate_truncated, which is built on first_sol_time_truncated: a run counts as
solved only if its real first-solution time is <= TIMEOUT_MS. See cost_panel_vs_kinopaxplus_4s.py
/ ttfs_panel_vs_kpax_4s.py for the same 4s-truncation idea applied to cost/TTFS instead of a
solved/unsolved count.

POOLING IS AT THE RAW-RUN LEVEL, NOT AN AVERAGE-OF-RATES: for a given (algorithm, discretization
[, cost metric]) cell, every individual per-run CSV across all 3 environments (and, for the
pooled-by-metric version, all 3 models too) is loaded into one combined pool first, and the
success rate is n_success/n_total over THAT pool -- not the mean of several already-computed
percentages. This weights every real run equally regardless of which environment/model produced
it, rather than implicitly giving a thinner-data cell the same influence as a fuller one.

TWO VERSIONS, BOTH GENERATED EVERY RUN:
- "pooled" -- also pools across cost metric (length + effort together), so a discretization
  section has exactly one row per algorithm. Same rationale ttfs_cost_table.py uses for TTFS
  (solve timing doesn't depend on which cost is being minimized) -- but success/failure CAN
  differ between the two metric sweeps (this project has already found real cases where one
  metric succeeds cleanly and the other fails outright for the same cell), so this version is a
  quick top-line number, not the authoritative one when that distinction matters -- see the
  "by_metric" version for that.
- "by_metric" -- keeps length and effort as separate sub-sections instead, still averaged over
  every environment and model within each.

MODELS ARE POOLED, NOT BROKEN OUT: unlike success_rate_table.py (one table per model), this
summary deliberately collapses the model dimension too, per the request that motivated it --
"just show success rate of each algorithm at each discretization." If a per-model breakdown is
ever needed again, success_rate_table.py already covers that.

Within each section, the algorithm with the HIGHEST rate is bolded in the LaTeX output (success
rate is the one metric in this table family where bigger is better).

No Jetson rows here (unlike success_rate_table.py) -- Jetson's per-model coverage is uneven
(different discretizations exist for different models), which doesn't pool cleanly across models
the way this summary needs to.

Writes a CSV + LaTeX (.tex.txt) for each version to plots/output/tables/.

Edit DATASET_DIR / TIMEOUT_MS / OUT_DIR below, then run:
    python plots/paper/success_rate_summary_4s.py
"""
from __future__ import annotations

import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # plots/, for zephyr_common
from zephyr_common import (  # noqa: E402
    BASE_DISPLAY,
    BASE_NAMES,
    COST_METRIC_LABELS,
    COST_METRICS,
    MODEL_IDS,
    discover_discretization_dirs,
    discretization_display,
    discretization_label_from_dir,
    load_runs,
    success_rate_truncated,
    warn_on_unexpected_star_suffixes,
)

PLOTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # plots/ (this file lives in plots/paper/)

# ================================================================================================
# EDIT THESE to point at a different dataset, change the time budget, or move the output location.
# ================================================================================================
DATASET_DIR = os.path.join(PLOTS_DIR, "DATA", "ZEPHYR_30_runs")
OUT_DIR = os.path.join(PLOTS_DIR, "output", "tables")
TIMEOUT_MS = 4000.0

ENVIRONMENTS = ["house", "narrowPassage", "zigzag"]  # on-disk spelling; "empty" excluded
TABLE_PLANNERS = list(BASE_NAMES)  # all four -- no paper-specific column drops here


def pooled_success_rate(planner: str, discretization_label: str, disc_dir: str, metrics: tuple) -> tuple:
    """(n_success, n_total, rate_fraction) pooled across every model and every environment --
    see module docstring for why this is raw-run pooling, not an average of per-cell rates."""
    all_runs = []
    for model_id in MODEL_IDS:
        for env in ENVIRONMENTS:
            env_dir = os.path.join(disc_dir, env)
            if not os.path.isdir(env_dir):
                continue
            warn_on_unexpected_star_suffixes(env_dir)
            all_runs.extend(load_runs(env_dir, env, planner, model_id, discretization_label, metrics=metrics))
    return success_rate_truncated(all_runs, TIMEOUT_MS)


def fmt_pct(rate: float, bold: bool = False) -> str:
    if rate is None or (isinstance(rate, float) and math.isnan(rate)):
        return "--"
    text = f"{100.0 * rate:.1f}\\%"
    return rf"\textbf{{{text}}}" if bold else text


def best_planner(rates: dict) -> object:
    """Planner key with the HIGHEST rate, ignoring NaN/missing; None if all missing."""
    valid = {p: r for p, r in rates.items() if r is not None and not (isinstance(r, float) and math.isnan(r))}
    return max(valid, key=valid.get) if valid else None


def build_pooled_table(discretization_dirs: list) -> pd.DataFrame:
    """Version A: cost metric also pooled in -- one row per (discretization, algorithm)."""
    rows = []
    for disc_dir in discretization_dirs:
        disc_label = discretization_label_from_dir(disc_dir)
        for planner in TABLE_PLANNERS:
            n_success, n_total, rate = pooled_success_rate(planner, disc_label, disc_dir, COST_METRICS)
            rows.append({
                "Discretization": discretization_display(disc_label),
                "Algorithm": BASE_DISPLAY[planner],
                "N_Success": n_success,
                "N_Total": n_total,
                "Success_Rate_Pct": 100.0 * rate if not math.isnan(rate) else math.nan,
            })
    return pd.DataFrame(rows)


def build_by_metric_table(discretization_dirs: list) -> pd.DataFrame:
    """Version B: cost metric kept separate -- one row per (discretization, cost metric, algorithm)."""
    rows = []
    for disc_dir in discretization_dirs:
        disc_label = discretization_label_from_dir(disc_dir)
        for metric in COST_METRICS:
            for planner in TABLE_PLANNERS:
                n_success, n_total, rate = pooled_success_rate(planner, disc_label, disc_dir, (metric,))
                rows.append({
                    "Discretization": discretization_display(disc_label),
                    "CostMetric": COST_METRIC_LABELS[metric],
                    "Algorithm": BASE_DISPLAY[planner],
                    "N_Success": n_success,
                    "N_Total": n_total,
                    "Success_Rate_Pct": 100.0 * rate if not math.isnan(rate) else math.nan,
                })
    return pd.DataFrame(rows)


def render_latex_pooled(table: pd.DataFrame, discretization_labels: list) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Success Rate within {TIMEOUT_MS/1000:.0f}s (pooled over environment, model, and cost metric)}}",
        r"\label{tab:success_rate_summary_4s_pooled}",
        r"\begin{tabular}{ll r}",
        r"\toprule",
        r"Discretization & Algorithm & Success Rate \\",
    ]
    for disc_label in discretization_labels:
        section = table[table["Discretization"] == disc_label]
        rates = dict(zip(section["Algorithm"], section["Success_Rate_Pct"] / 100.0))
        best = best_planner(rates)
        lines.append(r"\midrule")
        for i, (_, row) in enumerate(section.iterrows()):
            is_best = row["Algorithm"] == best
            disc_cell = disc_label if i == 0 else ""
            rate = row["Success_Rate_Pct"] / 100.0 if not math.isnan(row["Success_Rate_Pct"]) else math.nan
            lines.append(f"{disc_cell} & {row['Algorithm']} & {fmt_pct(rate, bold=is_best)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def render_latex_by_metric(table: pd.DataFrame, discretization_labels: list) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Success Rate within {TIMEOUT_MS/1000:.0f}s, by cost metric (pooled over environment and model)}}",
        r"\label{tab:success_rate_summary_4s_by_metric}",
        r"\begin{tabular}{ll r}",
        r"\toprule",
        r"Discretization & Algorithm & Success Rate \\",
    ]
    for disc_label in discretization_labels:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{disc_label}}}}} \\")
        for metric_label in COST_METRIC_LABELS.values():
            section = table[(table["Discretization"] == disc_label) & (table["CostMetric"] == metric_label)]
            if section.empty:
                continue
            rates = dict(zip(section["Algorithm"], section["Success_Rate_Pct"] / 100.0))
            best = best_planner(rates)
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{metric_label}}}}} \\")
            for _, row in section.iterrows():
                is_best = row["Algorithm"] == best
                rate = row["Success_Rate_Pct"] / 100.0 if not math.isnan(row["Success_Rate_Pct"]) else math.nan
                lines.append(f" & {row['Algorithm']} & {fmt_pct(rate, bold=is_best)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    if not os.path.isdir(DATASET_DIR):
        raise SystemExit(f"DATASET_DIR does not exist: {DATASET_DIR!r} -- edit it at the top of this file.")
    discretization_dirs = discover_discretization_dirs(DATASET_DIR)
    if not discretization_dirs:
        raise SystemExit(f"No 'discretization<LABEL>' folders found under {DATASET_DIR!r}.")
    discretization_labels = [discretization_display(discretization_label_from_dir(d)) for d in discretization_dirs]

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Discretizations found: {discretization_labels}")
    print(f"Timeout truncation: {TIMEOUT_MS} ms")

    pooled_table = build_pooled_table(discretization_dirs)
    by_metric_table = build_by_metric_table(discretization_dirs)

    for name, table, renderer in [
        ("success_rate_summary_4s_pooled", pooled_table, render_latex_pooled),
        ("success_rate_summary_4s_by_metric", by_metric_table, render_latex_by_metric),
    ]:
        csv_path = os.path.join(OUT_DIR, f"{name}.csv")
        table.to_csv(csv_path, index=False)
        latex = renderer(table, discretization_labels)
        txt_path = os.path.join(OUT_DIR, f"{name}.tex.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(latex + "\n")
        print(f"\n% ===== {name} =====")
        print(latex)
        print(f"\nWrote {csv_path} and {txt_path}")


if __name__ == "__main__":
    main()
