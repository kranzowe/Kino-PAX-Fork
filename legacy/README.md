# legacy/ -- archived, not built

This folder holds the planners and tooling that are **not** part of the `ReactionsOnly` branch. Nothing here is
referenced by `CMakeLists.txt`, so none of it is compiled. Paths mirror the original repo layout
(`legacy/src/planners/X.cu` was `src/planners/X.cu`), so restoring something is a `git mv` back plus the CMake lines.

## What the branch keeps

`KPAX`, `KinoPaxPlus`, `CountingStars` (Kino-PAX#), `ChainReaction`, `FissionReaction`, their shared libraries, the
paper benchmarks/dumps (`examples/gpu/paper/`), their runners (`scripts/paper/`) and the plots (`plots/`, `plots/paper/`).
See `../PAPER.md`.

## What is archived here

| Folder | Contents |
|---|---|
| `include/planners/`, `src/planners/` | `KPAXCap`, `KinoPaxPlusSpatialHash`, `KinoPaxSTAR` and all its variants (`COMBO`, `CleanCost`, `NoGoalBias`, `NoPruneNoSpatialHash`, `True` = SimpleCombo, `TrueWeightedCost`, `WeightedCost`, `costprune`, `costprunenoseed`, `noseed`, `sparsefill`), `PruneKPAX`, `originalKPAX`, `originalPlanner` |
| `include/ReKino/`, `src/ReKino/` | ReKino, ReKinoLite |
| `examples/gpu/` | `rekino_main`, `comprehensive_benchmark`, `kinopaxplus_{comprehensive,delta,sh}_benchmark`, `planner_comparison_benchmark`, `jetson_smoke_test`, `kinopaxstar_{large_benchmark,cost_benchmark,comparison,accept_breakdown,cost_tuning_sweep,combo_tuning_sweep}` |
| `scripts/` | the matching `run_*.sh` runners, MATLAB `process_*`/`plot_*`/`visualize_*` scripts, `cross_check_combo_grid.py`, `check_fanout_budget.py`, `VARIANTS.md` |
| `viz/`, `benchmarking/` | the older MATLAB tree-visualization and benchmarking-helper trees |
| `KINOPAXPLUS_BENCHMARK_README.md` | docs for the KinoPaxPlus delta/SH benchmarks |

## Deleted rather than archived

Superseded code that only used the kept planners was removed outright; it is still in git history (branch `MovingOn` and
earlier): the CountingStars tuning sweep (`countingstars_sweep.cu`, `run_countingstars_sweep.sh`, `countingstars.sbatch`,
its MATLAB and cross-check scripts), the v1 `paper_benchmark.cu` with its runner and MATLAB, the 8-iteration
`tree_growth_dump.cu` with its runner/MATLAB, an empty `plots/plots.py`, and MATLAB autosave (`.asv`) files.
Some provenance comments in the kept runners/benchmarks still mention those files by name.

## Restoring something

1. `git mv legacy/src/planners/<X>.cu src/planners/` (and the matching `.cuh` under `include/planners/`).
2. Add the `.cu` to `SOURCE_FILES` in `CMakeLists.txt`; for an executable also add its
   `add_executable` / `target_link_libraries` / `CUDA_SEPARABLE_COMPILATION` / `CUDA_ARCHITECTURES` lines.
3. To bring SimpleCombo back into the paper benchmarks/plots, the removed pieces are the `KinoPaxSTARTrue` include,
   `trueLabel()`, `benchmarkKinoPaxSTARTrue`/`runKinoPaxSTARTrueBenchmark`, the `--only-kinopaxstartrue` flag in the four
   `examples/gpu/paper/*benchmark*.cu`, the runners' `PLANNER_FLAGS`/`PLANNER_NAMES` entries, and `SIMPLECOMBO` in
   `plots/zephyr_common.py`. `git diff MovingOn..ReactionsOnly` shows each of them.
