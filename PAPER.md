# Paper artifacts index

Everything the paper's figures/tables depend on lives in three `paper/` subfolders. Everything else
in `scripts/`, `examples/gpu/`, `plots/` is non-paper (older benchmarks, tuning sweeps, superseded plots).

| Kind | Location |
|---|---|
| Runner scripts (build + run on GPU/cluster/Jetson) | `scripts/paper/` |
| Benchmark / dump CUDA tools (CMake targets below) | `examples/gpu/paper/` |
| Plotting + tables (run on the analysis machine) | `plots/paper/` |
| Shared plotting helper | `plots/zephyr_common.py` (stays in `plots/`: also imported by non-paper plots) |
| Local data / outputs (gitignored, never moved) | `plots/DATA/`, `plots/output/` |

## Paper planners

`KPAX` (Kino-PAX), `KinoPaxPlus` (Kino-PAX+), `KinoPaxSTARTrue` (SimpleCombo, token
`KinoPaxSTARTrue_cap100_anc1`), `CountingStars` (Kino-PAX#, token `CountingStars_bs120_bf40_ef150_cf750_hg1`).
Shared dependencies (all stay at the repo root): `Planner`, `Graph`, `KinoPaxPlusRegions`, `spatialHash`,
`collisionCheck`, `statePropagator` (+ `SpatialHash`), `helper`. Tree dumps use only KPAX, KinoPaxPlus, CountingStars.

`ChainReaction` (`include/planners/ChainReaction.cuh`, `src/planners/ChainReaction.cu`) is an experimental minimal
baseline: CountingStars with everything but the chain removed (budget B of candidate propagations per iteration, branching
factor B / |frontier|, accept = collision-free, no reactivation, no regions). It is NOT in `zephyr_common.BASE_NAMES`, so the
paper figures above are unaffected; only `chain_reaction_compare.py` plots it. Runner defaults: Dubins Airplane, 'large'
discretization, path-length cost, house/narrowPassage/zigzag, 5 runs, constant B = 5000 (`--budget`). It uses the main paper
data's settings (3M-node tree, CountingStars ramp 700, 4 s cap), not v3's 1M/210/3 s.

## Pipeline: runner -> target -> data -> plot

| Runner (`scripts/paper/`) | CMake target (`examples/gpu/paper/`) | Writes (under `build/`) | Copy to (`plots/DATA/`) | Plots (`plots/paper/`) |
|---|---|---|---|---|
| `paper_benchmarks.sbatch` -> `run_paper_benchmark_v2.sh` | `PaperBenchmarkV2` | `Data/Benchmarks/PaperBenchmarkV2/` | `ZEPHYR_*` | see "benchmark plots" below |
| `run_paper_benchmark_v3.sh` | `PaperBenchmarkV3` | `Data/Benchmarks/PaperBenchmarkV3/<COARSE\|FINE\|TINY>/<env>/` | `ZEPHYR_30_runs/` | benchmark plots |
| `run_paper_benchmark_v3_quad.sh` | `PaperBenchmarkV3Quad` | `Data/Benchmarks/PaperBenchmarkV3/MEDIUM/<env>/` | (Quad "medium") | benchmark plots |
| `run_jetson_paper_benchmark_v3.sh` | `JetsonPaperBenchmarkV3` | `Data/Benchmarks/PaperBenchmarkV3Jetson/FINE/<env>/` | `JETSON_20_runs/` | `ttfs_panel_vs_kpax_4s`, `ttfs_cost_table`, `cost_table_wide` |
| `run_tree_checkpoint_dump.sh` | `TreeCheckpointDump` | `Data/Viz/TreeCheckpoints/` | `tree_checkpoint/<run>/` | `tree_checkpoint.py` |
| `run_tree_growth_animation_dump.sh` | `TreeGrowthAnimationDump` | `Data/Viz/TreeGrowthAnimation/` | `tree_animation*/TreeGrowthAnimation/` | `tree_growth_animation.py` |
| `run_sample_trajectories.sh` | `TreeCheckpointDump` (x3 envs) | `Data/Viz/environment_trajectories/` | `environment_trajectories/` | `view_environment.py`, `plotly_env_viz.py` |
| `run_chain_reaction_benchmark.sh` | `ChainReactionBenchmark` | `Data/Benchmarks/ChainReaction/COARSE/<env>/` | `CHAIN_REACTION_runs/discretizationCOARSE/<env>/` | `chain_reaction_compare.py` |
| `run_chain_reaction_benchmark.sh --viz` | `TreeGrowthAnimationDump --planners=CountingStars,ChainReaction` | `Data/Viz/ChainReaction/` | `tree_animation_chain_reaction/TreeGrowthAnimation/` | `chain_reaction_growth_animation.py` (`--static` for checkpoint panels) |

`run_tree_checkpoint_dump.sh` = static figure: 6 wall-clock checkpoints (50-4000ms), 4% of leaf paths.
`run_tree_growth_animation_dump.sh` = animation: a checkpoint every 30ms to 3000ms, 1% seed-lineage
sampling (stable across frames). The animation is rendered by `plots/paper/tree_growth_animation.py`
(there is no shell wrapper: dump on the GPU box, render on the plotting box).

### Benchmark plots (read `plots/DATA/ZEPHYR_30_runs`, optionally `JETSON_20_runs`)

`ttfs_panel_vs_kpax_4s.py`, `cost_panel_vs_kinopaxplus_4s.py`, `success_rate_summary_4s.py`,
`ttfs_cost_table.py`, `cost_table_wide.py`, `cost_vs_time.py`, `tree_growth_rate.py`.
`convert_to_jpg.py` converts everything in `plots/output/` to `plots/output_jpg/`.

## Chain Reaction: what to run

```bash
bash scripts/paper/run_chain_reaction_benchmark.sh            # benchmark only (no tree dumps)
bash scripts/paper/run_chain_reaction_benchmark.sh --viz      # also dump tree growth (slow: copies the tree each 30 ms checkpoint)
```

Console output per run: `TTFS` (planner time + iteration of the first solution, or `none`), total planner time, iterations and
average ms/iteration, final tree size, cost, and why the run stopped (`tree_full`, `planner_stopped`, `time_cap`,
`max_iterations`). After each planner/environment a `>>` line gives solved k/n, mean TTFS (solved runs), mean total time and
mean ms/iteration. Both planners run up to 3000 iterations, 4 s, 3M-node tree.

| Script | Reads | Produces (`plots/output/...`) |
|---|---|---|
| `plots/paper/chain_reaction_compare.py` | `plots/DATA/CHAIN_REACTION_runs/discretizationCOARSE/<env>/` | `chain_reaction/chain_reaction_vs_countingstars_DubinsAirplane_coarse.{png,svg}`: per environment, best cost vs time, tree size vs time, time per iteration vs time (legend shows each planner's average ms/iter). `chain_reaction/chain_reaction_summary.csv`: n_runs, success_rate, mean/std TTFS, mean final cost, final tree size, run time, iterations, mean ms/iter. |
| `plots/paper/chain_reaction_growth_animation.py` (needs `--viz` data) | `plots/DATA/tree_animation_chain_reaction/TreeGrowthAnimation/` | `chain_reaction_animation/tree_growth_animation_zigzag.mp4` (or `.gif`); `--static` gives per-checkpoint panels in `chain_reaction_animation/static/`. |

The other paper plots (`ttfs_panel_vs_kpax_4s.py`, `cost_vs_time.py`, `tree_growth_rate.py`, ...) iterate `zephyr_common.BASE_NAMES`, which does
not include Chain Reaction, so they do not plot it.

## Running

```bash
# GPU / cluster (from the repo root unless noted)
bash scripts/paper/run_paper_benchmark_v3.sh
(cd scripts/paper && sbatch paper_benchmarks.sbatch)     # sbatch must be submitted from scripts/paper
bash scripts/paper/run_tree_checkpoint_dump.sh

# analysis machine
python plots/paper/ttfs_panel_vs_kpax_4s.py
python plots/paper/tree_growth_animation.py              # VIEW_MODE = "top_down" | "isometric"
```

Build only the paper targets:

```bash
cd build && cmake .. && make PaperBenchmarkV2 PaperBenchmarkV3 JetsonPaperBenchmarkV3 PaperBenchmarkV3Quad TreeCheckpointDump TreeGrowthAnimationDump
```
