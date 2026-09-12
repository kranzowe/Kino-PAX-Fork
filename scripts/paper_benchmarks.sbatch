#!/bin/bash

#SBATCH --job-name=paper_benchmarks
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
# was 0-08:00:00 -- run_paper_benchmark_v2.sh went from 480 to 2,880 planner-runs (6x: all three
# models, 30 runs each, one discretization per model instead of a 3-point sweep). Rough estimate:
# ~9.3h typical run-phase (96 invocations x up to ~350s each) + build-phase + per-invocation
# timeout margin -- see that script's own header for the arithmetic. 24h leaves headroom so SLURM
# itself doesn't kill the job partway through and masquerade as an unexplained hang.
#SBATCH --output=paper_benchmarks-%j.out
#SBATCH --error=paper_benchmarks-%j.err

cd "$SLURM_SUBMIT_DIR"
bash run_paper_benchmark_v2.sh