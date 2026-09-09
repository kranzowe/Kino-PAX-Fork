#!/bin/bash

#SBATCH --job-name=paper_benchmarks
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
#SBATCH --output=paper_benchmarks-%j.out
#SBATCH --error=paper_benchmarks-%j.err

cd "$SLURM_SUBMIT_DIR"
bash run_paper_benchmark.sh