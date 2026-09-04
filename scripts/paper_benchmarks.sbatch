#!/bin/bash

#SBATCH --job-name=paper_benchmarks
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=6-00:00:00
#SBATCH --output=paper_benchmarks-%j.out
#SBATCH --error=paper_benchmarks-%j.err

cd "$(dirname "${BASH_SOURCE[0]}")"
bash run_paper_benchmark.sh