#!/bin/bash

#SBATCH --job-name=countingstars
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=countingstars-%j.out
#SBATCH --error=countingstars-%j.err

bash run_paper_benchmark.sh