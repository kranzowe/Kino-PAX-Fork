#!/bin/bash

#SBATCH --job-name=countingstars
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=6-00:00:00
#SBATCH --output=countingstars-%j.out
#SBATCH --error=countingstars-%j.err

cd "$(dirname "${BASH_SOURCE[0]}")"
bash run_countingstars_sweep.sh