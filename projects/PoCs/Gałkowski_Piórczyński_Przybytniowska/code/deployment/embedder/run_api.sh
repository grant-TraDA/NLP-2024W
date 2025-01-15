#!/bin/bash
#SBATCH --account=<FILL_THIS>
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<FILL_THIS>
#SBATCH --job-name=embed
#SBATCH --output=<FILL_THIS>/embed.log

# this needs to be swapped with the path to the your conda path
. /mnt/evafs/groups/ganzha_23/mgalkowski/miniconda3/etc/profile.d/conda.sh
conda activate gpu_torch

uvicorn embedder_api:app --port 8080 --host 0.0.0.0
