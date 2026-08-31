#!/bin/bash
#
#SBATCH --job-name="NeptuneOD Hydra Run"
#SBATCH --partition=compute
#SBATCH --time=00:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3996M
#SBATCH --account=Education-AE-MSc-AE

# module load 2025
# module load openmpi

export HYDRA_FULL_ERROR=1
module load miniconda3
conda activate NeptuneOD
cd ~/NeptuneOD
srun python scripts/generate_collection.py
