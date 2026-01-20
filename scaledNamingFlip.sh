#!/bin/bash

#SBATCH --job-name="naming_flip"
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-bsc-ti

# --- Load necessary DelftBlue modules ---
# We use the 2025 software stack as per the tutorial
module load 2025 
module load python
module load cuda
module load py-numpy
module load py-torch

# --- Install missing dependencies ---
# Using --user to install in your local home directory if not present in modules
pip install --user tree-sitter-languages transformers pandas

# --- Experiment Execution ---
# The script uses relative paths like '../data/test.csv', 
# so we navigate to the 'experiments' directory first.
cd experiments

# Run the python script and redirect output to a log file
srun python scaled_naming_flip_step.py > naming_flip.log
