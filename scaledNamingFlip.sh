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
module load 2024r1
module load python
module load cuda
module load py-numpy
module load py-torch

# --- Setup Virtual Environment ---
# Creating a persistent virtual environment in your home directory
VENV_DIR="$HOME/venv-corefusion"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Ensure pip is up to date and install missing dependencies
python3 -m pip install --upgrade pip
python3 -m pip install tree-sitter-languages transformers pandas

# --- Experiment Execution ---
# The script uses relative paths like '../data/test.csv', 
# so we navigate to the 'experiments' directory first.
cd experiments

# Run the python script and redirect output to a log file
# Since the venv is active, 'python3' points to the venv's python
srun python3 scaled_naming_flip_step.py > naming_flip.log
