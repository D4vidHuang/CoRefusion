#!/bin/bash

#SBATCH --job-name="dream_flip"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-bsc-ti

# --- Load necessary DelftBlue modules ---
module load 2025 
module load python
module load cuda
module load py-numpy
module load py-torch

# --- Install missing dependencies ---
pip install --user tree-sitter-languages transformers pandas

# 注意：由于 DreamCoder 运行 768 步，生成时间会比 DiffuCoder 长，
# 因此我将 --time 增加到了 2 小时。

# --- Experiment Execution ---
cd experiments
    
# 运行 DreamCoder 脚本并将输出重定向到专用日志
srun python dreamcoder_naming_flip_step.py > dreamcoder_flip.log
