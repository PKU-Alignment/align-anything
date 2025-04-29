#!/bin/bash

#SBATCH --partition=verify

#SBATCH --nodes=1

#SBATCH --gres=gpu:8

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=32

#SBATCH --mem=200G

#SBATCH --job-name=debug_dpo

#SBATCH --output=debug_dpo_%j.log

#SBATCH --error=debug_dpo_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

#SBATCH --mail-user=lht1919810@gmail.com

# 激活conda环境

source $(conda info --base)/etc/profile.d/conda.sh
conda activate formal_pipeline

# 设置工作目录

cd ~/code/align-anything/scripts

# 设置GPU环境变量

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 执行训练脚本

bash llava/llava_dpo.sh

# 作业结束时通知

echo "Job completed at $(date)"
