#!/bin/bash
#SBATCH --job-name=llava_dpo          
#SBATCH --output=/path/to/your/align-anything/llava_dpo_%j.log    
#SBATCH --error=/path/to/your/align-anything/llava_dpo_%j.log      
#SBATCH --partition=your_partition_name
#SBATCH --account=your_account_name   
#SBATCH --nodes=1                           
#SBATCH --gres=gpu:8                  

# follow the same steps to activate the conda environment
conda activate align-anything
export PYTHONPATH=/path/to/your/align-anything/
cd /path/to/your/align-anything/scripts

# Run the script
srun  llava_dpo.sh