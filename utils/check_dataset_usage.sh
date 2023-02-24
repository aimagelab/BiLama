#!/bin/bash
#SBATCH --job-name=check_dataset_usage
#SBATCH --output=/work/FoMo_AIISDH/vpippi/BiLama/utils/check_dataset_usage.out
#SBATCH --error=/work/FoMo_AIISDH/vpippi/BiLama/utils/check_dataset_usage.err
#SBATCH --partition=prod
#SBATCH --mem=100G

source activate LaMa
/mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/utils

srun python3 check_dataset_usage.py
