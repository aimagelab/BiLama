#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=dev
#SBATCH -e /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_debug_%j.err
#SBATCH -o /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_debug_%j.out
#SBATCH --mem-per-gpu=18G
#SBATCH --cpus-per-gpu=6
#SBATCH -J debug

source activate LaMa
cd /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama || exit
srun --comment="bilama_@{n_blocks}_@{operation}_@{att}" python3 train.py -c base --n_blocks @{n_blocks|9} \
              --operation "@{operation|ffc}" --attention @{att|none} --num_workers 1 --epochs 500 \
              --train_data_path \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO09 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO10 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO11 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO12 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO13 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO14 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO16 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO17 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO19 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DirtyDocuments \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/PALM \
              --test_data_path \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO18
