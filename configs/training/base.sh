#!/bin/bash
#SBATCH --gres=gpu:2080:1
#SBATCH --partition=prod
#SBATCH -e /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/lama_@{n_blocks}_@{operation}_@{att}_%j.err
#SBATCH -o /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/lama_@{n_blocks}_@{operation}_@{att}_%j.out
#SBATCH -J bilama

source activate LaMa
cd /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama || exit

#export PYTHONPATH=$PYTHONPATH:/mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama
srun --comment="bilama_@{n_blocks}_@{operation}_@{att}" python3 train.py -c base --n_blocks @{n_blocks|9} \
              --operation "@{operation|ffc}" --attention @{att|none} --num_workers 4 --epochs 500 \
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
              /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO18 \
