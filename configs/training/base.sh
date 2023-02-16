
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=prod
#SBATCH -e /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_%j.err
#SBATCH -o /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_%j.out
#SBATCH -J bilama
#SBATCH --exclude=aimagelab-srv-00,aimagelab-srv-10,vegeta,carabbaggio

source activate LaMa
cd /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama || exit
scontrol update JobID="$SLURM_JOB_ID" name="bilama_@{n_blocks}_@{operation}_@{att}_SKIP@{skip}_patch"
srun python3 train.py -c base --n_blocks @{n_blocks|9} \
              --operation "@{operation|ffc}" --attention @{att|none} --num_workers 2 --epochs 500 --skip @{skip} --unet_layers @{unet} \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/DIBCO09 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/DIBCO10 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/DIBCO11 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/DIBCO12 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/DIBCO13 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/DIBCO14 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/DIBCO16 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/DIBCO17 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/DIBCO19 \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/DirtyDocuments \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/PALM \
              --test_data_path \
              /mnt/beegfs/work/FoMo_AIISDH/datasets/datasets_refactoring/DIBCO18
