#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=prod
#SBATCH -e /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_%j.err
#SBATCH -o /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_%j.out
#SBATCH -J bilama
#SBATCH --exclude=aimagelab-srv-00,aimagelab-srv-10,vegeta,carabbaggio

source activate LaMa
cd /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama || exit
scontrol update JobID="$SLURM_JOB_ID" name="bilama_@{n_blocks}_@{operation}_@{att}_SKIP@{skip}"
srun python3 train.py -c base --n_blocks @{n_blocks|6} \
              --operation "@{operation|ffc}" --attention @{att|none} --num_workers 2 \
              --epochs 500 --skip @{skip|none} --unet_layers @{unet|0} \
              --apply_threshold_to @{apply_thres} --threshold @{thres} \
              --lr_scheduler @{sche} --lr_scheduler_kwargs "@{sche_kwargs}" \
              --load_data @{load_data} --resume @{resume} \
              --n_downsampling @{n_down} --ema_rate "@{ema_rates}" \
              --loss @{loss} \
              --train_data_path \
              /scratch/fquattrini/binarization_datasets/DIBCO09 \
              /scratch/fquattrini/binarization_datasets/DIBCO10 \
              /scratch/fquattrini/binarization_datasets/DIBCO11 \
              /scratch/fquattrini/binarization_datasets/DIBCO12 \
              /scratch/fquattrini/binarization_datasets/DIBCO13 \
              /scratch/fquattrini/binarization_datasets/DIBCO14 \
              /scratch/fquattrini/binarization_datasets/DIBCO16 \
              /scratch/fquattrini/binarization_datasets/DIBCO17 \
              /scratch/fquattrini/binarization_datasets/DIBCO19 \
              /scratch/fquattrini/binarization_datasets/DirtyDocuments \
              /scratch/fquattrini/binarization_datasets/PALM \
              --test_data_path \
              /scratch/fquattrini/binarization_datasets/DIBCO18
