#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=prod
#SBATCH -e /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_%j.err
#SBATCH -o /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_%j.out
#SBATCH -J bilama
#SBATCH --exclude=aimagelab-srv-00,aimagelab-srv-10,vegeta,carabbaggio

conda deactivate
conda activate LaMa
cd /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama || exit
scontrol update JobID="$SLURM_JOB_ID" name="ALL_@{skip}_@{sche}_@{ema_rates}_@{loss}"
srun python3 train.py -c base --n_blocks @{n_blocks|6} \
              --operation "@{operation|ffc}" --attention @{att|none} --num_workers 2 \
              --epochs 500 --skip @{skip|add} --unet_layers @{unet|0} \
              --lr_scheduler @{sche|constant} --lr_scheduler_kwargs "@{sche_kwargs|dict()}" \
              --resume @{resume|none} --ema_rate "@{ema_rates|-1}" \
              --loss @{loss|binary_cross_entropy} --merge_image @{merge|true} \
              --train_transform_variant threshold_mask \
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
              /scratch/fquattrini/binarization_datasets/Nabuco \
              /scratch/fquattrini/binarization_datasets/SMADI \
              /scratch/fquattrini/binarization_datasets/BickleyDiary \
              /scratch/fquattrini/binarization_datasets/PHIBD \
              --test_data_path \
              /scratch/fquattrini/binarization_datasets/DIBCO18
