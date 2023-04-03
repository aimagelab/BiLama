#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=prod
#SBATCH -e /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_psquare_finetune_%j.err
#SBATCH -o /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_psquare_finetune_%j.out
#SBATCH --mem=24G
#SBATCH --exclude=aimagelab-srv-00,aimagelab-srv-10,vegeta,carabbaggio,germano,gervasoni,pippobaudo,rezzonico,ajeje,helmut,lurcanio
#SBATCH -J bilama_abla

cd /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama || exit
scontrol update JobID="$SLURM_JOB_ID" name="@{name}"
srun /homes/$(whoami)/.conda/envs/LaMa/bin/python /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/train.py -c base \
  --n_blocks @{n_blocks|3} --operation @{operation|ffc} --attention none --num_workers 2 --epochs 500 --skip cat \
  --unet_layers @{unet_layers|2} --lr_scheduler cosine --lr 1.5e-5 --lr_min 1.5e-6 --lr_scheduler_kwargs "dict()" --resume @{resume|none} --ema_rate -1 \
  --loss @{loss|CHAR} --merge_image false --train_transform_variant latin --finetuning true --lr_scheduler_warmup 10 \
  --patch_size 256 --patch_size_raw 384 --batch_size 4 --datasets \
    /mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO09 \
    /mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO10 \
    /mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO11 \
    /mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO12 \
    /mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO13 \
    /mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO14 \
    /mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO16 \
    /mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO17 \
    /mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO18 \
    /scratch/fquattrini/binarization_datasets/BickleyDiary \
    /scratch/fquattrini/binarization_datasets/SMADI \
    /scratch/fquattrini/binarization_datasets/Nabuco \
    --test_dataset @{test|DIBCO18} \
    --validation_dataset @{valid|DIBCO16}
