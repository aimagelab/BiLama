#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=prod
#SBATCH -e /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_psquare_%j.err
#SBATCH -o /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_psquare_%j.out
#SBATCH --mem=24G
#SBATCH --exclude=aimagelab-srv-00,aimagelab-srv-10,vegeta,carabbaggio,germano,gervasoni,pippobaudo,rezzonico,ajeje,helmut,lurcanio
#SBATCH -J bilama_abla

cd /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama || exit
scontrol update JobID="$SLURM_JOB_ID" name="@{name}_128"
srun /homes/$(whoami)/.conda/envs/LaMa/bin/python /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/train.py -c base \
  --n_blocks @{n_blocks|3} --operation @{operation|ffc} --attention none --num_workers 2 --epochs 500 --skip cat \
  --unet_layers @{unet_layers|2} --lr_scheduler cosine --lr_scheduler_kwargs "dict()" --resume @{resume|none} --ema_rate -1 \
  --loss @{loss|CHAR} --merge_image false --train_transform_variant latin --lr_scheduler_warmup 10 \
  --patch_size 256 --patch_size_raw 384 --batch_size 4 --datasets \
    /mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/patch_square \
    /mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO18 \
    --test_dataset @{test|DIBCO18}
