#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=prod
#SBATCH -e /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_new_%j.err
#SBATCH -o /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/bilama_new_%j.out
#SBATCH --mem=24G
#SBATCH --exclude=aimagelab-srv-00,aimagelab-srv-10,vegeta,carabbaggio
#SBATCH -J bilama_new

cd /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama || exit
echo "@{name}"
scontrol update JobID="$SLURM_JOB_ID" name="@{n_blocks}_@{operation}_@{sche}_@{loss}"
srun /homes/$(whoami)/.conda/envs/LaMa/bin/python train.py -c base --n_blocks "@{n_blocks|6}" \
              --operation "@{operation|ffc}" --attention "@{att|none}" --num_workers 2 \
              --epochs 500 --skip "@{skip|add}" --unet_layers "@{unet|0}" \
              --lr_scheduler "@{sche|cosine}" --lr_scheduler_kwargs "@{sche_kwargs|dict()}" \
              --resume "@{resume|none}" --ema_rate "@{ema_rates|-1}" \
              --loss "@{loss|charbonnier}" --merge_image "@{merge|false}" \
              --train_transform_variant "@{transform|latin}" --lr_scheduler_warmup 10 \
              --epochs @{epochs|500} --patch_size @{patch_size|256} \
              --datasets \
              /scratch/fquattrini/binarization_datasets/DIBCO09 \
              /scratch/fquattrini/binarization_datasets/DIBCO10 \
              /scratch/fquattrini/binarization_datasets/DIBCO11 \
              /scratch/fquattrini/binarization_datasets/DIBCO12 \
              /scratch/fquattrini/binarization_datasets/DIBCO13 \
              /scratch/fquattrini/binarization_datasets/DIBCO14 \
              /scratch/fquattrini/binarization_datasets/DIBCO16 \
              /scratch/fquattrini/binarization_datasets/DIBCO17 \
              /scratch/fquattrini/binarization_datasets/DIBCO18 \
              /scratch/fquattrini/binarization_datasets/DirtyDocuments \
              /scratch/fquattrini/binarization_datasets/PALM \
              /scratch/fquattrini/binarization_datasets/BickleyDiary \
              /scratch/fquattrini/binarization_datasets/SMADI \
              /scratch/fquattrini/binarization_datasets/Nabuco \
              --test_dataset "@{test|DIBCO18}"

#              --validation_dataset DIBCO16
#              /scratch/fquattrini/binarization_datasets/PHIBD \
#              /scratch/fquattrini/binarization_datasets/DIBCO19 \
#              /scratch/fquattrini/binarization_datasets/ISOSBTD \
