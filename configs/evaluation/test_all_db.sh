#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=prod
#SBATCH -e /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/test_db_%j.err
#SBATCH -o /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/jobs/test_db_%j.out
#SBATCH -J test_db
#SBATCH --exclude=aimagelab-srv-00,aimagelab-srv-10,vegeta,carabbaggio

conda deactivate
conda activate LaMa
cd /mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama || exit
srun python3 utils/lama_test.py -c base \
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
              /scratch/fquattrini/binarization_datasets/DIBCO19 \
              /scratch/fquattrini/binarization_datasets/DirtyDocuments \
              /scratch/fquattrini/binarization_datasets/PALM \
              /scratch/fquattrini/binarization_datasets/Nabuco \
              /scratch/fquattrini/binarization_datasets/SMADI \
              /scratch/fquattrini/binarization_datasets/BickleyDiary \
              /scratch/fquattrini/binarization_datasets/PHIBD \
              /scratch/fquattrini/binarization_datasets/ISOSBTD \
              --test_dataset DIBCO18
