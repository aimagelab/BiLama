#!/bin/bash

folders="/mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO09 /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO10 /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO11 /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO12 /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO13 /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO14 /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO16 /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO17 /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO18 /mnt/beegfs/work/FoMo_AIISDH/datasets/DIBCO19 /mnt/beegfs/work/FoMo_AIISDH/datasets/new_bin_datasets/Nabuco /mnt/beegfs/work/FoMo_AIISDH/datasets/new_bin_datasets/SMADI /mnt/beegfs/work/FoMo_AIISDH/datasets/new_bin_datasets/BickleyDiary"

destination_dir="/mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/"

for folder in $folders; do
	print $folder
	cp -r $folder $destination_dir
done