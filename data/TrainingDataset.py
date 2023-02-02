import json
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

from data.utils import get_path


class PatchSquare(Dataset):

    def __init__(self, path: str, transform=None):

        super(PatchSquare, self).__init__()
        self.path = Path(path)
        self.transform = transform

        self.full_images = list(self.path.rglob('*/full/*'))

    def __len__(self):
        return len(self.full_images)

    def __getitem__(self, index, merge_image=False):
        full_image_path = self.full_images[index]
        full_image_idx = full_image_path.stem
        folder_idx = full_image_path.parent.parent.stem
        full_image = Image.open(full_image_path).convert("RGB")
        background_image = Image.open(self.path / folder_idx / 'bg' / f'{full_image_idx}_bg.jpg').convert("RGB")
        mask_image = Image.open(self.path / folder_idx / 'mask' / f'{full_image_idx}_mask.jpg').convert("L")

        with open(self.path / folder_idx / 'metadata' / f'{full_image_idx}_metadata.json', 'r') as f:
            metadata = json.load(f)

        if self.transform:
            transform = self.transform({'image': full_image, 'gt': mask_image})
            full_image = transform['image']
            mask_image = transform['gt']

        # Merge two images
        if merge_image:
            random_index = random.randint(0, len(self.full_images) - 1)
            random_sample, random_gt_sample = self.__getitem__(index=random_index, merge_image=False)

            full_image = np.minimum(full_image, random_sample)
            mask_image = np.minimum(mask_image, random_gt_sample)

        return full_image, mask_image


class TrainingDataset(Dataset):

    def __init__(self, root_dg_dir: str, root_gt_dir: str, split_size=256, transform=None):
        assert len(os.listdir(root_dg_dir)) == len(os.listdir(root_gt_dir))

        super(TrainingDataset, self).__init__()
        self.root_dg_dir = root_dg_dir
        self.root_gt_dir = root_gt_dir
        self.split_size = split_size
        self.transform = transform

        self.path_images = os.listdir(self.root_dg_dir)

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index, merge_image=True):
        path_image_deg = get_path(self.root_dg_dir, self.path_images, index)
        path_image_gtr = get_path(self.root_gt_dir, self.path_images, index)

        sample = Image.open(path_image_deg).convert("RGB")
        gt_sample = Image.open(path_image_gtr).convert("L")

        if self.transform:
            transform = self.transform({'image': sample, 'gt': gt_sample})
            sample = transform['image']
            gt_sample = transform['gt']

        # Merge two images
        if merge_image:
            random_index = random.randint(0, len(self.path_images) - 1)
            random_sample, random_gt_sample = self.__getitem__(index=random_index, merge_image=False)

            sample = np.minimum(sample, random_sample)
            gt_sample = np.minimum(gt_sample, random_gt_sample)

        return sample, gt_sample
