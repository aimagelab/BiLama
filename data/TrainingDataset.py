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
        full_image_idx = int(full_image_path.stem.split('_')[0])
        folder_idx = full_image_path.parent.parent.stem
        full_image = Image.open(full_image_path).convert("RGB")
        # background_image = Image.open(self.path / folder_idx / 'bg' / f'{full_image_idx}_bg.jpg').convert("RGB")
        mask_image = Image.open(self.path / folder_idx / 'mask' / f'{full_image_idx}_mask.png').convert("L")

        # with open(self.path / folder_idx / 'metadata' / f'{full_image_idx}_metadata.json', 'r') as f:
        #     metadata = json.load(f)

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

    def __init__(self, data_path, split_size=256, patch_size=384, transform=None):
        super(TrainingDataset, self).__init__()
        self.imgs = list(Path(data_path).rglob(f'imgs_{patch_size}/*'))
        self.gt_imgs = [img_path.parent.parent / ('gt_' + img_path.parent.name) / img_path.name for img_path in self.imgs]

        self.imgs = [Image.open(img_path).convert("RGB") for img_path in self.imgs]
        self.gt_imgs = [Image.open(gt_img_path).convert("L") for gt_img_path in self.gt_imgs]

        self.split_size = split_size
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index, merge_image=True):
        sample = self.imgs[index]
        gt_sample = self.gt_imgs[index]

        if self.transform:
            transform = self.transform({'image': sample, 'gt': gt_sample})
            sample = transform['image']
            gt_sample = transform['gt']

        # Merge two images
        if merge_image:
            random_index = random.randint(0, len(self.imgs) - 1)
            random_sample, random_gt_sample = self.__getitem__(index=random_index, merge_image=False)

            sample = np.minimum(sample, random_sample)
            gt_sample = np.minimum(gt_sample, random_gt_sample)

        gt_sample = gt_sample.float()
        return sample, gt_sample
