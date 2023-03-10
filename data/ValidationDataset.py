import json
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

from data.utils import get_path


class ValidationPatchSquare(Dataset):
    def __init__(self, path: Path, transform=None):
        super(ValidationPatchSquare, self).__init__()
        self.path = Path(path)
        self.transform = transform

        self.full_images_paths = list(self.path.rglob('*/full/*'))

        self.mask_images_paths = [
            self.path / full_image.parent.parent.stem / 'mask' / f'{full_image.stem.split("_")[0]}_mask.png'
            for full_image in self.full_images_paths
        ]

        # self.full_images = [Image.open(image_path).convert("RGB") for image_path in self.full_images]
        # self.mask_images = [Image.open(mask_image_path).convert("L") for mask_image_path in self.mask_images]

    def __len__(self):
        return len(self.full_images_paths)

    def __getitem__(self, index):
        full_image = Image.open(self.full_images_paths[index]).convert("RGB")
        mask_image = Image.open(self.mask_images_paths[index]).convert("L")

        # full_image_path = self.full_images[index]
        # full_image_idx = int(full_image_path.stem.split('_')[0])
        # folder_idx = full_image_path.parent.parent.stem
        # with open(self.path / folder_idx / 'metadata' / f'{full_image_idx}_metadata.json', 'r') as f:
        #     metadata = json.load(f)

        if self.transform:
            transform = self.transform({'image': full_image, 'gt': mask_image})
            full_image = transform['image']
            mask_image = transform['gt']

        mask_image = mask_image.float()
        return full_image, mask_image


class ValidationDataset(Dataset):

    def __init__(self, data_path, split_size=256, patch_size=384, transform=None, load_data=True):
        super(ValidationDataset, self).__init__()
        self.imgs = list(Path(data_path).rglob(f'val_imgs_{split_size}/*'))
        self.gt_imgs = [img_path.parent.parent / ('val_gt_' + img_path.parent.name[4:]) / img_path.name for img_path in self.imgs]

        self.load_data = load_data
        if self.load_data:
            self.imgs = [Image.open(img_path).convert("RGB") for img_path in self.imgs]
            self.gt_imgs = [Image.open(gt_img_path).convert("L") for gt_img_path in self.gt_imgs]

        self.split_size = split_size
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.load_data:
            sample = self.imgs[index]
            gt_sample = self.gt_imgs[index]
        else:
            sample = Image.open(self.imgs[index]).convert("RGB")
            gt_sample = Image.open(self.gt_imgs[index]).convert("L")

        if self.transform:
            transform = self.transform({'image': sample, 'gt': gt_sample})
            sample = transform['image']
            gt_sample = transform['gt']

        gt_sample = gt_sample.float()

        return sample, gt_sample

class ErrorValidationDataset(ValidationDataset):
    def __init__(self, data_path, patch_size=256, transform=None, discard_padding=True):
        super(ValidationDataset, self).__init__()
        self.imgs = list((Path(data_path) / f'imgs_{patch_size}').iterdir())
        self.gt_imgs = [Path(data_path) / f'gt_imgs_{patch_size}' / img_path.name for img_path in self.imgs]

        self.load_data = True
        self.imgs = [Image.open(img_path).convert("RGB") for img_path in self.imgs]
        self.gt_imgs = [Image.open(gt_img_path).convert("L") for gt_img_path in self.gt_imgs]

        def has_padding(img):
            img = np.array(img).mean(-1)
            bottom_padding = (img[-1, :] == 255.0).all()
            if bottom_padding: return True
            right_padding = (img[:, -1] == 255.0).all()
            if right_padding: return True
            return False

        if discard_padding:
            self.couples = [(img, gt) for img, gt in zip(self.imgs, self.gt_imgs) if not has_padding(img)]
            self.imgs, self.gt_imgs = zip(*self.couples)

        self.split_size = patch_size
        self.transform = transform

    def __getitem__(self, index):
        if self.load_data:
            sample = self.imgs[index]
            gt_sample = self.gt_imgs[index]
        else:
            sample = Image.open(self.imgs[index]).convert("RGB")
            gt_sample = Image.open(self.gt_imgs[index]).convert("L")

        if self.transform:
            transform = self.transform({'image': sample, 'gt': gt_sample})
            sample = transform['image']
            gt_sample = transform['gt']

        gt_sample = gt_sample.float()

        return sample, gt_sample
