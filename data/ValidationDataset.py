import os
from pathlib import Path
import torchvision.transforms.functional as functional
from PIL import Image
from torch.utils.data import Dataset

from data.utils import get_path

class ValidationPatchSquare(Dataset):
    pass

class ValidationDataset(Dataset):

    def __init__(self, data_path, patch_size=256, stride=256, transform=None):
        super(ValidationDataset, self).__init__()
        self.imgs = list(Path(data_path).rglob(f'imgs/*'))
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        gt_img_path = img_path.parent.parent / 'gt_imgs' / img_path.name

        sample = Image.open(img_path).convert("RGB")
        gt_sample = Image.open(gt_img_path).convert("L")

        # Create patches
        padding_bottom = ((sample.height // self.patch_size) + 1) * self.patch_size - sample.height
        padding_right = ((sample.width // self.patch_size) + 1) * self.patch_size - sample.width

        tensor_padding = functional.to_tensor(sample).unsqueeze(0)
        batch, channels, _, _ = tensor_padding.shape
        tensor_padding = functional.pad(img=tensor_padding, padding=[0, 0, padding_right, padding_bottom], fill=1)
        patches = tensor_padding.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        num_rows = patches.shape[3]
        patches = patches.reshape(batch, channels, -1, self.patch_size, self.patch_size)

        if self.transform:
            sample = self.transform(sample)
            gt_sample = self.transform(gt_sample)

        item = {
            'image_name': str(self.imgs[index]),
            'sample': sample,
            'num_rows': num_rows,
            'samples_patches': patches,
            'gt_sample': gt_sample
        }

        return item
