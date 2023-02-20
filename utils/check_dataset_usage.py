import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import psutil
import time
import sys


class TrainPatchSquare(Dataset):
    def __init__(self, path: Path, transform=None, training_only_with_patch_square=False, load_images=False):
        super(TrainPatchSquare, self).__init__()
        self.path = Path(path)
        self.transform = transform

        self.full_images = list(self.path.rglob('*/full/*'))

        if training_only_with_patch_square:
            self.mask_images = [
                self.path / full_image.parent.parent.stem / 'mask' / f'{full_image.stem.split("_")[0]}_mask.png'
                for full_image in self.full_images
            ]
        else:
            self.mask_images = [
                self.path / full_image.parent.parent.parent.stem / full_image.parent.parent.stem / 'mask'
                / f'{full_image.stem.split("_")[0]}_mask.png' for full_image in self.full_images
            ]

        self.load_images = load_images
        if self.load_images:
            tmp_full_images = []
            for i, image_path in enumerate(self.full_images):
                tmp_full_images.append(Image.open(image_path).convert("RGB"))
                if i % 1000 == 0:
                    print(f'{i + 1:>6}/{len(self.full_images)}' + '-' * 50)
                    print(f'{sys.getsizeof(tmp_full_images) / 1024 ** 3:.4f} GB')
                    print('RAM memory % used:', psutil.virtual_memory()[2])
                    print('RAM Used (GB):', psutil.virtual_memory()[3] / 1024 ** 3)
            self.full_images = tmp_full_images

            print('Loading mask images...')
            tmp_mask_images = []
            for i, mask_image_path in enumerate(self.mask_images):
                tmp_mask_images.append(Image.open(mask_image_path).convert("L"))
                if i % 1000 == 0:
                    print(f'{i + 1:>6}/{len(self.mask_images)}' + '-' * 50)
                    print(f'{sys.getsizeof(tmp_mask_images) / 1024 ** 3:.4f} GB')
                    print('RAM memory % used:', psutil.virtual_memory()[2])
                    print('RAM Used (GB):', psutil.virtual_memory()[3] / 1024 ** 3)
            self.mask_images = tmp_mask_images
            # self.full_images = [Image.open(image_path).convert("RGB") for image_path in self.full_images]
            # self.mask_images = [Image.open(mask_image_path).convert("L") for mask_image_path in self.mask_images]

    def __len__(self):
        return len(self.full_images)

    def __getitem__(self, index, merge_image=False):
        full_image = self.full_images[index]
        mask_image = self.mask_images[index]
        if not self.load_images:
            full_image = Image.open(full_image).convert("RGB")
            mask_image = Image.open(mask_image).convert("L")

        # full_image_path = self.full_images[index]
        # full_image_idx = int(full_image_path.stem.split('_')[0])
        # folder_idx = full_image_path.parent.parent.stem
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

        mask_image = mask_image.float()
        return full_image, mask_image


start_mem_perc = psutil.virtual_memory()[2]
start_mem_gb = psutil.virtual_memory()[3] / 1024 ** 3
start_time = time.time()

print(psutil.virtual_memory())
print(f'RAM memory % used: {start_mem_perc:.02f}')
print(f'RAM Used (GB): {start_mem_gb:.02f}')

print('Loading dataset...')
TrainPatchSquare('/work/FoMo_AIISDH/datasets/patch_square', transform=None, load_images=True)

end_time = time.time()
end_mem_perc = psutil.virtual_memory()[2]
end_mem_gb = psutil.virtual_memory()[3] / 1024 ** 3

print(f'RAM memory % used: {start_mem_perc:.02f} -> {end_mem_perc:.02f} ({end_mem_perc - start_mem_perc:.02f})')
print(f'RAM Used (GB): {start_mem_gb:.02f} -> {end_mem_gb:.02f} ({end_mem_gb - start_mem_gb:.02f})')
print(f'Time: {end_time - start_time:.02f} s  ({(end_time - start_time) / 60:.02f} min)')
print(psutil.virtual_memory())
print(f'Dataset size {sys.getsizeof(TrainPatchSquare) / 1024 ** 3:.2f} GB')
exit()
