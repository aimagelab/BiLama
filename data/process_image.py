import argparse
import logging
import os
import random

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path


def check_or_create_folder(name: str):
    pass


class PatchImage:

    def __init__(self, patch_size: int, overlap_size: int, patch_size_valid: int, destination_root: str):
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        destination_root = Path(destination_root)
        self.train_folder = destination_root / f'imgs_{patch_size}/'
        self.train_gt_folder = destination_root / f'gt_imgs_{patch_size}/'
        self.valid_folder = destination_root / f'val_imgs_{patch_size_valid}/'
        self.valid_gt_folder = destination_root / f'val_gt_imgs_{patch_size_valid}/'

        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.patch_size_valid = patch_size_valid
        self.number_image = 1
        self.image_name = ""

        logging.info("Configuration patches ...")
        logging.info(f"Using Patch size: {self.patch_size} - Overlapping: {self.overlap_size}")
        logging.info(f"Using Valid patch size: {self.patch_size_valid}")
        self._create_folders()

    def _create_folders(self):
        self.train_folder.mkdir(parents=True, exist_ok=True)
        self.train_gt_folder.mkdir(parents=True, exist_ok=True)
        self.valid_folder.mkdir(parents=True, exist_ok=True)
        self.valid_gt_folder.mkdir(parents=True, exist_ok=True)
        logging.info("Configuration folders ...")

    def create_patches(self, root_original: str, root_ground_truth: str, test_dataset, validation_dataset):
        logging.info("Start process ...")
        root_original = Path(root_original)
        gt = root_original / 'gt_imgs'
        imgs = root_original / 'imgs'

        path_imgs = list(imgs.rglob('*.png')) + list(imgs.rglob('*.jpg')) + list(imgs.rglob('*.bmp'))
        for i, img in enumerate(path_imgs):
            or_img = cv2.imread(str(img))
            gt_img = cv2.imread(str(gt / img.name))
            if i < len(path_imgs) * 0.1:
                self._split_train_images(or_img, gt_img, type="valid")
            else:
                self._split_train_images(or_img, gt_img, type="train")

    def _split_train_images(self, or_img: np.ndarray, gt_img: np.ndarray, type: str):
        runtime_size = self.overlap_size if type == "train" else self.patch_size_valid
        patch_size = self.patch_size if type == "train" else self.patch_size_valid
        for i in range(0, or_img.shape[0], runtime_size):
            for j in range(0, or_img.shape[1], runtime_size):

                if i + patch_size <= or_img.shape[0] and j + patch_size <= or_img.shape[1]:
                    dg_patch = or_img[i:i + patch_size, j:j + patch_size, :]
                    gt_patch = gt_img[i:i + patch_size, j:j + patch_size, :]

                elif i + patch_size > or_img.shape[0] and j + patch_size <= or_img.shape[1]:
                    dg_patch = np.ones((patch_size, patch_size, 3)) * 255
                    gt_patch = np.ones((patch_size, patch_size, 3)) * 255

                    dg_patch[0:or_img.shape[0] - i, :, :] = or_img[i:or_img.shape[0], j:j + patch_size, :]
                    gt_patch[0:or_img.shape[0] - i, :, :] = gt_img[i:or_img.shape[0], j:j + patch_size, :]

                elif i + patch_size <= or_img.shape[0] and j + patch_size > or_img.shape[1]:
                    dg_patch = np.ones((patch_size, patch_size, 3)) * 255
                    gt_patch = np.ones((patch_size, patch_size, 3)) * 255

                    dg_patch[:, 0:or_img.shape[1] - j, :] = or_img[i:i + patch_size, j:or_img.shape[1], :]
                    gt_patch[:, 0:or_img.shape[1] - j, :] = gt_img[i:i + patch_size, j:or_img.shape[1], :]

                else:
                    dg_patch = np.ones((patch_size, patch_size, 3)) * 255
                    gt_patch = np.ones((patch_size, patch_size, 3)) * 255

                    dg_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = or_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]
                    gt_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = gt_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]
                    gt_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = gt_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]

                if type == "train":
                    cv2.imwrite(str(self.train_folder / (str(self.number_image) + '.png')), dg_patch)
                    cv2.imwrite(str(self.train_gt_folder / (str(self.number_image) + '.png')), gt_patch)
                    self.number_image += 1
                    print(self.number_image)
                elif type == "valid":
                    cv2.imwrite(str(self.valid_folder / (str(self.number_image) + '.png')), dg_patch)
                    cv2.imwrite(str(self.valid_gt_folder / (str(self.number_image) + '.png')), gt_patch)
                    self.number_image += 1
                    print(self.number_image)

    def _create_name(self, folder: str, i: int, j: int):
        return folder + self.image_name.split('.')[0] + '_' + str(i) + '_' + str(j) + '.png'


def configure_args(path_configuration: str):
    with open(path_configuration) as file:
        config_options = yaml.load(file, Loader=yaml.Loader)
        file.close()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    paths = config_options['paths']

    parser.add_argument('-destination', '--path_destination',
                        metavar='<path>',
                        type=str,
                        help=f"destination folder path with contains the patches",
                        default=paths['destination'])
    parser.add_argument('-original', '--path_original',
                        metavar='<path>',
                        type=str,
                        help="phe path witch contains the ruined images",
                        default=paths['train'])
    parser.add_argument('-ground_truth', '--path_ground_truth',
                        metavar='<path>',
                        type=str,
                        help="path which contains the ground truth images",
                        default=paths['ground_truth'])
    parser.add_argument('-size', '--patch_size',
                        metavar='<number>',
                        type=int,
                        help="size of ruined patch",
                        default=config_options['patch_size'])
    parser.add_argument('-size_valid', '--patch_size_valid',
                        metavar='<number>',
                        type=int,
                        help='size of valid image patch',
                        default=config_options['patch_size_valid'])
    parser.add_argument('-overlap', '--overlap_size',
                        metavar='<number>',
                        type=int,
                        help='overlap_size',
                        default=config_options['overlap_size'])
    parser.add_argument('-validation', '--validation_dataset',
                        metavar='<path>',
                        type=str,
                        help='folder which contains images will are used to create the validation dataset',
                        default=config_options['validation_dataset'])
    parser.add_argument('-test', '--testing_dataset',
                        dest="testing_dataset",
                        metavar='<path>',
                        type=str,
                        help='folder which contains images will are used to create the training dataset',
                        default=config_options['testing_dataset'])

    return parser.parse_args()
