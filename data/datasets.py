from torchvision.transforms import transforms
import time
from data.TrainingDataset import TrainingDataset, PatchSquare
from data.ValidationDataset import ValidationPatchSquare, TestDataset
from data.utils import get_transform
from utils.htr_logging import get_logger
from torch.utils.data import ConcatDataset, random_split
from pathlib import Path

logger = get_logger(__file__)


def make_train_val_dataset(config: dict):
    train_data_path = config['train_data_path']
    transform_variant = config['train_transform_variant'] if 'train_transform_variant' in config else None
    patch_size = config['train_patch_size']

    logger.info(f"Train path: \"{train_data_path}\"")
    logger.info(f"Transform Variant: {transform_variant} - Training Patch Size: {patch_size}")

    transform = get_transform(transform_variant=transform_variant, output_size=patch_size)

    logger.info(f"Loading train datasets...")
    time_start = time.time()
    datasets = []
    for i, path in enumerate(train_data_path):
        logger.info(f"[{i+1}/{len(train_data_path)}] Loading train dataset from \"{path}\"")
        if Path(path).name == 'patch_square':
            datasets.append(PatchSquare(path, transform=transform))
        else:
            datasets.append(TrainingDataset(path, split_size=patch_size, transform=transform))
    logger.info(f"Loading train datasets took {time.time() - time_start:.2f} seconds")

    train_dataset = ConcatDataset(datasets)
    train_set_size = int(len(train_dataset) * 0.9)
    valid_set_size = len(train_dataset) - train_set_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_set_size, valid_set_size])

    # TODO fix that
    # for dataset in valid_dataset.dataset.datasets:
    #     dataset.transform = transforms.Compose([transforms.ToTensor()])

    logger.info(f"Training set has {len(train_dataset)} instances")

    return train_dataset, valid_dataset


def make_test_dataset(config: dict):
    test_data_path = config['test_data_path']
    patch_size = config['test_patch_size']
    stride = config['test_stride']

    transform = transforms.Compose([transforms.ToTensor()])

    logger.info(f"Loading test datasets...")
    time_start = time.time()
    datasets = []
    for path in test_data_path:
        if Path(path).name == 'patch_square':
            datasets.append(ValidationPatchSquare(path, transform=transform))
        else:
            datasets.append(TestDataset(path, patch_size=patch_size, stride=stride, transform=transform))
    logger.info(f"Loading test datasets took {time.time() - time_start:.2f} seconds")

    valid_dataset = ConcatDataset(datasets)

    logger.info(f"Test set has {len(valid_dataset)} instances")
    return valid_dataset
