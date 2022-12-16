import torch
from torch.utils.data import Dataset

from utils.htr_logging import get_logger

logger = get_logger(__file__)


def make_train_dataloader(train_dataset: Dataset, config: dict):
    train_dataloader_config = config['train_kwargs']
    train_data_loader = torch.utils.data.DataLoader(train_dataset, **train_dataloader_config)

    return train_data_loader


def make_valid_dataloader(valid_dataset: Dataset, config: dict):
    valid_dataloader_config = config['valid_kwargs']
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, **valid_dataloader_config)

    return valid_data_loader
