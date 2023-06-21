import errno
import numpy as np
import math
import os
import wandb
from pathlib import Path
import copy
import random

import torch
import torch.utils.data
from torchvision.transforms import functional
from typing_extensions import TypedDict
import matplotlib.pyplot as plt

from data.dataloaders import make_train_dataloader, make_valid_dataloader, make_test_dataloader
from data.datasets import make_train_dataset, make_val_dataset, make_test_dataset
from data.utils import reconstruct_ground_truth
from modules.FFC import LaMa
from trainer.EMA import params_to_model_state_dict, model_state_dict_to_params
from trainer.Losses import make_criterion
from trainer.Optimizers import make_optimizer
from trainer.Schedulers import make_lr_scheduler
from trainer.Validator import Validator
from utils.htr_logging import get_logger


def main():
    config = {'input_channels': 3, 'output_channels': 1, 'kind_loss': 'binary_cross_entropy', 'kind_optimizer': 'Adam',
              'train_transform_variant': 'latin',
              'path_checkpoint': '/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/checkpoints/',
              'learning_rate': 0.00015, 'threshold': 0.5,
              'optimizer': {'betas': [0.9, 0.95], 'eps': 1e-08, 'weight_decay': 0.05, 'amsgrad': False},
              'init_conv_kwargs': {'ratio_gin': 0, 'ratio_gout': 0},
              'down_sample_conv_kwargs': {'ratio_gin': 0, 'ratio_gout': 0},
              'resnet_conv_kwargs': {'ratio_gin': 0.75, 'ratio_gout': 0.75}, 'train_patch_size': 256,
              'train_log_every': 100, 'train_max_value': 500,
              'train_kwargs': {'shuffle': True, 'pin_memory': True, 'num_workers': 2, 'batch_size': 2},
              'valid_patch_size': 256, 'valid_stride': 256,
              'valid_kwargs': {'shuffle': False, 'pin_memory': True, 'num_workers': 2, 'batch_size': 1},
              'test_patch_size': 256, 'test_stride': 256,
              'test_kwargs': {'shuffle': False, 'pin_memory': True, 'num_workers': 2, 'batch_size': 1},
              'finetuning': False,
              'experiment_name': 'FFC_3RB_catSKIP_2UL_3DS_CHARLOSS_cosiSCHE_f8e3', 'use_convolutions': False,
              'skip_connections': 'cat',
              'unet_layers': 2, 'n_blocks': 3, 'n_downsampling': 3, 'cross_attention': 'none', 'losses': ['CHAR'],
              'lr_scheduler': 'cosine',
              'lr_scheduler_kwargs': {}, 'lr_scheduler_warmup': 2, 'learning_rate_min': 1.5e-05, 'seed': 742,
              'valid_data_path': ['/mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO16'],
              'train_data_path': ['/mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO09'],
              'test_data_path': ['/mnt/beegfs/scratch/fquattrini/binarization_datasets_one_for_eval/DIBCO17'],
              'aux_data_path': [], 'merge_image': False,
              'cross_attention_args': {'num_heads': 4, 'attention_channel_scale_factor': 1},
              'train_batch_size': 2, 'valid_batch_size': 1, 'test_batch_size': 1, 'num_epochs': 20, 'patience': 60,
              'ema_rate': None,
              'apply_threshold_to_train': False, 'apply_threshold_to_valid': False, 'apply_threshold_to_test': True,
              'load_data': True,
              'train_patch_size_raw': 384, 'num_params': 17575809}

    learning_rate = config['learning_rate']

    model = LaMa(input_nc=config['input_channels'], output_nc=config['output_channels'],
                 n_downsampling=config['n_downsampling'], init_conv_kwargs=config['init_conv_kwargs'],
                 downsample_conv_kwargs=config['down_sample_conv_kwargs'],
                 resnet_conv_kwargs=config['resnet_conv_kwargs'], n_blocks=config['n_blocks'],
                 use_convolutions=config['use_convolutions'],
                 cross_attention=config['cross_attention'],
                 cross_attention_args=config['cross_attention_args'],
                 skip_connections=config['skip_connections'],
                 unet_layers=config['unet_layers'], )

    optimizer = make_optimizer(model, learning_rate, config['kind_optimizer'], config['optimizer'])
    lr_scheduler = make_lr_scheduler(config['lr_scheduler'], optimizer, config['lr_scheduler_kwargs'],
                                     config['lr_scheduler_warmup'], config)

    num_epochs = config['num_epochs']

    lrs = []
    stop_point = 10
    for i in range(num_epochs):
        lr_scheduler.step()
        lrs.append(lr_scheduler.get_lr()[0])
        if i == stop_point-1:
            break
    print(lrs)
    # plt.plot(np.arange(stop_point), lrs)
    # plt.show()

    torch.save(i, 'epoch.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth')
    torch.save(lr_scheduler.state_dict(), 'lr_scheduler.pth')

    model = LaMa(input_nc=config['input_channels'], output_nc=config['output_channels'],
                 n_downsampling=config['n_downsampling'], init_conv_kwargs=config['init_conv_kwargs'],
                 downsample_conv_kwargs=config['down_sample_conv_kwargs'],
                 resnet_conv_kwargs=config['resnet_conv_kwargs'], n_blocks=config['n_blocks'],
                 use_convolutions=config['use_convolutions'],
                 cross_attention=config['cross_attention'],
                 cross_attention_args=config['cross_attention_args'],
                 skip_connections=config['skip_connections'],
                 unet_layers=config['unet_layers'], )
    optimizer = make_optimizer(model, learning_rate, config['kind_optimizer'], config['optimizer'])
    lr_scheduler = make_lr_scheduler(config['lr_scheduler'], optimizer, config['lr_scheduler_kwargs'],
                                     config['lr_scheduler_warmup'], config)

    curr_epoch = torch.load('epoch.pth') + 1
    optimizer.load_state_dict(torch.load('optimizer.pth'))
    lr_scheduler.load_state_dict(torch.load('lr_scheduler.pth'))

    for i in range(curr_epoch, num_epochs):
        lr_scheduler.step()
        lrs.append(lr_scheduler.get_lr()[0])
    print(lrs)
    print(len(lrs))
    lrs.pop(-1)
    print(len(lrs))
    print(num_epochs-1)
    plt.plot(np.arange(num_epochs-1), lrs)
    plt.show()

    print(lrs[-1])


if __name__ == '__main__':
    main()
