import argparse
import random
import sys
import time
import uuid
import traceback
from pathlib import Path
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torchvision.transforms import functional

from trainer.LaMaTrainer import LaMaTrainingModule
from trainer.Validator import Validator
from utils.WandbLog import WandbLog
from utils.htr_logging import get_logger, DEBUG
from utils.ioutils import store_images
from train import set_seed

logger = get_logger('main')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert torch.cuda.is_available(), 'CUDA is not available. Please use a GPU to run this code.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_folder', type=str, required=True)
    parser.add_argument('--attention_num_heads', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--attention_channel_scale_factor', type=int, default=1)
    parser.add_argument('--seed', type=int, default=742)
    parser.add_argument('--train_data_path', type=str, nargs='+', required=True)
    parser.add_argument('--test_data_path', type=str, nargs='+', required=True)
    parser.add_argument('-c', '--configuration', metavar='<name>', type=str,
                        help=f"The configuration name will use on WandB", default="debug_patch_square")

    args = parser.parse_args()
    config_filename = args.configuration
    logger.info("Start process ...")
    configuration_path = f"configs/training/{config_filename}.yaml"
    logger.info(f"Selected \"{configuration_path}\" configuration file")

    with open(configuration_path) as file:
        train_config = yaml.load(file, Loader=yaml.Loader)

    train_config['train_data_path'] = args.train_data_path
    train_config['valid_data_path'] = args.train_data_path
    train_config['test_data_path'] = args.test_data_path

    checkpoints = list(Path(args.checkpoints_folder).glob('*.pth'))
    for checkpoint in checkpoints:
        logger.info(f"Processing {checkpoint}")
        exp_name = checkpoint.stem.split('_')
        # Parse the exp_name
        args.operation = exp_name[0].lower()
        args.n_blocks = int(exp_name[1][:-2])
        args.train_patch_size = int(exp_name[2][:-2])

        if any('global' in s for s in exp_name) or any('local' in s for s in exp_name):
            args.attention = (f'{exp_name[3]}_{exp_name[4]}')[:-3]
        else:
            args.attention = exp_name[3][:-3]
        args.use_skip_connections = exp_name[4] == 'SKIP'
        train_config['use_convolutions'] = args.operation == 'conv'
        train_config['use_skip_connections'] = args.use_skip_connections
        train_config['n_blocks'] = args.n_blocks
        train_config['cross_attention'] = args.attention

        if args.attention_num_heads and args.attention_channel_scale_factor:
            train_config['cross_attention_args'] = {
                'num_heads': args.attention_num_heads,
                'attention_channel_scale_factor': args.attention_channel_scale_factor}
        else:
            train_config['cross_attention_args'] = None
        train_config['train_data_path'] = args.train_data_path
        train_config['valid_data_path'] = args.train_data_path
        train_config['test_data_path'] = args.test_data_path

        train_config['train_kwargs']['num_workers'] = args.num_workers
        train_config['valid_kwargs']['num_workers'] = args.num_workers
        train_config['test_kwargs']['num_workers'] = args.num_workers
        train_config['train_kwargs']['batch_size'] = args.batch_size
        train_config['valid_kwargs']['batch_size'] = args.batch_size
        train_config['test_kwargs']['batch_size'] = 1

        train_config['train_batch_size'] = train_config['train_kwargs']['batch_size']
        train_config['valid_batch_size'] = train_config['valid_kwargs']['batch_size']
        train_config['test_batch_size'] = train_config['test_kwargs']['batch_size']

        train_config['num_epochs'] = 100

        set_seed(args.seed)

        try:
            trainer = LaMaTrainingModule(train_config, device=device)
            if torch.cuda.is_available():
                trainer.model.cuda()
            trainer.load_checkpoints(args.checkpoints_folder, checkpoint.stem[:-10])
            validator = Validator()
            trainer.model.eval()
            test_psnr, test_precision, test_recall, test_loss, images = trainer.test()
            logger.info(f"PSNR: {test_psnr}")
        except Exception as e:
            logger.error(f"Error processing {checkpoint}")
            logger.error(e)
            traceback.print_exc()
            continue

    sys.exit()