import argparse
import random
import sys
import time
import uuid
import traceback
from pathlib import Path
from datetime import timedelta
import csv

import numpy as np
import torch
import wandb
import yaml
from torchvision.transforms import functional

from trainer.LaMaTrainer import LaMaTrainingModule, set_seed
from data.dataloaders import make_test_dataloader
from data.datasets import make_test_dataset
from trainer.Validator import Validator
from utils.WandbLog import WandbLog
from utils.htr_logging import get_logger, DEBUG
from utils.ioutils import store_images

logger = get_logger('main')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert torch.cuda.is_available(), 'CUDA is not available. Please use a GPU to run this code.'


def test(config):
    path_checkpoints = Path(config['path_checkpoint'])
    results = []

    test_loaders = []
    new_binarization_datasets = ['ISOSBTD', 'PHIBD', 'Nabuco', 'BickleyDiary', 'SMADI']
    for dataset in config['datasets']:
        print(f'Loading {dataset}')
        is_validation = False
        if any(elem in dataset for elem in new_binarization_datasets):
            is_validation = True
        tmp_config = config.copy()
        tmp_config['test_data_path'] = [dataset]
        test_dataset = make_test_dataset(tmp_config, is_validation=is_validation)
        test_data_loader = make_test_dataloader(test_dataset, tmp_config)
        test_loaders.append(test_data_loader)

    path_checkpoints = list(path_checkpoints.glob('*_best_psnr*.pth'))
    for i, path_checkpoint in enumerate(path_checkpoints):
        try:
            config['resume'] = path_checkpoint
            print(f'Processing {path_checkpoint} ({i + 1}/{len(path_checkpoints)})')
            checkpoint = torch.load(path_checkpoint)
            if not 'config' in checkpoint:
                print(f"Checkpoint {path_checkpoint} is not compatible with this version of the code")
                continue
            trainer = LaMaTrainingModule(config, device=device, make_loaders=False)
            data = {'checkpoint': path_checkpoint.name}

            for dataset, loader in zip(config['datasets'], test_loaders):
                print(f'Processing {dataset}')
                tmp_config = config.copy()
                tmp_config['test_data_path'] = [dataset]
                trainer.test_data_loader = loader

                avg_metrics, avg_loss, images = trainer.test()
                data[Path(dataset).name] = avg_metrics['psnr']

            results.append(data)
            print('\t'.join([f'{k}: {v}' for k, v in data.items()]))

            with open('/mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/all_test_results.csv', 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                if i == 0:
                    writer.writeheader()
                writer.writerows(results)
        except Exception as e:
            print(f'Error while processing {path_checkpoint}')
            traceback.print_exc()
            continue

    with open('/mnt/beegfs/work/FoMo_AIISDH/vpippi/BiLama/final_all_test_results.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiment_name', metavar='<name>', type=str,
                        help=f"The experiment name which will use on WandB")
    parser.add_argument('-c', '--configuration', metavar='<name>', type=str,
                        help=f"The configuration name will use on WandB", default="debug_patch_square")
    parser.add_argument('-w', '--use_wandb', type=bool, default=not DEBUG)
    parser.add_argument('-t', '--train', type=bool, default=True)
    parser.add_argument('--attention', type=str, default='none',
                        choices=['none', 'cross', 'self', 'cross_local', 'cross_global'])
    parser.add_argument('--attention_num_heads', type=int, default=4)
    parser.add_argument('--attention_channel_scale_factor', type=int, default=1)
    parser.add_argument('--n_blocks', type=int, default=9)
    parser.add_argument('--n_downsampling', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--operation', type=str, default='ffc', choices=['ffc', 'conv'])
    parser.add_argument('--skip', type=str, default='none', choices=['none', 'add', 'cat'])
    parser.add_argument('--resume', type=str, default='none')
    parser.add_argument('--wandb_dir', type=str, default='/tmp')
    parser.add_argument('--unet_layers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=742)
    parser.add_argument('--patience', type=int, default=60)
    parser.add_argument('--apply_threshold_to', type=str, default='test', choices=['none', 'val_test', 'test', 'all'])
    parser.add_argument('--loss', type=str, nargs='+', default=['binary_cross_entropy'],
                        choices=['mean_square_error', 'cross_entropy', 'negative_log_likelihood',
                                 'custom_mse', 'charbonnier', 'binary_cross_entropy'])
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--lr_min', type=float, default=1.5e-5)
    parser.add_argument('--lr_scheduler', type=str, default='constant',
                        choices=['constant', 'exponential', 'multistep', 'linear', 'cosine', 'plateau', 'step'])
    parser.add_argument('--lr_scheduler_warmup', type=int, default=0)
    parser.add_argument('--lr_scheduler_kwargs', type=eval, default={})
    parser.add_argument('--ema_rate', type=float, default=-1)
    parser.add_argument('--load_data', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--train_transform_variant', type=str, default='none', choices=['threshold_mask', 'none'])
    parser.add_argument('--merge_image', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--test_dataset', type=str, required=True)

    args = parser.parse_args()

    config_filename = args.configuration

    logger.info("Start process ...")
    configuration_path = f"configs/training/{config_filename}.yaml"
    logger.info(f"Selected \"{configuration_path}\" configuration file")

    with open(configuration_path) as file:
        train_config = yaml.load(file, Loader=yaml.Loader)

    if args.resume != 'none':
        checkpoint_path = Path(train_config['path_checkpoint'])
        checkpoints = sorted(checkpoint_path.glob(f"*_{args.resume}*.pth"))
        assert len(checkpoints) > 1, f"Found {len(checkpoints)} checkpoints with uuid {args.resume}"
        train_config['resume'] = checkpoints[0]
        args.experiment_name = checkpoints[0].stem.rstrip('_best_psnr')

    if args.experiment_name is None:
        exp_name = [
            args.operation.upper(),
            str(args.n_blocks) + 'RB',
            args.skip + 'SKIP',
            str(args.unet_layers) + 'UL',
            str(args.n_downsampling) + 'DS',
            '+'.join(l[:4] for l in args.loss) + 'LOSS',
            args.lr_scheduler[:4] + 'SCHE',
            str(uuid.uuid4())[:4]
        ]
        if args.ema_rate > 0: exp_name.insert(-1, f"{args.ema_rate}EMA")
        args.experiment_name = '_'.join(exp_name)

    train_config['experiment_name'] = args.experiment_name
    train_config['use_convolutions'] = args.operation == 'conv'
    train_config['skip_connections'] = args.skip
    train_config['unet_layers'] = args.unet_layers
    train_config['n_blocks'] = args.n_blocks
    train_config['n_downsampling'] = args.n_downsampling
    train_config['cross_attention'] = args.attention
    train_config['losses'] = args.loss