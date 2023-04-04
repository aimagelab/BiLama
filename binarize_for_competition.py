import argparse
import random
import sys
import time
import uuid
import traceback
from pathlib import Path
from datetime import timedelta
import csv
import datetime
from torchvision import transforms

import numpy as np
import torch
import wandb
import yaml
from torchvision.transforms import functional
from data.TestDataset import FolderDataset

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
today = datetime.date.today()
date_str = today.strftime('%Y%m%d')


def binarize_for_competition(config_args, config, patch_sizes=[256], strides=[256]):
    load_data = config['load_data']
    trainer = LaMaTrainingModule(config, device=device, make_loaders=False)
    trainer.config['train_batch_size'] = config_args.batch_size
    test_dataset_path = config['test_data_path']
    print(f'Loading {test_dataset_path}')
    tmp_config = config.copy()
    data = {'checkpoint': config['resume'].name, 'test_dataset': Path(test_dataset_path[0]).name}

    for patch_size, stride in zip(patch_sizes, strides):
        # try:
        save_folder = Path(
            f'{args.outputs_path}BiLama_binarization_results_{date_str}') / f'{config_args.experiment_name}_ps{patch_size}_s{stride}'
        print(f'Saving results in {save_folder}')
        save_folder.mkdir(exist_ok=True, parents=True)
        tmp_config['test_stride'] = stride
        tmp_config['test_patch_size'] = patch_size
        tmp_config['test_data_path'] = test_dataset_path
        trainer.config = tmp_config
        if args.eval_mode == 'true':
            src = Path(test_dataset_path[0])
            test_dataset = FolderDataset(src,
                                         patch_size=patch_size,
                                         overlap=True,
                                         transform=transforms.ToTensor(),
                                         load_data=load_data)
        else:
            test_dataset = make_test_dataset(tmp_config)

        test_data_loader = make_test_dataloader(test_dataset, tmp_config)
        trainer.model.eval()
        validator = Validator(apply_threshold=True, threshold=0.5)
        with torch.no_grad():
            for i, item in enumerate(test_data_loader):
                image_name = item['image_name'][0]
                test_loss_item, validator, images_item = trainer.eval_item(item, validator, 0.5)

                images_item[image_name][0].save(Path(save_folder, f"{Path(image_name).stem}_test_img.png"))
                images_item[image_name][1].save(Path(save_folder, f"{Path(image_name).stem}_pred_img.png"))
                images_item[image_name][2].save(Path(save_folder, f"{Path(image_name).stem}_gt_test_img.png"))

        avg_metrics = validator.get_metrics()
        data[f'PS{patch_size}_S{stride}'] = avg_metrics['psnr']
        print(f'Resulting PSNR {patch_size=} {stride=} for the images: {avg_metrics["psnr"]:.4f}\n\n')

        # except Exception as e:
        #     print(f'Error while binarizing for {patch_size=} {stride=}')
        #     traceback.print_exc()
        #     continue

    out_file = f'{args.outputs_path}{date_str}_patch_size_stride_sweep_{config["resume"].name}_{Path(config["test_data_path"][0]).stem}.csv'
    print(f'Writing results to csv file: {out_file}')
    with open(out_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiment_name', metavar='<name>', type=str,
                        help=f"The experiment name which will use on WandB")
    parser.add_argument('-c', '--configuration', metavar='<name>', type=str,
                        help=f"The configuration name will use on WandB", default="base_bevagna")
    parser.add_argument('-w', '--use_wandb', type=bool, default=not DEBUG)
    parser.add_argument('-t', '--train', type=bool, default=True)
    parser.add_argument('--attention', type=str, default='none',
                        choices=['none', 'cross', 'self', 'cross_local', 'cross_global'])
    parser.add_argument('--attention_num_heads', type=int, default=4)
    parser.add_argument('--attention_channel_scale_factor', type=int, default=1)
    parser.add_argument('--n_blocks', type=int, default=9)
    parser.add_argument('--n_downsampling', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--operation', type=str, default='ffc', choices=['ffc', 'conv'])
    parser.add_argument('--skip', type=str, default='none', choices=['none', 'add', 'cat'])
    parser.add_argument('--resume_ids', type=str, nargs='+', required=True)
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
    parser.add_argument('--use_specified_test_dataset', type=str, default='false', choices=['true', 'false'])
    parser.add_argument('--outputs_path', type=str, required=True)
    parser.add_argument('--fast', type=str, default='false', choices=['true', 'false'])
    parser.add_argument('--min_patch_size', type=int, default=128)
    parser.add_argument('--max_patch_size', type=int, default=768)
    parser.add_argument('--eval_mode', type=str, default='false', choices=['true', 'false'])
    parser.add_argument('--finetuning', type=str, default='false', choices=['true', 'false'])

    args = parser.parse_args()

    config_filename = args.configuration

    logger.info("Start process ...")
    configuration_path = f"configs/training/{config_filename}.yaml"
    logger.info(f"Selected \"{configuration_path}\" configuration file")

    with open(configuration_path) as file:
        train_config = yaml.load(file, Loader=yaml.Loader)

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
    train_config['lr_scheduler'] = args.lr_scheduler
    train_config['lr_scheduler_kwargs'] = args.lr_scheduler_kwargs
    train_config['lr_scheduler_warmup'] = args.lr_scheduler_warmup
    train_config['learning_rate'] = args.lr
    train_config['learning_rate_min'] = args.lr_min
    train_config['seed'] = args.seed
    if args.attention == 'self':
        raise NotImplementedError('Self attention is not implemented yet')
    args.train_data_path = [dataset for dataset in args.datasets]

    train_config['train_data_path'] = args.train_data_path
    train_config['valid_data_path'] = args.train_data_path
    train_config['merge_image'] = args.merge_image == 'true'
    train_config['finetuning'] = args.finetuning == 'true'

    if args.attention_num_heads and args.attention_channel_scale_factor:
        train_config['cross_attention_args'] = {
            'num_heads': args.attention_num_heads,
            'attention_channel_scale_factor': args.attention_channel_scale_factor}
    else:
        train_config['cross_attention_args'] = None

    train_config['train_kwargs']['num_workers'] = args.num_workers
    train_config['valid_kwargs']['num_workers'] = args.num_workers
    train_config['test_kwargs']['num_workers'] = args.num_workers
    train_config['train_kwargs']['batch_size'] = args.batch_size
    train_config['valid_kwargs']['batch_size'] = 1
    train_config['test_kwargs']['batch_size'] = 1
    train_config[
        'train_transform_variant'] = args.train_transform_variant if args.train_transform_variant != 'none' else None

    train_config['train_batch_size'] = train_config['train_kwargs']['batch_size']
    train_config['valid_batch_size'] = train_config['valid_kwargs']['batch_size']
    train_config['test_batch_size'] = train_config['test_kwargs']['batch_size']

    train_config['num_epochs'] = args.epochs
    train_config['patience'] = args.patience
    train_config['ema_rate'] = args.ema_rate if args.ema_rate > 0 else None

    train_config['apply_threshold_to_train'] = args.apply_threshold_to
    train_config['apply_threshold_to_valid'] = args.apply_threshold_to
    train_config['apply_threshold_to_test'] = args.apply_threshold_to
    train_config['threshold'] = args.threshold
    train_config['load_data'] = args.load_data == 'true'

    train_config['apply_threshold_to_train'] = True
    train_config['apply_threshold_to_valid'] = True
    train_config['apply_threshold_to_test'] = True
    if args.apply_threshold_to == 'none':
        train_config['apply_threshold_to_train'] = False
        train_config['apply_threshold_to_valid'] = False
        train_config['apply_threshold_to_test'] = False
    elif args.apply_threshold_to == 'val_test':
        train_config['apply_threshold_to_train'] = False
    elif args.apply_threshold_to == 'test':
        train_config['apply_threshold_to_train'] = False
        train_config['apply_threshold_to_valid'] = False

    set_seed(args.seed)
    min_patch_size = args.min_patch_size
    max_patch_size = args.max_patch_size
    offset = 64
    patches_sizes = list(range(min_patch_size, max_patch_size, offset))
    strides = list(patch_size // 2 for patch_size in patches_sizes)
    patches_sizes += list(range(min_patch_size, max_patch_size, offset))
    strides += list(range(min_patch_size, max_patch_size, offset))

    if args.fast == 'true':
        patches_sizes = [256, 256, 512, 512, 768, 768]
        strides = [128, 256, 256, 512, 384, 768]

    if args.use_specified_test_dataset == 'true':
        datasets = {Path(dataset).name: dataset for dataset in args.datasets}
        args.test_data_path = [datasets[args.test_dataset]]
    checkpoint_path = Path(train_config['path_checkpoint'])
    results = []

    for i, resume_id in enumerate(args.resume_ids):
        checkpoints = sorted(checkpoint_path.rglob(f"*_{resume_id}*test*.pth"))
        assert len(checkpoints) > 0, f"Found {len(checkpoints)} checkpoints with uuid {resume_id} in {checkpoint_path}"
        for j, checkpoint in enumerate(checkpoints):
            if not args.use_specified_test_dataset == 'true':
                loaded_checkpoint = torch.load(checkpoint)
                test_dataset_name = Path(loaded_checkpoint['config']['test_data_path'][0]).name
                args.test_data_path = [dataset for dataset in args.datasets if dataset.endswith(test_dataset_name)]
            train_config['test_data_path'] = args.test_data_path
            assert len(
                train_config['test_data_path']) > 0, f"Test dataset {args.test_dataset} not found in {args.datasets}"
            train_config['resume'] = checkpoint
            args.experiment_name = checkpoint.stem
            print(f"---------------------------------------------------------------------\n")
            print(f"[{i}]/[{len(args.resume_ids)}]-[{j}]/[{len(checkpoints)}] -- Running {args.experiment_name} \n")
            results.append(binarize_for_competition(args, train_config, patches_sizes, strides))

    output_file = f'{args.outputs_path}{date_str}_patch_size_stride_sweep_{"_".join(args.resume_ids)}.csv'
    if args.use_specified_test_dataset == 'true':
        output_file = f'{args.outputs_path}{date_str}_patch_size_stride_sweep_{"_".join(args.resume_ids)}_{args.test_dataset}.csv'

    print(f"Saving results to {output_file}")
    resume_ids = [str(resume_id) for resume_id in args.resume_ids]
    with open(output_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"---------------------------------------------------------------------\n")
    print(f"Done! \n")
    sys.exit()
