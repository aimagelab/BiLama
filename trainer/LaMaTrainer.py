import errno
import math
import os
import wandb
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import functional
from typing_extensions import TypedDict

from data.dataloaders import make_train_dataloader, make_valid_dataloader, make_test_dataloader
from data.datasets import make_train_dataset, make_val_dataset, make_test_dataset
from data.utils import reconstruct_ground_truth
from modules.FFC import LaMa
from trainer.Losses import make_criterion
from trainer.Optimizers import make_optimizer
from trainer.Validator import Validator
from utils.htr_logging import get_logger


def calculate_psnr(predicted: torch.Tensor, ground_truth: torch.Tensor, threshold=0.5):
    pred_img = predicted.detach().cpu().numpy()
    gt_img = ground_truth.detach().cpu().numpy()

    pred_img = (pred_img > threshold) * 1.0

    mse = np.mean((pred_img - gt_img) ** 2)
    psnr = 100 if mse == 0 else (20 * math.log10(1.0 / math.sqrt(mse)))
    return psnr


class LaMaTrainingModule:

    def __init__(self, config, device=None):

        self.config = config
        self.device = device
        self.checkpoint = None

        if 'resume' in self.config:
            self.checkpoint = torch.load(config['resume'])
            self.config = self.checkpoint['config'] if 'config' in self.checkpoint else self.config

        self.training_only_with_patch_square = False
        if len(config['train_data_path']) == 1 and 'patch_square' in config['train_data_path'][0]:
            self.training_only_with_patch_square = True
        self.train_dataset = make_train_dataset(config, self.training_only_with_patch_square)
        self.valid_dataset = make_val_dataset(config, self.training_only_with_patch_square)
        self.test_dataset = make_test_dataset(config)
        self.train_data_loader = make_train_dataloader(self.train_dataset, config)
        self.valid_data_loader = make_valid_dataloader(self.valid_dataset, config)
        self.test_data_loader = make_test_dataloader(self.test_dataset, config)

        self.model = LaMa(input_nc=config['input_channels'], output_nc=config['output_channels'],
                          n_downsampling=config['n_downsampling'], init_conv_kwargs=config['init_conv_kwargs'],
                          downsample_conv_kwargs=config['down_sample_conv_kwargs'],
                          resnet_conv_kwargs=config['resnet_conv_kwargs'], n_blocks=config['n_blocks'],
                          use_convolutions=config['use_convolutions'],
                          cross_attention=config['cross_attention'],
                          cross_attention_args=config['cross_attention_args'],
                          skip_connections=config['skip_connections'],
                          unet_layers=config['unet_layers'],)

        config['num_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # Training
        self.epoch = 0
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']

        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['model'], strict=True)
            self.epoch = self.checkpoint['epoch']
            self.best_psnr = self.checkpoint['best_psnr']
            self.learning_rate = self.checkpoint['learning_rate']

        self.model = self.model.to(self.device)
        self.optimizer = make_optimizer(self.model, self.learning_rate, config['kind_optimizer'], config['optimizer'])
        self.criterion = make_criterion(kind=config['kind_loss'])

        # Validation
        self.best_epoch = 0
        self.best_psnr = 0.
        self.best_precision = 0.
        self.best_recall = 0.

        # Logging
        self.logger = get_logger(LaMaTrainingModule.__name__)

        # Resume
        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.logger.info(f"Loaded pretrained checkpoint model from \"{config['resume']}\"")

        # self.criterion = self.criterion.to(self.device)

    def load_checkpoints(self, folder: str, filename: str):
        checkpoints_path = f"{folder}{filename}_best_psnr.pth"

        if not os.path.exists(path=checkpoints_path):
            self.logger.warning(f"Checkpoints {checkpoints_path} not found.")
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoints_path)

        checkpoint = torch.load(checkpoints_path, map_location=None)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.best_psnr = checkpoint['best_psnr']
        self.learning_rate = checkpoint['learning_rate']
        self.logger.info(f"Loaded pretrained checkpoint model from \"{checkpoints_path}\"")

    def save_checkpoints(self, filename: str):
        root_folder = Path(self.config['path_checkpoint'])
        root_folder.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'best_psnr': self.best_psnr,
            'learning_rate': self.learning_rate,
            'config': self.config
        }

        if wandb.run is not None:
            checkpoint['wandb_id'] = wandb.run.id

        dir_path = root_folder / f"{filename}.pth"
        torch.save(checkpoint, dir_path)
        self.logger.info(f"Stored checkpoints {dir_path}")

    def set_train_transforms(self, transforms):
        self.train_data_loader.dataset.set_transforms(transforms)

    @torch.no_grad()
    def test(self):
        test_loss = 0.0
        threshold = self.config['threshold']

        images = {}
        validator = Validator(apply_threshold=self.config['apply_threshold_to_test'], threshold=threshold)

        for item in self.test_data_loader:
            image_name = item['image_name'][0]
            sample = item['sample']
            num_rows = item['num_rows'].item()
            samples_patches = item['samples_patches']
            gt_sample = item['gt_sample']

            samples_patches = samples_patches.squeeze(0)
            test = samples_patches.to(self.device)
            gt_test = gt_sample.to(self.device)

            test = test.squeeze(0)
            test = test.permute(1, 0, 2, 3)
            pred = self.model(test)

            pred = reconstruct_ground_truth(pred, gt_test, num_rows=num_rows, config=self.config)

            loss = self.criterion(pred, gt_test)
            test_loss += loss.item()

            pred = torch.where(pred > threshold, 1., 0.)
            validator.compute(pred, gt_test)

            test = sample.squeeze(0).detach()
            pred = pred.squeeze(0).detach()
            gt_test = gt_test.squeeze(0).detach()
            test_img = functional.to_pil_image(test)
            pred_img = functional.to_pil_image(pred)
            gt_test_img = functional.to_pil_image(gt_test)
            images[image_name] = [test_img, pred_img, gt_test_img]

        avg_loss = test_loss / len(self.test_data_loader)
        avg_metrics = validator.get_metrics()

        return avg_metrics, avg_loss, images

    @torch.no_grad()
    def validation(self):
        valid_loss = 0.0
        threshold = self.config['threshold']

        images = {}
        validator = Validator(apply_threshold=self.config['apply_threshold_to_valid'], threshold=threshold)

        for item in self.valid_data_loader:
            image_name = item['image_name'][0]
            sample = item['sample']
            num_rows = item['num_rows'].item()
            samples_patches = item['samples_patches']
            gt_sample = item['gt_sample']

            samples_patches = samples_patches.squeeze(0)
            valid = samples_patches.to(self.device)
            gt_valid = gt_sample.to(self.device)

            valid = valid.squeeze(0)
            valid = valid.permute(1, 0, 2, 3)
            pred = self.model(valid)

            pred = reconstruct_ground_truth(pred, gt_valid, num_rows=num_rows, config=self.config)

            loss = self.criterion(pred, gt_valid)
            valid_loss += loss.item()

            pred = torch.where(pred > threshold, 1., 0.)
            validator.compute(pred, gt_valid)

            valid = sample.squeeze(0).detach()
            pred = pred.squeeze(0).detach()
            gt_valid = gt_valid.squeeze(0).detach()
            valid_img = functional.to_pil_image(valid)
            pred_img = functional.to_pil_image(pred)
            gt_test_img = functional.to_pil_image(gt_valid)
            images[image_name] = [valid_img, pred_img, gt_test_img]

        avg_loss = valid_loss / len(self.valid_data_loader)
        avg_metrics = validator.get_metrics()

        return avg_metrics, avg_loss, images

    @torch.no_grad()
    def validation_patch_square(self):
        valid_loss = 0.0
        validator = Validator(apply_threshold=self.config['apply_threshold_to_validation'], threshold=threshold)

        for batch_idx, (valid_in, valid_out) in enumerate(self.valid_data_loader):
            inputs, outputs = valid_in.to(self.device), valid_out.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            loss = self.criterion(predictions, outputs)
            validator.compute(predictions, outputs)

            valid_loss += loss.item()

        avg_loss = valid_loss / len(self.valid_data_loader)
        avg_metrics = validator.get_metrics()
        return avg_metrics, avg_loss,
