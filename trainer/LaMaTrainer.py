import errno
import math
import os
import wandb
from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import functional
from typing_extensions import TypedDict

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_psnr(predicted: torch.Tensor, ground_truth: torch.Tensor, threshold=0.5):
    pred_img = predicted.detach().cpu().numpy()
    gt_img = ground_truth.detach().cpu().numpy()

    pred_img = (pred_img > threshold) * 1.0

    mse = np.mean((pred_img - gt_img) ** 2)
    psnr = 100 if mse == 0 else (20 * math.log10(1.0 / math.sqrt(mse)))
    return psnr


class LaMaTrainingModule:

    def __init__(self, config, device=None, make_loaders=True):

        self.config = config
        self.device = device
        self.checkpoint = None

        if 'resume' in self.config:
            self.checkpoint = torch.load(config['resume'], map_location=device)
            checkpoint_config = self.checkpoint['config'] if 'config' in self.checkpoint else {}
            if 'train_data_path' in checkpoint_config: del checkpoint_config['train_data_path']
            if 'valid_data_path' in checkpoint_config: del checkpoint_config['valid_data_path']
            if 'test_data_path' in checkpoint_config: del checkpoint_config['test_data_path']
            self.config.update(checkpoint_config)
            config = self.config

        self.training_only_with_patch_square = False
        if make_loaders:
            if len(config['train_data_path']) == 1 and 'patch_square' in config['train_data_path'][0]:
                self.training_only_with_patch_square = True

            self.train_dataset = make_train_dataset(config, self.training_only_with_patch_square)
            self.valid_dataset = make_val_dataset(config, self.training_only_with_patch_square)
            self.test_dataset = make_test_dataset(config)
            self.aux_datasets = []
            self.aux_loaders = []
            for aux_dataset in config['aux_data_path']:
                config_copy = copy.deepcopy(config)
                config_copy['test_data_path'] = [aux_dataset]
                dataset = make_test_dataset(config_copy)
                self.aux_datasets.append(dataset)
                self.aux_loaders.append(make_test_dataloader(dataset, config_copy))

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
                          unet_layers=config['unet_layers'], )

        config['num_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # Training
        self.epoch = 0
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']
        self.seed = config['seed']

        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['model'], strict=True)
            self.epoch = self.checkpoint['epoch']
            self.best_psnr = self.checkpoint['best_psnr']
            self.learning_rate = self.checkpoint['learning_rate']

        self.model = self.model.to(self.device)
        self.ema_rate = config['ema_rate']
        if self.ema_rate is not None:
            self.ema_parameters = copy.deepcopy(list(self.model.parameters()))
            self.model_state_dict = {}
            self.ema_psnr_list = []
            self.ema_best_psnr = 0.
            self.ema_best_psnr_running_mean = 0.
            self.ema_best_psnr_test = 0.

        self.optimizer = make_optimizer(self.model, self.learning_rate, config['kind_optimizer'], config['optimizer'])
        self.criterion = make_criterion(losses=config['losses'])
        self.lr_scheduler = make_lr_scheduler(config['lr_scheduler'], self.optimizer, config['lr_scheduler_kwargs'],
                                              config['lr_scheduler_warmup'], config)

        # Validation
        self.best_epoch = 0
        self.psnr_list = []
        self.best_psnr = 0.
        self.best_psnr_running_mean = 0.
        self.best_psnr_test = 0.
        self.best_precision = 0.
        self.best_recall = 0.

        # Logging
        self.logger = get_logger(LaMaTrainingModule.__name__)

        # Resume
        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            if 'lr_scheduler' in self.checkpoint:
                if self.checkpoint['lr_scheduler'] is not None:
                    self.lr_scheduler = make_lr_scheduler(config['lr_scheduler'], self.optimizer, config['lr_scheduler_kwargs'],
                                                          config['lr_scheduler_warmup'], config)
                    self.lr_scheduler.load_state_dict(self.checkpoint['lr_scheduler'])
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

        if 'ema_rate' in checkpoint:
            self.ema_rate = checkpoint['ema_rate']
            self.ema_parameters = model_state_dict_to_params(checkpoint['ema_parameters'], self.model)

        if 'random_settings' in checkpoint:
            set_seed(checkpoint['random_settings']['seed'])
            random.setstate(checkpoint['random_settings']['random_rng_state'])
            np.random.set_state(checkpoint['random_settings']['numpy_rng_state'])
            torch.set_rng_state(checkpoint['random_settings']['torch_rng_state'])
            torch.cuda.set_rng_state(checkpoint['random_settings']['cuda_rng_state'])

        self.logger.info(f"Loaded pretrained checkpoint model from \"{checkpoints_path}\"")

    def update_ema(self):
        if self.ema_rate:
            for target, source in zip(self.ema_parameters, self.model.parameters()):
                target.detach().mul_(self.ema_rate).add_(source.detach(), alpha=1 - self.ema_rate)

    def _save_checkpoint(self, model_state_dict, root_folder, filename: str):
        random_settings = {'random_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state(),
                           'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state(),
                           'seed': self.seed}

        checkpoint = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch + 1,
            'best_psnr': self.best_psnr,
            'learning_rate': self.learning_rate,
            'config': self.config,
            'random_settings': random_settings,
            'lr_scheduler': self.lr_scheduler.state_dict()
        }

        if wandb.run is not None:
            checkpoint['wandb_id'] = wandb.run.id

        dir_path = root_folder / f"{filename}.pth"
        torch.save(checkpoint, dir_path)
        return dir_path

    def save_checkpoints(self, filename: str):
        root_folder = Path(self.config['path_checkpoint'])
        root_folder.mkdir(parents=True, exist_ok=True)
        dir_path = self._save_checkpoint(self.model.state_dict(), root_folder, filename)
        self.logger.info(f"Stored checkpoints {dir_path}")

        if self.ema_rate:
            dir_path = self._save_checkpoint(
                model_state_dict=params_to_model_state_dict(self.ema_parameters, self.model),
                root_folder=root_folder,
                filename=f"{filename}_{self.ema_rate}_ema")
            self.logger.info(f"Stored checkpoints {dir_path}")

    def set_train_transforms(self, transforms):
        self.train_data_loader.dataset.set_transforms(transforms)

    def load_ema(self):
        if self.ema_rate:
            self.model_state_dict = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(params_to_model_state_dict(self.ema_parameters, self.model))

    def load_model(self):
        if self.model_state_dict:
            self.model.load_state_dict(self.model_state_dict)
        else:
            raise Exception("This function has to be called after load_ema")

    def eval_item(self, item, validator, threshold):
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

        pred = []
        for chunk in torch.split(test, self.config['train_batch_size']):
            pred.append(self.model(chunk))
        pred = torch.cat(pred)

        pred = reconstruct_ground_truth(pred, gt_test, num_rows=num_rows, config=self.config)

        loss = self.criterion(pred, gt_test)

        pred = torch.where(pred > threshold, 1., 0.)
        validator.compute(pred, gt_test)

        test = sample.squeeze(0).detach()
        pred = pred.squeeze(0).detach()
        gt_test = gt_test.squeeze(0).detach()
        test_img = functional.to_pil_image(test)
        pred_img = functional.to_pil_image(pred)
        gt_test_img = functional.to_pil_image(gt_test)
        images = {image_name: [test_img, pred_img, gt_test_img]}

        return loss.item(), validator, images

    @torch.no_grad()
    def test(self):
        self.model.eval()
        test_loss = 0.0
        threshold = self.config['threshold']

        images = {}
        validator = Validator(apply_threshold=self.config['apply_threshold_to_test'], threshold=threshold)

        for i, item in enumerate(self.test_data_loader):
            test_loss_item, validator, images_item = self.eval_item(item, validator, threshold)
            test_loss += test_loss_item
            images.update(images_item)

        avg_loss = test_loss / len(self.test_data_loader)
        avg_metrics = validator.get_metrics()

        self.model.train()
        return avg_metrics, avg_loss, images

    @torch.no_grad()
    def folder_test(self):
        self.model.eval()
        threshold = self.config['threshold']

        validator = Validator(apply_threshold=self.config['apply_threshold_to_test'], threshold=threshold)

        for i, item in enumerate(self.test_data_loader):
            _, _, images_item = self.eval_item(item, validator, threshold)
            yield images_item

    @torch.no_grad()
    def validation(self):
        self.model.eval()
        if self.training_only_with_patch_square:
            return self.validation_patch_square()
        valid_loss = 0.0
        threshold = self.config['threshold']

        images = {}
        validator = Validator(apply_threshold=self.config['apply_threshold_to_valid'], threshold=threshold)

        for item in self.valid_data_loader:
            valid_loss_item, validator, images_item = self.eval_item(item, validator, threshold)
            valid_loss += valid_loss_item
            images.update(images_item)

        avg_loss = valid_loss / len(self.valid_data_loader)
        avg_metrics = validator.get_metrics()

        self.model.train()
        return avg_metrics, avg_loss, images


    @torch.no_grad()
    def aux_test(self):
        old_loader = self.test_data_loader
        results = {}
        for loader in self.aux_loaders:
            self.test_data_loader = loader
            results[Path(loader.dataset.datasets[0].data_path).name] = self.test()
        self.test_data_loader = old_loader
        return results

    @torch.no_grad()
    def validation_patch_square(self):
        self.model.eval()

        valid_loss = 0.0
        threshold = self.config['threshold']

        validator = Validator(apply_threshold=self.config['apply_threshold_to_valid'], threshold=threshold)

        for images, mask_images in self.valid_data_loader:
            inputs, outputs = images.to(self.device), mask_images.to(self.device)
            predictions = self.model(inputs)
            loss = self.criterion(predictions, outputs)
            predictions = torch.where(predictions > threshold, 1., 0.)
            validator.compute(predictions, outputs)

            valid_loss += loss.item()

        avg_loss = valid_loss / len(self.valid_data_loader)
        avg_metrics = validator.get_metrics()
        self.model.train()
        return avg_metrics, avg_loss, None
