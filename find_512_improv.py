import argparse
import torch
from pathlib import Path
from trainer.LaMaTrainer import LaMaTrainingModule
from data.ValidationDataset import ErrorValidationDataset
from torchvision import transforms
from torch.utils.data import ConcatDataset
import data.CustomTransforms as CustomTransform

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binarize a folder of images')
    parser.add_argument('model', type=str, metavar='PATH', help='path to the model file')
    parser.add_argument('--datasets', type=str, metavar='PATH', nargs='+', help='path to the model file')
    parser.add_argument('--dst', type=str, default='error_img.pt', help='path output img')
    parser.add_argument('--src_patch_size', type=int, default=512, help='patch size')
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fourbi = LaMaTrainingModule(config={'resume': args.model}, device=device, make_loaders=False)

    t = transforms.Compose([
        CustomTransform.ToTensor(),
    ])

    dataset = ConcatDataset([
        ErrorValidationDataset(dataset, patch_size=512, transform=t, discard_padding=False)
        for dataset in args.datasets
    ])

    dst = Path(f'good_samples')
    dst.mkdir(parents=True, exist_ok=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    center_crop = transforms.CenterCrop(256)

    counter = 0
    fourbi.model.eval()
    with torch.no_grad():
        for batch_idx, (sample, gt) in enumerate(loader):
            sample_512, gt_512 = sample.to(device), gt.to(device)
            pred_512 = fourbi.model(sample_512)
            sample_256 = center_crop(sample_512)
            gt_256 = center_crop(gt_512)
            pred_512to256 = center_crop(pred_512)
            pred_256 = fourbi.model(sample_256)

            pred_512to256 = torch.where(pred_512to256 > 0.5, 1.0, 0.0)
            pred_256 = torch.where(pred_256 > 0.5, 1.0, 0.0)

            loss_256 = torch.abs(pred_256 - gt_256).sum((1, 2, 3))
            loss_512 = torch.abs(pred_512to256 - gt_256).sum((1, 2, 3))

            loss_delta = loss_256 - loss_512
            for idx, delta in enumerate(loss_delta):
                to_pil = transforms.ToPILImage()
                if delta > 500:
                    to_pil(sample_512[idx, :].cpu()).save(dst / f'{int(delta):04d}_{int(loss_512[idx])}_{counter:03d}_s512.png')
                    to_pil(gt_512[idx, :].cpu()).save(dst / f'{int(delta):04d}_{int(loss_512[idx])}_{counter:03d}_gt512.png')
                    to_pil(pred_512[idx, :].cpu()).save(dst / f'{int(delta):04d}_{int(loss_512[idx])}_{counter:03d}_p512.png')
                    to_pil(sample_256[idx, :].cpu()).save(dst / f'{int(delta):04d}_{int(loss_512[idx])}_{counter:03d}_s256.png')
                    to_pil(gt_256[idx, :].cpu()).save(dst / f'{int(delta):04d}_{int(loss_512[idx])}_{counter:03d}_gt256.png')
                    to_pil(pred_256[idx, :].cpu()).save(dst / f'{int(delta):04d}_{int(loss_512[idx])}_{counter:03d}_p256.png')
                    to_pil(pred_512to256[idx, :].cpu()).save(dst / f'{int(delta):04d}_{int(loss_512[idx])}_{counter:03d}_p512to256.png')

                    # transforms.ToPILImage()(s512.cpu()).save(dst / f'{int(delta):04d}_{counter:03d}_s512.png')
                    # transforms.ToPILImage()(gt512.cpu()).save(dst / f'{int(delta):04d}_{counter:03d}_gt512.png')
                    # transforms.ToPILImage()(p512.cpu()).save(dst / f'{int(delta):04d}_{counter:03d}_p512.png')
                    # transforms.ToPILImage()(s256.cpu()).save(dst / f'{int(delta):04d}_{counter:03d}_s256.png')
                    # transforms.ToPILImage()(gt256.cpu()).save(dst / f'{int(delta):04d}_{counter:03d}_gt256.png')
                    # transforms.ToPILImage()(p256.cpu()).save(dst / f'{int(delta):04d}_{counter:03d}_p256.png')
                    counter += 1
            print(f'(batch {batch_idx + 1}/{len(loader)}) {loss_delta.max().item():.2f} {loss_delta.min().item():.2f}')
    print('Done!')
