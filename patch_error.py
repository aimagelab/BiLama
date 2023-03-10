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
    parser.add_argument('--patch_size', type=int, default=256, help='patch size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fourbi = LaMaTrainingModule(config={'resume': args.model}, device=device, make_loaders=False)

    t = transforms.Compose([
            CustomTransform.RandomRotation((-10, 10)),
            CustomTransform.RandomCrop(args.patch_size),
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.ToTensor(),
        ])

    dataset = ConcatDataset([
        ErrorValidationDataset(dataset, patch_size=args.src_patch_size, transform=t, discard_padding=True)
        for dataset in args.datasets
    ])

    dst = Path(f'error_maps_{args.patch_size}')
    dst.mkdir(parents=True, exist_ok=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    error_map = torch.zeros((args.patch_size, args.patch_size), dtype=torch.float32, device=device)
    counter = 0
    fourbi.model.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            for batch_idx, (sample, gt) in enumerate(loader):
                sample, gt = sample.to(device), gt.to(device)
                pred = fourbi.model(sample)
                error_map += torch.abs(pred - gt).sum(dim=(0, 1))
                counter += sample.shape[0]
                print(f'(epoch {epoch + 1}/{args.epochs}) (batch {batch_idx + 1}/{len(loader)})')


            data = {
                'error_map': error_map,
                'patch_size': args.patch_size,
                'count': len(dataset),
                'epochs': args.epochs,
                'real_counter': counter,
            }
            torch.save(data, dst / f'{epoch:03d}.pt')
            img = (error_map / error_map.max()).cpu().numpy() * 255
            transforms.ToPILImage()(img).convert('L').save(dst / f'{epoch:03d}.png')
    print('Done!')

