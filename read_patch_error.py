import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import timeit
from pathlib import Path

import torch
from PIL import Image

# mpl.use('TkAgg')

ffc_256 = '/home/vpippi/BiLama/ffc_error_maps_256'
ffc_512 = '/home/vpippi/BiLama/ffc_error_maps_512'
conv_256 = '/home/vpippi/BiLama/conv_error_maps_256'
conv_512 = '/home/vpippi/BiLama/conv_error_maps_512'

ffc_256_last = sorted(Path(ffc_256).rglob('*.pt'))[-1]
ffc_512_last = sorted(Path(ffc_512).rglob('*.pt'))[-1]
conv_256_last = sorted(Path(conv_256).rglob('*.pt'))[-1]
conv_512_last = sorted(Path(conv_512).rglob('*.pt'))[-1]

ffc_256_dict = torch.load(ffc_256_last, map_location=torch.device('cpu'))
ffc_512_dict = torch.load(ffc_512_last, map_location=torch.device('cpu'))
conv_256_dict = torch.load(conv_256_last, map_location=torch.device('cpu'))
conv_512_dict = torch.load(conv_512_last, map_location=torch.device('cpu'))

ffc_256_iter = ffc_256_dict['epochs'] * ffc_256_dict['count']
ffc_512_iter = ffc_512_dict['epochs'] * ffc_512_dict['count']
conv_256_iter = conv_256_dict['epochs'] * conv_256_dict['count']
conv_512_iter = conv_512_dict['epochs'] * conv_512_dict['count']

ffc_256_img = (ffc_256_dict['error_map'] / ffc_256_iter).cpu().numpy()
ffc_512_img = (ffc_512_dict['error_map'] / ffc_512_iter).cpu().numpy()
conv_256_img = (conv_256_dict['error_map'] / conv_256_iter).cpu().numpy()
conv_512_img = (conv_512_dict['error_map'] / conv_512_iter).cpu().numpy()

# Max values
print('Min and Max values')
print(f'ffc_256: {ffc_256_img.min()}, {ffc_256_img.max()}')
print(f'ffc_512: {ffc_512_img.min()}, {ffc_512_img.max()}')
print(f'conv_256: {conv_256_img.min()}, {conv_256_img.max()}')
print(f'conv_512: {conv_512_img.min()}, {conv_512_img.max()}')

def get_hist(img, hist_range):
    patch_size = img.shape[0]
    hist_width = patch_size // 16
    hist, xedges, yedges = np.histogram2d(
        np.arange(patch_size).repeat(patch_size),
        np.tile(np.arange(patch_size), patch_size),
        bins=[hist_width, hist_width],
        weights=img.flatten(),
    )

    return hist

# range_min = min(ffc_256_img.min(), ffc_512_img.min(), conv_256_img.min(), conv_512_img.min())
# range_max = max(ffc_256_img.max(), ffc_512_img.max(), conv_256_img.max(), conv_512_img.max())
range_min = 0.5
range_max = 6
hist_range = ((range_min, range_max), (range_min, range_max))

hist1 = get_hist(ffc_256_img, hist_range=hist_range)
hist2 = get_hist(ffc_512_img, hist_range=hist_range)
hist3 = get_hist(conv_256_img, hist_range=hist_range)
hist4 = get_hist(conv_512_img, hist_range=hist_range)

# plot the first hist
fig, ax = plt.subplots()
im = ax.imshow(hist4, cmap='coolwarm')
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
ax.axis('off')

# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

# Plot the histograms on each subplot
# im1 = axs[0, 0].imshow(hist1, cmap='coolwarm', vmin=range_min, vmax=range_max)
# axs[0, 0].grid(which='minor', color='w', linestyle='-', linewidth=2)
# axs[0, 0].set_title('ffc_256')
# im2 = axs[0, 1].imshow(hist2, cmap='coolwarm', vmin=range_min, vmax=range_max)
# axs[0, 1].set_title('ffc_512')
# im3 = axs[1, 0].imshow(hist3, cmap='coolwarm', vmin=range_min, vmax=range_max)
# axs[1, 0].set_title('conv_256')
# im4 = axs[1, 1].imshow(hist4, cmap='coolwarm', vmin=range_min, vmax=range_max)
# axs[1, 1].set_title('conv_512')

# Add a colorbar to the figure
cbar = plt.colorbar(im, extend='neither')
cbar.set_label('Counts')

# Adjust the layout of the subplots
# fig.tight_layout()

# Show the plot
plt.savefig('conv_512.pdf', bbox_inches='tight', pad_inches=0)

print('Done')