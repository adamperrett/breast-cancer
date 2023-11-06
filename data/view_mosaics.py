import torch
from dadaptation import DAdaptAdam
import os
import re
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import pad
from torchvision import transforms
from tqdm import tqdm
from skimage import filters

class MammogramDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset = torch.load(dataset_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            transformed_image = [self.transform(im.unsqueeze(0)).squeeze(0) for im in image]
            image = transformed_image
        return image, label

# processed_dataset_path = '../data/mosaics_processed/dataset_only_vas_0_bs.pth'
# processed_dataset_path = '../data/mosaics_processed/full_dataset.pth'
prepro_type = 'log'
processed_dataset_path = '../data/mosaics_processed/full_mosaic_dataset_'+prepro_type+'.pth'
save_location = 'D:/mosaic_data/raw/images/'+prepro_type
# Load dataset from saved path
dataset = MammogramDataset(processed_dataset_path)

all_data = [[im, vas] for im, vas in dataset]

n = 6

# Sort all_data based on vas score
sorted_data = sorted(all_data, key=lambda x: x[1])

# Get n smallest and largest
n_smallest = sorted_data[:n]
n_largest = sorted_data[-n:]

# Combine n_smallest and n_largest
result_data = n_smallest + n_largest

# Split into images and vas scores if needed
result_small_images = [entry[0] for entry in result_data]
result_small_vas_scores = [entry[1] for entry in result_data]

import matplotlib.pyplot as plt

def visualize_tensor(tensor_and_vas, save=True):
    # combined_image = torch.cat(tensor.unbind(0), dim=-1)

    # Create a figure with 1 row and 2 columns
    fig, axs = plt.subplots(int(n/2), 2, figsize=(16, 9))

    # Display the combined_image on the first subplot
    # axs[0].imshow(combined_image, cmap='gray')
    # axs[0].set_title(f'Combined Image - {vas}')
    # axs[0].axis('off')

    vas_scores = []
    for i, (t, v) in enumerate(tensor_and_vas):
        vas_scores.append(v)
        t = torch.cat(t.unbind(0), dim=-1)
        axs[int(i/2)][i%2].imshow(t, cmap='gray')
        axs[int(i/2)][i%2].set_title(f'VAS - {v}')
        axs[int(i/2)][i%2].axis('off')

    # Threshold the combined_image using Otsu's method
    # cut_off = combined_image > filters.threshold_otsu(combined_image.numpy())
    # cut_off = cut_off.float()
    # thresholded_image = cut_off * combined_image
    #
    # # Display the thresholded image on the second subplot
    # axs[1].imshow(thresholded_image, cmap='gray')
    # axs[1].set_title(f'Thresholded Image - {vas}')
    # axs[1].axis('off')

    # Add colorbar for the thresholded image (optional)
    # fig.colorbar(im, ax=axs[1])

    plt.tight_layout()
    if save:
        plt.savefig(save_location+"VAS{}.png".format(vas_scores), bbox_inches='tight',
                    # dpi=200,
                    format='png')
        plt.close()
    else:
        plt.show()

# for im, vas in sorted_data:
#     visualize_tensor(im, vas)
# for im, vas in sorted_data:
#     #view in temporal order
#     visualize_tensor(im, vas)

pbar = tqdm(total=len(sorted_data)/n)
im = 0
while im < len(sorted_data):
    visualize_tensor(sorted_data[im:im+n])
    im += n
    pbar.update(1)
pbar.close()

print("done")