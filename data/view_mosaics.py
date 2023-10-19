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

processed_dataset_path = '../data/mosaics_processed/dataset_only_vas_0.pth'
# Load dataset from saved path
dataset = MammogramDataset(processed_dataset_path)

all_data = [[im, vas] for im, vas in dataset]

n = 3

# Sort all_data based on vas score
sorted_data = sorted(all_data, key=lambda x: x[1])

# Get n smallest and largest
n_smallest = sorted_data[:n]
n_largest = sorted_data[-n:]

# Combine n_smallest and n_largest
result_data = n_smallest + n_largest

# Split into images and vas scores if needed
result_images = [entry[0] for entry in result_data]
result_vas_scores = [entry[1] for entry in result_data]

import matplotlib.pyplot as plt

def visualize_tensor(tensor, vas):
    plt.figure()
    plt.imshow(torch.cat(tensor.unbind(0), dim=-1), cmap='gray')  # 'gray' for grayscale. Remove if you want colormap
    plt.title(vas)
    plt.colorbar()
    plt.show()

for im, vas in zip(result_images, result_vas_scores):
    visualize_tensor(im, vas)

print("done")