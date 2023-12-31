import os
import random
import shutil
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
from colorama import Fore, Style
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import gaussian_kde
from tqdm import tqdm

sns.set(style='dark')

# Paths and Dataframe
dataset_dir = '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset'
df = pd.read_csv(f'{dataset_dir}/csv/dicom_info.csv')
df['image_path'] = df['image_path'].apply(lambda x: x.replace('CBIS-DDSM', dataset_dir))


def show_img(path):
    img = pydicom.dcmread(path).pixel_array
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='bone')
    plt.show()


show_img(df['image_path'].iloc[10])

# Image Sizes
data = df['image_path'].map(lambda path: pydicom.dcmread(path).pixel_array.shape)
df['height'], df['width'] = zip(*data)

# KDE Plots
plt.figure(figsize=(12, 8))
sns.kdeplot(df['width'], shade=True, color='limegreen')
sns.kdeplot(df['height'], shade=True, color='gold')
plt.legend(['width', 'height'])
plt.show()

# Scatter plot with densities
x_val, y_val = df['width'].values, df['height'].values
xy = np.vstack([x_val, y_val])
z = gaussian_kde(xy)(xy)

plt.figure(figsize=(10, 10))
plt.scatter(x_val, y_val, c=z, s=100, cmap='viridis')
plt.show()

# Display info
for info in zip(df.iloc[0].index, df.iloc[0]):
    print(f'{Fore.GREEN}{info[0]}{Style.RESET_ALL}:', info[1])

# Bar plot
plt.figure(figsize=(12, 8))
sns.barplot(df['SeriesDescription'].value_counts(dropna=False).index,
            df['SeriesDescription'].value_counts(dropna=False), palette='viridis')
plt.show()


def show_grid(files, row=3, col=3):
    grid_files = random.sample(files, row * col)
    images = [cv2.resize(pydicom.dcmread(image_path).pixel_array, dsize=(512, 512)) for image_path in tqdm(grid_files)]

    fig = plt.figure(figsize=(col * 5, row * 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(col, row), axes_pad=0.05)

    for ax, im in zip(grid, images):
        ax.imshow(im, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


show_grid(df[df['SeriesDescription'] == 'cropped images']['image_path'].tolist(), row=4)
show_grid(df[df['SeriesDescription'] == 'full mammogram images']['image_path'].tolist(), row=4)
