import os
import random
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore, Style
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import gaussian_kde
from tqdm import tqdm

sns.set(style='dark')

# Paths and Dataframe
parent_directory = 'data/PROCAS_large_jpgs'  # Update this path to your parent directory

# Glob for .png files recursively within subfolders containing "PROCAS" in the filename
image_paths = glob.glob(os.path.join(parent_directory, '*/*PROCAS*.png'))

df = pd.DataFrame({'image_path': image_paths})

def show_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='bone')
    plt.show()

show_img(df['image_path'].iloc[10])

# Image Sizes
data = [cv2.imread(path, cv2.IMREAD_GRAYSCALE).shape for path in df['image_path']]
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

# Display first row info
for info in zip(df.iloc[0].index, df.iloc[0]):
    print(f'{Fore.GREEN}{info[0]}{Style.RESET_ALL}:', info[1])

def show_grid(files, row=3, col=3):
    grid_files = random.sample(files, row * col)
    images = [cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), dsize=(512, 512)) for image_path in tqdm(grid_files)]

    fig = plt.figure(figsize=(col * 5, row * 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(col, row), axes_pad=0.05)

    for ax, im in zip(grid, images):
        ax.imshow(im, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# This will show grid images (you can choose a filter if needed)
show_grid(df['image_path'].tolist(), row=4)
