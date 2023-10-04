import pandas as pd, numpy as np
import os, shutil
from glob import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Back, Style
sns.set(style='dark')

dataset_dir = '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset'

df = pd.read_csv(f'{dataset_dir}/csv/dicom_info.csv')
df['image_path'] = df.image_path.apply(lambda x: x.replace('CBIS-DDSM', dataset_dir))
df.head()

def show_img(path):
    img = cv2.imread(path,0)
    plt.figure(figsize=(10,10))
    plt.imshow(img,cmap='bone')

show_img(df.image_path.iloc[10])

%%time
import imagesize
data = df['image_path'].map(lambda path: imagesize.get(path))
width, height = list(zip(*data))
df['width'] = width
df['height'] = height
# df.head()

plt.figure(figsize=(12,8))
sns.kdeplot(df['width'], shade=True, color='limegreen')
sns.kdeplot(df['height'], shade=True, color='gold')
plt.legend(['width','height'])

from scipy.stats import gaussian_kde


x_val = df.width.values
y_val = df.height.values

# Calculate the point density
xy = np.vstack([x_val,y_val])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots(figsize = (10, 10))
# ax.axis('off')
ax.scatter(x_val, y_val, c=z, s=100, cmap='viridis')
# ax.set_xlabel('x_mid')
# ax.set_ylabel('y_mid')
plt.show()

for info in zip(df.iloc[0].index, df.iloc[0]):
    print(f'{Fore.GREEN}{info[0]}{Style.RESET_ALL}:',info[1])

plt.figure(figsize=(12,8))
sns.barplot(df.SeriesDescription.value_counts(dropna=False).index, df.SeriesDescription.value_counts(dropna=False), palette='viridis')
#.plot.bar(rot=0, color=['deepskyblue', 'royalblue', 'deeppink'])

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import random

def show_grid(files, row=3, col=3):
    grid_files = random.sample(files, row*col)
    images     = []
    for image_path in tqdm(grid_files):
        img          = cv2.resize(cv2.imread(image_path), dsize=(512,512))
        images.append(img)

    fig = plt.figure(figsize=(col*5, row*5))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(col, row),  # creates 2x2 grid of axes
                     axes_pad=0.05,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

show_grid(df[df.SeriesDescription=='cropped images'].image_path.tolist(), row=4)

show_grid(df[df.SeriesDescription=='full mammogram images'].image_path.tolist(), row=4)

