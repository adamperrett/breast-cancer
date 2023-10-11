import numpy
import torch

import os
import random
import glob
import re

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

sns.set(style='dark')

# parent_directory = 'data/procas_processed_sample'
parent_directory = 'data/procas_raw_sample'
image_paths = glob.glob(os.path.join(parent_directory, '*/*PROCAS*.dcm'))
df = pd.DataFrame({'image_path': image_paths})

def extract_identifier(filename):
    match = re.search(r'-([\d]+)-[A-Z]+', filename)
    return match.group(1) if match else None


df['identifier'] = df['image_path'].apply(lambda x: extract_identifier(os.path.basename(x)))

data = []
for path in df['image_path']:
    dicom_data = pydicom.dcmread(path, force=True)
    if hasattr(dicom_data, 'pixel_array'):
        data.append(dicom_data.pixel_array.shape)
    else:
        print(f"Skipped {path} due to missing pixel data")
        df = df[df.image_path != path]

df['height'], df['width'] = zip(*data)

views = ["LCC", "LMLO", "RMLO", "RCC"]

# Sample 4 random identifiers and display all views of those subjects
chosen_identifiers = random.sample(list(df['identifier'].unique()), 4)
dfs = [[df[(df['identifier'] == identifier) & (df['image_path'].str.contains(view))] for view in views] for identifier in chosen_identifiers]

fig, axes = plt.subplots(4, len(views), figsize=(15, 15))
for row, identifier_dfs in enumerate(dfs):
    for idx, view in enumerate(views):
        if not identifier_dfs[idx].empty:
            path = identifier_dfs[idx]['image_path'].iloc[0]
            dicom_data = pydicom.dcmread(path, force=True)
            if hasattr(dicom_data, 'pixel_array'):
                axes[row, idx].imshow(dicom_data.pixel_array, cmap='gray')
        axes[row, idx].set_xticks([])
        axes[row, idx].set_yticks([])
        if row == 0:
            axes[row, idx].set_title(f'{view}')
        if idx == 0:
            axes[row, idx].set_ylabel(f'ID: {chosen_identifiers[row]}', rotation=0, labelpad=50, verticalalignment='center')

plt.tight_layout()
plt.show()



def pre_process_mammograms(mammographic_image, side, height, width, pixel_size, image_type):
    """
    Preprocesses images to match those on which the models were trained.

    Parameters
    ----------
    mammographic_image: numpy array
        image extracted from the dicom file
    side: string
        breast side (e.g. 'L' or 'R')
    height: int
        height of the mammographic image
    width: int
        width of the mammographic image
    pixel_size: float
        pixel size of the dicom image
    image_type: string
        type of the mammographic image (i.e. raw or procesed)

    Returns
    -------
    mammographic_image: numpy array NXM
        pre-processed mamographic image
    """

    # Reshape image array to the 2D shape 
    mammographic_image = np.reshape(mammographic_image, (height, width))

    # DO NOT CHANGE THIS!!!! 
    target_pixel_size = 0.0941  # All models have been trained on images with this pixel size

    new_height = int(np.ceil(mammographic_image.shape[0] * pixel_size[0] / target_pixel_size))
    new_width = int(np.ceil(mammographic_image.shape[1] * pixel_size[1] / target_pixel_size))

    max_intensity = np.amax(mammographic_image)
    mammographic_image = resize(mammographic_image, (new_height, new_width))

    # Rescale intensity values to their original range
    mammographic_image = mammographic_image * max_intensity / np.amax(mammographic_image)

    # FLIP IMAGE!
    if side == 'R':
        mammographic_image = np.fliplr(mammographic_image)

    if image_type == 'raw':
        # Apply log transform and inverse pixel intensities
        mammographic_image = np.log(mammographic_image)
        mammographic_image = np.amax(mammographic_image) - mammographic_image

    # Pad images
    padded_image = np.zeros(
        (np.amax([2995, mammographic_image.shape[0]]), np.amax([2394, mammographic_image.shape[1]])))
    padded_image[0:mammographic_image.shape[0], 0:mammographic_image.shape[1]] = mammographic_image[:, :]
    mammographic_image = padded_image[0:2995, 0:2394]

    # Resize to 640X512
    mammographic_image = resize(mammographic_image, (10 * 64, 8 * 64))

    # Rescale intensities to [0, 1]
    mammographic_image = mammographic_image / np.amax(mammographic_image)

    return mammographic_image