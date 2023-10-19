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
parent_directory = 'data/mosaics_processed'
# parent_directory = 'data/procas_raw_sample'
image_paths = glob.glob(os.path.join(parent_directory, '*/*.dcm'))
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

plt.figure(figsize=(16, 12))
for view in views:
    df_view = df[df['image_path'].str.contains(view)]
    sns.kdeplot(df_view['width'], shade=True, label=f'{view} Width')
    sns.kdeplot(df_view['height'], shade=True, label=f'{view} Height')
plt.legend()
plt.show()

plt.figure(figsize=(16, 12))
colors = ['limegreen', 'gold', 'cyan', 'magenta']
markers = ['o', 's', '^', 'D']

for idx, view in enumerate(views):
    df_view = df[df['image_path'].str.contains(view)]
    x_val, y_val = df_view['width'].values, df_view['height'].values
    plt.scatter(x_val, y_val, c=colors[idx], s=100, label=f'{view}', alpha=0.6, marker=markers[idx])
plt.legend()
plt.show()

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
