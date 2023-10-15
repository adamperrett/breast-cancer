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
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

sns.set(style='dark')

print("Reading data")

parent_directory = 'Z:/PROCAS_ALL_PROCESSED'
new_parent_directory = "../data/mosaics_processed"

mosaic_data = pd.read_csv('../data/matched_mosaics.csv', sep=',')
mosaic_ids = mosaic_data['Patient']
vas_density_data = mosaic_data['VASCombinedAvDensity']

mosaic_ids = mosaic_ids.unique()

look_for_sub = ['7e', '10f']
for i in range(8, 11):
    look_for_sub.append('{}a'.format(i))
    look_for_sub.append('{}b'.format(i))
    look_for_sub.append('{}c'.format(i))
    look_for_sub.append('{}d'.format(i))
    look_for_sub.append('{}e'.format(i))

mosaic_directories = []
for id in tqdm(mosaic_ids):
    checked = False
    for check in look_for_sub:
        if check in id:
            checked = True
            mosaic_directories.append(parent_directory + '/' + 'Densitas_{}_anon/{}'.format(check, id))
            break
    if not checked:
        mosaic_directories.append(parent_directory + '/' + id)

import shutil

print("Copying files")
for dir_path in tqdm(mosaic_directories):
    # Get the last part of the directory (i.e., the folder name)
    folder_name = os.path.basename(dir_path)

    # Create the new path where you want to copy the directory
    new_path = os.path.join(new_parent_directory, folder_name)

    # Copy the directory to the new location
    shutil.copytree(dir_path, new_path)

print("Finished copying values")
