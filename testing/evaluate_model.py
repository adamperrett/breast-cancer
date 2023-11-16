import torch
from dadaptation import DAdaptAdam, DAdaptSGD
import os
import re
import pydicom
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import pad
from torchvision import transforms
from tqdm import tqdm
from skimage import filters, exposure
from training.models import *
from copy import deepcopy
from training.process_all_mosaics import train_loader, val_loader, test_loader, mean, std

model_dir = 'C:/Users/adam_/PycharmProjects/breast-cancer/training/models/'
# model_name = 'clahe_rand_two_ResnetTransformer'
# model_name = 'histo_rand_two_ResnetTransformer'
# model_name = 'log_rand_two_ResnetTransformer'
model_name = 'proc_rand_two_ResnetTransformer'
# model_name = 'clahe_weighted_rand_two_ResnetTransformer'

save_dir = 'D:/mosaic_data/results/'+model_name+'/'

loaders = {'training': train_loader,
           'validation': val_loader,
           'testing': test_loader}

def plot_error_vs_vas(true_values, errors, title, save_location=None, min_e=-40, max_e=40):
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, errors, alpha=0.5)
    plt.xlabel('VAS')
    plt.ylabel('Error')
    plt.ylim([min_e, max_e])
    if np.sum([e < min_e or e > max_e for e in errors]):
        print("An error was out of bounds")
        print("Max:", np.max(errors), "Min:", np.min(errors))
    plt.title(title)
    if save_location:
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        plt.savefig(save_location+"{}.png".format(title), bbox_inches='tight',
                    # dpi=200,
                    format='png')
    else:
        plt.show()
    plt.close()

def visualize_errors(all_data, indexes, x, y, save_location=None):
    fig, axs = plt.subplots(y, x, figsize=(16, 9))

    errors = []
    for i, ndx in enumerate(indexes):
        error = all_data['error'][ndx]
        image = all_data['image'][ndx]
        vas = all_data['vas'][ndx]
        set = all_data['set'][ndx]
        title = f'Error={error:.4f} VAS={vas:.4f} Set={set}'
        axs[int(i/x)][i%x].imshow(image, cmap='gray')
        axs[int(i/x)][i%x].set_title(title)
        axs[int(i/x)][i%x].axis('off')
        errors.append(round(error, 1))

    plt.tight_layout()
    if save_location:
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        plt.savefig(save_location+"errors{}.png".format(errors), bbox_inches='tight',
                    # dpi=200,
                    format='png')
    else:
        plt.show()
    plt.close()

# removes the weighting and transforms
train_loader.dataset.dataset = val_loader.dataset.dataset

# n_images = 9
# for load_type in loaders:
#     loaders[load_type].batch_size = n_images

model = ResNetTransformer().cuda()#.load_state_dict(torch.load(model_dir+model_name))
model.load_state_dict(torch.load(model_dir+model_name))
model.eval()

all_metrics = {
    'error': [],
    'image': [],
    'vas': [],
    'set': []}

criterion = nn.MSELoss(reduction='none')

with torch.no_grad():
    all_targets = []
    all_predictions = []
    for load_type in loaders:
        running_loss = 0.
        starting_idx = len(all_targets)
        print("Starting", load_type)
        for inputs, targets, _ in tqdm(loaders[load_type]):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs.unsqueeze(1))
            outputs_original_scale = inverse_standardize_targets(outputs.squeeze(1), mean, std)
            targets_original_scale = inverse_standardize_targets(targets.float(), mean, std)
            loss = outputs_original_scale - targets_original_scale
            for i, t, e in zip(inputs, targets_original_scale, loss):
                all_metrics['error'].append(e.cpu())
                all_metrics['image'].append(i.cpu())
                all_metrics['vas'].append(t.cpu())
                all_metrics['set'].append(load_type)
            all_targets.extend(targets_original_scale.cpu().numpy())
            all_predictions.extend(outputs_original_scale.cpu().numpy())

            running_loss += criterion(outputs_original_scale,
                                      targets_original_scale).mean().item() * inputs.size(0)

        print("Loss for", load_type, "=", running_loss / len(loaders[load_type].dataset))
        r2 = r2_score(all_targets[starting_idx:], all_predictions[starting_idx:])
        print("R2 for", load_type, "=", r2)
    r2 = r2_score(all_targets, all_predictions)
    print("R2 across all examples =", r2)


plot_error_vs_vas(all_metrics['vas'], all_metrics['error'], 'Absolute error vs VAS for '+model_name, save_dir)
sorted_indices = np.argsort(-np.abs(all_metrics['error']))
for key in all_metrics:
    all_metrics[key] = np.array(all_metrics[key])[sorted_indices].tolist()

'''
plots:
-error vs vas
-error vs image
'''

print("Plotting")

# plot_error_vs_vas(all_metrics['vas'], all_metrics['error'], 'Absolute error vs VAS for '+model_name, save_dir)
n_examples = len(all_metrics['error'])
x = 4
y = 3
n_images = x * y
pbar = tqdm(total=n_examples / n_images)
im = 0
while im < n_examples:
    visualize_errors(all_metrics, range(0, n_examples)[im:im+n_images], 4, 3, save_location=save_dir)
    im += n_images
    pbar.update(1)
pbar.close()

print("Done")