import torch
from dadaptation import DAdaptAdam, DAdaptSGD
import os
import random
import re
import math

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


seed_value = 272727
np.random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sns.set(style='dark')

def plot_scatter(true_values, pred_values, title):
    plt.figure(figsize=(10,6))
    plt.scatter(true_values, pred_values, alpha=0.5)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], '--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.show()

def plot_error_distribution(true_values, pred_values, title):
    errors = np.array(true_values) - np.array(pred_values)
    sns.histplot(errors, bins=50, kde=True)
    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs.unsqueeze(1))
            test_outputs_original_scale = inverse_standardize_targets(outputs.squeeze(), mean, std)
            test_targets_original_scale = inverse_standardize_targets(targets.float(), mean, std)
            loss = criterion(test_outputs_original_scale, test_targets_original_scale)
            running_loss += loss.item() * inputs.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    r2 = r2_score(all_targets, all_predictions)
    return epoch_loss, all_targets, all_predictions, r2

def compute_target_statistics(dataset):
    labels = [label for _, label in dataset]
    mean = np.mean(labels)
    std = np.std(labels)
    return mean, std

def standardize_targets(target, mean, std):
    return (target - mean) / std

def inverse_standardize_targets(target, mean, std):
    return target * std + mean

def extract_identifier(filename):
    match = re.search(r'-([\d]+)-[A-Z]+', filename)
    return match.group(1) if match else None

print("Reading data")

parent_directory = '../data/mosaics_processed'

mosaic_data = pd.read_csv('../data/matched_mosaics.csv', sep=',')
mosaic_ids = mosaic_data['Patient']
vas_density_data = mosaic_data['VASCombinedAvDensity']

processed = True
processed_dataset_path = '../data/mosaics_processed/dataset_only_vas_0_bs.pth'

best_model_name = 'first_two_ResnetTransformer'
# best_model_name = 'only_first_Transformer'

# save mosaic_ids which have a vas score
id_vas_dict = {}
for id, vas in zip(mosaic_ids, vas_density_data):
    if not np.isnan(vas) and vas > 0:
        id_vas_dict[id] = vas

mosaic_ids = mosaic_ids.unique()

def pre_process_mammograms(mammographic_images, sides, heights, widths, pixel_sizes, image_types):
    target_pixel_size = 0.0941
    processed_images = []
    print("Beginning processing images")
    for idx, mammographic_image in enumerate(tqdm(mammographic_images)):
        # Extract parameters for each image
        side = sides[idx]
        height = heights[idx]
        width = widths[idx]
        pixel_size = pixel_sizes[idx]
        image_type = image_types[idx]

        # Reshape and preprocess
        mammographic_image = np.reshape(mammographic_image, (height, width))
        new_height = int(np.ceil(height * pixel_size[0] / target_pixel_size))
        new_width = int(np.ceil(width * pixel_size[1] / target_pixel_size))
        max_intensity = np.amax(mammographic_image)
        mammographic_image = resize(mammographic_image, (new_height, new_width))
        mammographic_image = mammographic_image * max_intensity / np.amax(mammographic_image)
        if side == 'R':
            mammographic_image = np.fliplr(mammographic_image)
        if image_type == 'raw':
            mammographic_image = np.log(mammographic_image)
            mammographic_image = np.amax(mammographic_image) - mammographic_image
        padded_image = np.zeros((max(2995, mammographic_image.shape[0]), max(2394, mammographic_image.shape[1])))
        padded_image[:mammographic_image.shape[0], :mammographic_image.shape[1]] = mammographic_image
        mammographic_image = resize(padded_image[:2995, :2394], (10 * 64, 8 * 64))
        mammographic_image = mammographic_image / np.amax(mammographic_image)
        processed_images.append(mammographic_image)
    return torch.stack([torch.from_numpy(img).float() for img in processed_images], dim=0)

# This function will preprocess and zip all images for individual directories and return a list of datasets for each directory
def preprocess_and_zip_per_directory(parent_directory, id_vas_dict):
    dir_datasets = []

    patient_dirs = [d for d in os.listdir(parent_directory) if d in id_vas_dict]
    patient_dirs.sort()  # Ensuring a deterministic order

    for patient_dir in tqdm(patient_dirs):
        patient_path = os.path.join(parent_directory, patient_dir)
        image_files = [f for f in os.listdir(patient_path) if f.endswith('.dcm')]

        all_images = [pydicom.dcmread(os.path.join(patient_path, f), force=True).pixel_array for f in image_files]
        all_sides = ['L' if 'LCC' in f or 'LMLO' in f else 'R' for f in image_files]
        all_heights = [img.shape[0] for img in all_images]
        all_widths = [img.shape[1] for img in all_images]
        all_pixel_sizes = [(0.0941, 0.0941) for _ in all_images]
        all_image_types = ['raw' if ('raw' in patient_path or 'RAW' in patient_path)
                           else 'processed' for _ in image_files]

        preprocessed_images = pre_process_mammograms(all_images, all_sides, all_heights, all_widths, all_pixel_sizes, all_image_types)

        dir_datasets.append([preprocessed_images, id_vas_dict[patient_dir]])

    return dir_datasets

def custom_collate(batch):
    # Separate images and labels
    images, labels = zip(*batch)

    # Stack the images and labels into tensors
    images_tensor = torch.stack(images)

    # Standardize the labels and stack them into a tensor
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    labels_tensor = standardize_targets(labels_tensor, mean, std)

    return images_tensor, labels_tensor

class MammogramDataset(Dataset):
    def __init__(self, dir_datasets, transform=None):
        # Flatten the list of datasets to have a single list of samples
        self.transform = transform
        self.dataset = []
        for inputs, target in dir_datasets:
                self.dataset.append([torch.cat(tuple(inputs[:2]), dim=-1), target])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image.unsqueeze(0)).squeeze(0)
        return image, label

if not processed:
    if not os.path.exists(processed_dataset_path):
        dir_datasets = preprocess_and_zip_per_directory(parent_directory, id_vas_dict)
        torch.save(dir_datasets, processed_dataset_path)
else:
    dir_datasets = torch.load(processed_dataset_path)

# We'll shuffle the directories to create the split
random.shuffle(dir_datasets)
train_ratio, val_ratio = 0.7, 0.2
num_train_dirs = int(train_ratio * len(dir_datasets))
num_val_dirs = int(val_ratio * len(dir_datasets))
num_test_dirs = len(dir_datasets) - num_train_dirs - num_val_dirs

train_dirs, val_dirs, test_dirs = dir_datasets[:num_train_dirs], dir_datasets[num_train_dirs:num_train_dirs+num_val_dirs], dir_datasets[num_train_dirs+num_val_dirs:]

# Define your augmentations
data_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    # Assuming images are PIL images; if not, you'll need to adjust or implement suitable transformations
    transforms.RandomCrop(size=(10 * 64, 8 * 64), padding=4),
    # Add any other desired transforms here
])

train_dataset = MammogramDataset(train_dirs)#, transform=data_transforms)
val_dataset = MammogramDataset(val_dirs)
test_dataset = MammogramDataset(test_dirs)

mean, std = compute_target_statistics(train_dataset)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)



if __name__ == "__main__":
    from training.models import *


    # Initialize model, criterion, optimizer
    # model = SimpleCNN().cuda()  # Assuming you have a GPU. If not, remove .cuda()
    model = ResNetTransformer().cuda()
    epsilon = 0.
    # model = TransformerModel(epsilon=epsilon).cuda()
    criterion = nn.MSELoss()  # Mean squared error for regression
    lr = 0.001
    momentum = 0.9
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = DAdaptAdam(model.parameters())
    optimizer = DAdaptSGD(model.parameters())
    # optimizer = optim.SGD(model.parameters(),
    #                          lr=lr, momentum=momentum)

    # best_model_name += '_{}s'.format(lr)

    # Training parameters
    num_epochs = 600
    patience = 600
    not_improved = 0
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(patience/10), factor=0.9, verbose=True)

    print("Beginning training")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        scaled_train_loss = 0.0
        for inputs, targets in train_loader:  # Simplified unpacking
            inputs, targets = inputs.cuda(), targets.cuda()  # Send data to GPU

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs.unsqueeze(1))  # Add channel dimension
            loss = criterion(outputs.squeeze(1), targets.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            with torch.no_grad():
                train_outputs_original_scale = inverse_standardize_targets(outputs.squeeze(1), mean, std)
                train_targets_original_scale = inverse_standardize_targets(targets.float(), mean, std)
                scaled_train_loss += criterion(train_outputs_original_scale, train_targets_original_scale).item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        scaled_train_loss /= len(train_loader.dataset)

        # Validation
        val_loss, val_labels, val_preds, val_r2 = evaluate_model(model, val_loader, criterion,
                                                                 inverse_standardize_targets, mean, std)

        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            not_improved = 0
            print("Validation loss improved. Saving best_model.")
            torch.save(model.state_dict(), 'models/'+best_model_name)
            val_test_loss, test_labels, test_preds, test_r2 = evaluate_model(model, test_loader, criterion,
                                                                         inverse_standardize_targets, mean, std)
            best_test_loss = val_test_loss
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {scaled_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Test loss: {val_test_loss:.4f} "
                  f"\nBest val: {best_val_loss:.4f} at epoch {epoch - not_improved} had test loss {best_test_loss:.4f}")
        else:
            val_test_loss, test_labels, test_preds, test_r2 = evaluate_model(model, test_loader, criterion,
                                                                         inverse_standardize_targets, mean, std)
            not_improved += 1
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {scaled_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Test loss: {val_test_loss:.4f} "
                  f"\nBest val: {best_val_loss:.4f} at epoch {epoch - not_improved} had test loss {best_test_loss:.4f}")
            if not_improved >= patience:
                print("Early stopping")
                break

        # scheduler.step(val_loss)

    # # Test the best model
    # model.load_state_dict(torch.load('models/'+best_model_name))
    #
    # test_loss, test_labels, test_preds, test_r2 = evaluate_model(model, test_loader, criterion,
    #                                                                  inverse_standardize_targets, mean, std)

    print(f"Test Loss: {val_test_loss:.4f}")

    print("Loading best model weights!")
    model.load_state_dict(torch.load('models/'+best_model_name))

    # Evaluating on all datasets: train, val, test
    train_loss, train_labels, train_preds, train_r2 = evaluate_model(model, train_loader, criterion, inverse_standardize_targets, mean, std)
    val_loss, val_labels, val_preds, val_r2 = evaluate_model(model, val_loader, criterion, inverse_standardize_targets, mean, std)
    test_loss, test_labels, test_preds, test_r2 = evaluate_model(model, test_loader, criterion, inverse_standardize_targets, mean, std)

    # R2 Scores
    print(f"Train R2 Score: {train_r2:.4f}")
    print(f"Validation R2 Score: {val_r2:.4f}")
    print(f"Test R2 Score: {test_r2:.4f}")
    print(f"Test Loss: {val_test_loss:.4f}")

    # Scatter plots
    plot_scatter(train_labels, train_preds, "Train Scatter Plot "+best_model_name)
    plot_scatter(val_labels, val_preds, "Validation Scatter Plot "+best_model_name)
    plot_scatter(test_labels, test_preds, "Test Scatter Plot "+best_model_name)

    # Error distributions
    plot_error_distribution(train_labels, train_preds, "Train Error Distribution "+best_model_name)
    plot_error_distribution(val_labels, val_preds, "Validation Error Distribution "+best_model_name)
    plot_error_distribution(test_labels, test_preds, "Test Error Distribution "+best_model_name)

    print("Done")

