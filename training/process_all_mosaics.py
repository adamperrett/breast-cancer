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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage import filters
from training.models import *


seed_value = 272727
np.random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sns.set(style='dark')

def extract_identifier(filename):
    match = re.search(r'-([\d]+)-[A-Z]+', filename)
    return match.group(1) if match else None

print("Reading data")

image_directory = 'D:/mosaic_data/raw'
csv_directory = 'C:/Users/adam_/PycharmProjects/breast-cancer/data'
csv_name = 'full_procas_info3.csv'

mosaic_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')
mosaic_ids = mosaic_data['Unnamed: 0']
vas_density_data = mosaic_data['VASCombinedAvDensity']

processed = True
# processed_dataset_path = os.path.join(csv_directory, 'mosaics_processed/full_scaled_dataset.pth')
processed_dataset_path = os.path.join(csv_directory, 'mosaics_processed/full_mosaic_dataset_histo.pth')
image_statistics_pre = []
image_statistics_post = []

results_dir = 'C:/Users/adam_/PycharmProjects/breast-cancer/training/results'
best_model_name = 'histo_rand_two_ResnetTransformer'

# save mosaic_ids which have a vas score and Study_type was Breast Screening
id_vas_dict = {}
for id, vas in zip(mosaic_ids, vas_density_data):
    if not np.isnan(vas) and vas > 0:
        id_vas_dict["PROCAS_ALL_{:05}".format(id)] = vas

mosaic_ids = mosaic_ids.unique()

def pre_process_mammograms(mammographic_images, sides, heights, widths, pixel_sizes, image_types):
    target_pixel_size = 0.0941
    processed_images = []
    print("Beginning processing images")
    print(heights, widths)
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
            image_statistics_pre.append(compute_sample_weights(np.ravel(mammographic_image), n_bins=100, only_bins=True))
            mammographic_image += (mammographic_image == 0.0).astype(int) * 1
            mammographic_image = np.log(mammographic_image)
            mammographic_image = np.amax(mammographic_image) - mammographic_image
        cut_off = mammographic_image > filters.threshold_otsu(mammographic_image)
        cut_off = cut_off.astype(float)
        mammographic_image = cut_off * mammographic_image
        padded_image = np.zeros((max(2995, mammographic_image.shape[0]), max(2394, mammographic_image.shape[1])))
        padded_image[:mammographic_image.shape[0], :mammographic_image.shape[1]] = mammographic_image
        mammographic_image = resize(padded_image[:2995, :2394], (10 * 64, 8 * 64))
        mammographic_image = mammographic_image / np.amax(mammographic_image)
        image_statistics_pre.append(compute_sample_weights(np.ravel(mammographic_image),
                                                           n_bins=100,
                                                           only_bins=True,
                                                           minv=0,
                                                           maxv=1))
        processed_images.append(mammographic_image)
    return torch.stack([torch.from_numpy(img).float() for img in processed_images], dim=0)

# This function will preprocess and zip all images and return a dataset ready for saving
def preprocess_and_zip_all_images(parent_directory, id_vas_dict):
    dataset_entries = []

    patient_dirs = [d for d in os.listdir(parent_directory) if d in id_vas_dict]
    patient_dirs.sort()  # Ensuring a deterministic order

    for patient_dir in tqdm(patient_dirs):
        patient_path = os.path.join(parent_directory, patient_dir)
        image_files = [f for f in os.listdir(patient_path) if f.endswith('.dcm')]

        # Load all images for the given patient/directory
        dcm_files = [pydicom.dcmread(os.path.join(patient_path, f), force=True) for f in image_files]
        if not all([hasattr(dcm, 'StudyDescription') for dcm in dcm_files]):
            print("StudyDescription attribute missing")
        else:
            studies = [dcm.StudyDescription for dcm in dcm_files]
            if any(s != 'Breast Screening' and
                   s != 'XR MAMMOGRAM BILATERAL' and
                   s != 'BILATERAL MAMMOGRAMS 2 VIEWS' for s in studies):
                print("Skipped because not all breast screening - studies =", studies)
                continue
        all_images = [dcm.pixel_array for dcm in dcm_files]
        all_sides = ['L' if 'LCC' in f or 'LMLO' in f else 'R' for f in image_files]
        all_heights = [img.shape[0] for img in all_images]
        if any(num > 4000 for num in all_heights):
            continue
        all_widths = [img.shape[1] for img in all_images]
        if any(num > 4000 for num in all_widths):
            continue
        all_pixel_sizes = [(0.0941, 0.0941) for _ in all_images]
        all_image_types = ['raw' if ('raw' in patient_path or 'RAW' in patient_path or '_PROC_' not in patient_path)
                           else 'processed' for _ in image_files]

        preprocessed_images = pre_process_mammograms(all_images, all_sides, all_heights, all_widths, all_pixel_sizes,
                                                     all_image_types)

        dataset_entries.append((preprocessed_images, id_vas_dict[patient_dir]))

    return dataset_entries

def custom_collate(batch):
    # Separate images and labels
    images, labels, weights = zip(*batch)

    # Determine the max combined width
    max_width = max([sum(img.size(-1) for img in img_list) for img_list in images])

    # Stack images horizontally with padding
    stacked_images = []
    for img_list in images:
        combined_width = sum(img.size(-1) for img in img_list)
        padding_size = max_width - combined_width
        combined_img = torch.cat(tuple(img_list), dim=-1)  # Use tuple() here
        if padding_size > 0:
            # Pad on the right
            combined_img = pad(combined_img, (0, 0, padding_size, 0))
        stacked_images.append(combined_img)

    # Stack the processed images into a batch
    images_tensor = torch.stack(stacked_images)

    # Convert the list of regression targets to a tensor and standardize
    labels_tensor = torch.tensor(labels, dtype=torch.float32)  # Change the dtype if needed
    labels_tensor = standardize_targets(labels_tensor, mean, std)

    # if weights[0]:
    return images_tensor, labels_tensor, torch.tensor(weights, dtype=torch.float32)
    # else:
    #     return images_tensor, labels_tensor, weights

if not processed:
    # Generate the dataset and save it
    if not os.path.exists(processed_dataset_path):
        dataset_entries = preprocess_and_zip_all_images(image_directory, id_vas_dict)
        torch.save(dataset_entries, processed_dataset_path)



# Define your augmentations
data_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    # Assuming images are PIL images; if not, you'll need to adjust or implement suitable transformations
    transforms.RandomCrop(size=(10 * 64, 8 * 64), padding=4),
    # Add any other desired transforms here
])

class MammogramDataset(Dataset):
    def __init__(self, dataset_path, transform=None, n=2, weights=None, rand_select=True):
        self.dataset = torch.load(dataset_path)
        self.transform = transform
        self.n = n
        self.weights = weights
        self.rand_select = rand_select

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if len(image) < self.n:
            for i in range(self.n - len(image)):
                image = torch.vstack([image, torch.zeros_like(image[0]).unsqueeze(0)])
        if self.rand_select:
            sample = random.sample(range(len(image)), self.n)
        else:
            sample = range(self.n)
        if self.transform:
            transformed_image = [self.transform(im.unsqueeze(0)).squeeze(0) for im in image[sample]]
            image = transformed_image
        else:
            image = image[sample]

        # If weights are provided, return them as well
        if self.weights is not None:
            return image, label, self.weights[idx]
        else:
            return image, label, 1

# Load dataset from saved path
print("Creating Dataset")
dataset = MammogramDataset(processed_dataset_path)

# Splitting the dataset
train_ratio, val_ratio = 0.7, 0.2
num_train = int(train_ratio * len(dataset))
num_val = int(val_ratio * len(dataset))
num_test = len(dataset) - num_train - num_val

train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

# Compute weights for the training set
# targets = [label for _, label in train_dataset.dataset.dataset]
# sample_weights = compute_sample_weights(targets)
sample_weights = None

# Applying the transform only to the training dataset
train_dataset.dataset = MammogramDataset(processed_dataset_path, transform=data_transforms)#, weights=sample_weights)

mean, std = compute_target_statistics(train_dataset)

# from torch.utils.data import WeightedRandomSampler
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset))

# Use this sampler in your DataLoader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=custom_collate)

# Create DataLoaders
print("Creating DataLoaders")
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)



if __name__ == "__main__":

    # Initialize model, criterion, optimizer
    # model = SimpleCNN().cuda()  # Assuming you have a GPU. If not, remove .cuda()
    model = ResNetTransformer().cuda()
    epsilon = 0.
    # model = TransformerModel(epsilon=epsilon).cuda()
    criterion = nn.MSELoss(reduction='none')  # Mean squared error for regression
    lr = 1
    momentum = 0.9
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = DAdaptAdam(model.parameters())
    optimizer = DAdaptSGD(model.parameters())
    # optimizer = optim.SGD(model.parameters(),
    #                          lr=lr, momentum=momentum)

    # best_model_name += '_{}mda_bs'.format(epsilon)

    # Training parameters
    num_epochs = 600
    patience = 600
    not_improved = 0
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(patience/10), factor=0.9, verbose=True)
    writer = SummaryWriter(results_dir+'/tb_'+best_model_name)

    print("Beginning training")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        all_targets = []
        all_predictions = []
        train_loss = 0.0
        scaled_train_loss = 0.0
        for inputs, targets, weights in train_loader:  # Simplified unpacking
            inputs, targets, weights = inputs.cuda(), targets.cuda(), weights.cuda()  # Send data to GPU

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs.unsqueeze(1))  # Add channel dimension
            losses = criterion(outputs.squeeze(1), targets.float())  # Get losses for each sample
            weighted_loss = (losses * weights).mean()  # Weighted loss

            # Backward + optimize
            weighted_loss.backward()
            optimizer.step()

            train_loss += weighted_loss.item() * inputs.size(0)

            with torch.no_grad():
                train_outputs_original_scale = inverse_standardize_targets(outputs.squeeze(1), mean, std)
                train_targets_original_scale = inverse_standardize_targets(targets.float(), mean, std)
                all_targets.extend(train_targets_original_scale.cpu().numpy())
                all_predictions.extend(train_outputs_original_scale.cpu().numpy())
                scaled_train_loss += criterion(train_outputs_original_scale,
                                               train_targets_original_scale).mean().item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        scaled_train_loss /= len(train_loader.dataset)

        train_r2 = r2_score(all_targets, all_predictions)
        # Validation
        val_loss, val_labels, val_preds, val_r2 = evaluate_model(model, val_loader, criterion,
                                                                 inverse_standardize_targets, mean, std)
        val_test_loss, test_labels, test_preds, test_r2 = evaluate_model(model, test_loader, criterion,
                                                                     inverse_standardize_targets, mean, std)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"\nTrain Loss: {scaled_train_loss:.4f}, Val Loss: {val_loss:.4f}, Test loss: {val_test_loss:.4f}"
              f"\nTrain R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, Test R2: {test_r2:.4f}")

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('R2/Train', train_r2, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('R2/Validation', val_r2, epoch)
        writer.add_scalar('Loss/Test', val_test_loss, epoch)
        writer.add_scalar('R2/Test', test_r2, epoch)

        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            not_improved = 0
            print("Validation loss improved. Saving best_model.")
            torch.save(model.state_dict(), 'models/'+best_model_name)
            best_test_loss = val_test_loss
            print(f"Best val: {best_val_loss:.4f} at epoch {epoch - not_improved} had test loss {best_test_loss:.4f}")
        else:
            not_improved += 1
            print(f"Best val: {best_val_loss:.4f} at epoch {epoch - not_improved} had test loss {best_test_loss:.4f}")
            if not_improved >= patience:
                print("Early stopping")
                break

        # scheduler.step(val_loss)


    writer.close()
    print("Loading best model weights!")
    model.load_state_dict(torch.load('models/'+best_model_name))

    train_dataset.dataset = MammogramDataset(processed_dataset_path, transform=None)

    # Evaluating on all datasets: train, val, test
    train_loss, train_labels, train_preds, train_r2 = evaluate_model(model, train_loader, criterion, inverse_standardize_targets, mean, std)
    val_loss, val_labels, val_preds, val_r2 = evaluate_model(model, val_loader, criterion, inverse_standardize_targets, mean, std)
    test_loss, test_labels, test_preds, test_r2 = evaluate_model(model, test_loader, criterion, inverse_standardize_targets, mean, std)

    # R2 Scores
    print(f"Train R2 Score: {train_r2:.4f}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation R2 Score: {val_r2:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Test R2 Score: {test_r2:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Scatter plots
    plot_scatter(train_labels, train_preds, "Train Scatter Plot "+best_model_name, results_dir)
    plot_scatter(val_labels, val_preds, "Validation Scatter Plot "+best_model_name, results_dir)
    plot_scatter(test_labels, test_preds, "Test Scatter Plot "+best_model_name, results_dir)

    # Error distributions
    plot_error_distribution(train_labels, train_preds, "Train Error Distribution "+best_model_name, results_dir)
    plot_error_distribution(val_labels, val_preds, "Validation Error Distribution "+best_model_name, results_dir)
    plot_error_distribution(test_labels, test_preds, "Test Error Distribution "+best_model_name, results_dir)

    print("Done")

'''

import matplotlib.pyplot as plt

def visualize_tensor(tensor):
    plt.figure()
    plt.imshow(tensor, cmap='gray')  # 'gray' for grayscale. Remove if you want colormap
    plt.colorbar()
    plt.show()

visualize_tensor(inputs[0].cpu().squeeze(0))
visualize_tensor(inputs[1].cpu().squeeze(0))

'''