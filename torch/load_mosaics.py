import numpy
import torch

import os
import random
import glob
import re
import math

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import pad
from tqdm import tqdm

sns.set(style='dark')

def extract_identifier(filename):
    match = re.search(r'-([\d]+)-[A-Z]+', filename)
    return match.group(1) if match else None

print("Reading data")

parent_directory = '../data/mosaics_processed'

mosaic_data = pd.read_csv('../data/matched_mosaics.csv', sep=',')
mosaic_ids = mosaic_data['Patient']
vas_density_data = mosaic_data['VASCombinedAvDensity']

processed = True
processed_dataset_path = '../data/mosaics_processed/dataset_only_vas.pth'

# save mosaic_ids which have a vas score
id_vas_dict = {}
for id, vas in zip(mosaic_ids, vas_density_data):
    if not np.isnan(vas):
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

# This function will preprocess and zip all images and return a dataset ready for saving
def preprocess_and_zip_all_images(parent_directory, id_vas_dict):
    dataset_entries = []

    patient_dirs = [d for d in os.listdir(parent_directory) if d in id_vas_dict]
    patient_dirs.sort()  # Ensuring a deterministic order

    for patient_dir in tqdm(patient_dirs):
        patient_path = os.path.join(parent_directory, patient_dir)
        image_files = [f for f in os.listdir(patient_path) if f.endswith('.dcm')]

        # Load all images for the given patient/directory
        all_images = [pydicom.dcmread(os.path.join(patient_path, f), force=True).pixel_array for f in image_files]
        all_sides = ['L' if 'LCC' in f or 'LMLO' in f else 'R' for f in image_files]
        all_heights = [img.shape[0] for img in all_images]
        all_widths = [img.shape[1] for img in all_images]
        all_pixel_sizes = [(0.0941, 0.0941) for _ in all_images]
        all_image_types = ['raw' if ('raw' in patient_path or 'RAW' in patient_path)
                           else 'processed' for _ in image_files]

        preprocessed_images = pre_process_mammograms(all_images, all_sides, all_heights, all_widths, all_pixel_sizes,
                                                     all_image_types)

        dataset_entries.append((preprocessed_images, id_vas_dict[patient_dir]))

    return dataset_entries

def custom_collate(batch):
    # Separate images and labels
    images, labels = zip(*batch)

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

    # Convert the list of regression targets to a tensor
    labels_tensor = torch.tensor(labels, dtype=torch.float32)  # Change the dtype if needed

    return images_tensor, labels_tensor

if not processed:
    # Generate the dataset and save it
    if not os.path.exists(processed_dataset_path):
        dataset_entries = preprocess_and_zip_all_images(parent_directory, id_vas_dict)
        torch.save(dataset_entries, processed_dataset_path)


class MammogramDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset = torch.load(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# Load dataset from saved path
dataset = MammogramDataset(processed_dataset_path)

# Splitting the dataset
train_ratio, val_ratio = 0.7, 0.2
num_train = int(train_ratio * len(dataset))
num_val = int(val_ratio * len(dataset))
num_test = len(dataset) - num_train - num_val

train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

# Create DataLoaders
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)



if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torchvision.models import resnet34
    from torch.nn import TransformerEncoder, TransformerEncoderLayer


    # Define a simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self, dropout_prob=0.5):
            super(SimpleCNN, self).__init__()
            self.conv_layer = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.MaxPool2d(kernel_size=2),
            )
            self.fc_layer = nn.Sequential(
                nn.Linear(64 * 32 * 16 * 10, 256),
                nn.ReLU(),
                nn.Dropout(dropout_prob),

                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout_prob),

                nn.Linear(64, 1)  # Regression output
            )

        def forward(self, x):
            x = self.conv_layer(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layer(x)
            return x


    # define complex resnet into transformer model
    class ResNetTransformer(nn.Module):
        def __init__(self):
            super(ResNetTransformer, self).__init__()

            # Using ResNet-34 as a feature extractor
            self.resnet = resnet34(pretrained=False)  # set pretrained to False

            # Modify the first layer to accept single-channel (grayscale) images
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

            self.resnet.fc = nn.Identity()  # Removing the fully connected layer

            # Assuming we are using an average pool and get a 512-dimensional vector
            d_model = 512
            nhead = 4  # Number of self-attention heads
            num_encoder_layers = 2  # Number of Transformer encoder layers

            # Transformer Encoder layers
            encoder_layers = TransformerEncoderLayer(d_model, nhead)
            self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

            # Final regressor
            self.regressor = nn.Linear(d_model, 1)

        def forward(self, x):
            # Extract features using ResNet
            x = self.resnet(x)
            x = x.unsqueeze(1)  # Add sequence length dimension for Transformer

            # Pass features through Transformer
            x = self.transformer_encoder(x)

            # Regression
            x = x.squeeze(1)
            x = self.regressor(x)

            return x


    class PatchEmbedding(nn.Module):
        def __init__(self, in_channels=1, patch_size=16, embed_dim=512):
            super().__init__()
            self.patch_size = patch_size
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, x):
            x = self.proj(x)  # (B, embed_dim, H', W')
            x = x.flatten(2)  # (B, embed_dim, H'*W')
            x = x.transpose(1, 2)  # (B, H'*W', embed_dim)
            return x


    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)


    class TransformerModel(nn.Module):
        def __init__(self, embed_dim=256, num_heads=8, num_layers=2, num_classes=1):
            super(TransformerModel, self).__init__()
            self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
            self.pos_encoder = PositionalEncoding(embed_dim)

            # Transformer Encoder layers
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
            self.classifier = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            x = self.patch_embed(x)  # Patch embedding
            x = self.pos_encoder(x)  # Add positional encoding

            x = self.transformer_encoder(x)  # Transformer encoder
            x = x.mean(dim=1)  # Global average pooling
            x = self.classifier(x)  # Classifier

            return x


    # Initialize model, criterion, optimizer
    # model = SimpleCNN().cuda()  # Assuming you have a GPU. If not, remove .cuda()
    # model = ResNetTransformer().cuda()
    model = TransformerModel().cuda()
    criterion = nn.MSELoss()  # Mean squared error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training parameters
    num_epochs = 100
    patience = 20
    not_improved = 0
    best_val_loss = float('inf')
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(patience/3), factor=0.75, verbose=True)

    print("Beginning training")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:  # Simplified unpacking
            inputs, targets = inputs.cuda(), targets.cuda()  # Send data to GPU

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs.unsqueeze(1))  # Add channel dimension
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:  # Simplified unpacking
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs.unsqueeze(1))
                loss = criterion(outputs.squeeze(), targets.float())
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            not_improved = 0
            print("Validation loss improved. Saving best_model.")
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            not_improved += 1
            if not_improved >= patience:
                print("Early stopping")
                break

        scheduler.step(val_loss)

    # Test the best model
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:  # Simplified unpacking
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs.squeeze(), targets.float())
            test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}")

    print("Done")

