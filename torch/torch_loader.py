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

def extract_identifier(filename):
    match = re.search(r'-([\d]+)-[A-Z]+', filename)
    return match.group(1) if match else None

print("Reading data")

parent_directory = '../data/procas_raw_sample'
image_paths = glob.glob(os.path.join(parent_directory, '*/*PROCAS*.dcm'))
df = pd.DataFrame({'image_path': image_paths})
df['identifier'] = df['image_path'].apply(lambda x: extract_identifier(os.path.basename(x)))

df_csv = pd.read_csv('../data/full_procas_info3.csv', sep=',')

# Extract unique identifiers from df
unique_identifiers = df['identifier'].unique().astype(int)
# Filter df_csv based on these unique identifiers
filtered_df = df_csv[df_csv['Unnamed: 0'].isin(unique_identifiers)]

# Select only the 'VASCombinedAvDensity' column
vas_density_data = filtered_df['VASCombinedAvDensity']

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

chosen_identifiers = random.sample(list(df['identifier'].unique()), 4)
dfs = [[df[(df['identifier'] == identifier) & (df['image_path'].str.contains(view))] for view in views] for identifier in chosen_identifiers]

# fig, axes = plt.subplots(4, len(views), figsize=(15, 15))
# for row, identifier_dfs in enumerate(dfs):
#     for idx, view in enumerate(views):
#         if not identifier_dfs[idx].empty:
#             path = identifier_dfs[idx]['image_path'].iloc[0]
#             dicom_data = pydicom.dcmread(path, force=True)
#             if hasattr(dicom_data, 'pixel_array'):
#                 axes[row, idx].imshow(dicom_data.pixel_array, cmap='gray')
#         axes[row, idx].set_xticks([])
#         axes[row, idx].set_yticks([])
#         if row == 0:
#             axes[row, idx].set_title(f'{view}')
#         if idx == 0:
#             axes[row, idx].set_ylabel(f'ID: {chosen_identifiers[row]}', rotation=0, labelpad=50, verticalalignment='center')
#
# plt.tight_layout()
# plt.show()

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


class MammogramDataset(Dataset):
    def __init__(self, images, targets, identifiers, sides, heights, widths, pixel_sizes, image_types):
        self.images = images
        self.targets = targets
        self.identifiers = identifiers
        self.sides = sides
        self.heights = heights
        self.widths = widths
        self.pixel_sizes = pixel_sizes
        self.image_types = image_types

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (self.images[idx], self.targets[idx], self.identifiers[idx], self.sides[idx], self.heights[idx],
                self.widths[idx], self.pixel_sizes[idx], self.image_types[idx])


# Load and preprocess all images
all_images = [pydicom.dcmread(path, force=True).pixel_array for path in df['image_path']]
all_sides = ['L' if 'LCC' in path or 'LMLO' in path else 'R' for path in df['image_path']]
all_heights = df['height'].tolist()
all_widths = df['width'].tolist()
# Assuming the same pixel size for all images for simplicity, modify as needed
all_pixel_sizes = [(0.0941, 0.0941) for _ in range(len(all_images))]
# all_image_types = ['raw' if 'raw' in path else 'processed' for path in df['image_path']]
all_image_types = ['raw' if 'raw' in parent_directory else 'processed' for path in df['image_path']]

processed = True

if not processed:
    preprocessed_images = pre_process_mammograms(all_images, all_sides, all_heights, all_widths, all_pixel_sizes, all_image_types)
    # Include VASCombinedAvDensity as targets.
    # all_targets = vas_density_data.tolist()

    # Initialize dataset with the targets
    all_indentities = df['identifier'].astype(int).tolist()
    all_targets = [vas_density_data[i] for i in all_indentities]
    dataset = MammogramDataset(preprocessed_images, all_targets, all_indentities, all_sides,
                               all_heights, all_widths, all_pixel_sizes, all_image_types)

    torch.save(dataset, '../data/procas_raw_sample/dataset.pth')
else:
    print("Loading data")
    dataset = torch.load('../data/procas_raw_sample/dataset.pth')

print("Creating dataloaders")
random.shuffle(unique_identifiers)
train_ratio, val_ratio = 0.7, 0.2
num_train = int(train_ratio * len(unique_identifiers))
num_val = int(val_ratio * len(unique_identifiers))
train_ids, val_ids, test_ids = unique_identifiers[:num_train], unique_identifiers[num_train:num_train+num_val], unique_identifiers[num_train+num_val:]

# indices_only = [i for i, num in enumerate(dataset.identifiers) if num in train_ids.astype(int)]
train_dataset = [dataset[i] for i, num in enumerate(dataset.identifiers) if num in train_ids.astype(int)]
val_dataset = [dataset[i] for i, num in enumerate(dataset.identifiers) if num in val_ids.astype(int)]
test_dataset = [dataset[i] for i, num in enumerate(dataset.identifiers) if num in test_ids.astype(int)]

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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


    class TransformerModel(nn.Module):
        def __init__(self, embed_dim=512, num_heads=8, num_layers=2, num_classes=1):
            super(TransformerModel, self).__init__()

            self.patch_embed = PatchEmbedding(embed_dim=embed_dim)

            # Transformer Encoder layers
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

            # Classifier head
            self.classifier = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            # Create patch embeddings
            x = self.patch_embed(x)

            # Pass through transformer
            x = self.transformer_encoder(x)

            # Global average pooling and classifier
            x = x.mean(dim=1)  # (B, embed_dim)
            x = self.classifier(x)

            return x


    # Initialize model, criterion, optimizer
    # model = SimpleCNN().cuda()  # Assuming you have a GPU. If not, remove .cuda()
    # model = ResNetTransformer().cuda()
    model = TransformerModel().cuda()
    criterion = nn.MSELoss()  # Mean squared error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        for data in train_loader:
            inputs, targets, _, _, _, _, _, _ = data  # Unpack data from DataLoader
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
            for data in val_loader:
                inputs, targets, _, _, _, _, _, _ = data
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
        for data in test_loader:
            inputs, targets, _, _, _, _, _, _ = data
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs.squeeze(), targets.float())
            test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}")

    print("Done")
