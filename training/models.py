import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet34
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

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

            nn.Linear(64, 1, bias=False)  # Regression output
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
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, x):
        x = self.patch_embed(x)  # Patch embedding
        x = self.pos_encoder(x)  # Add positional encoding

        x = self.transformer_encoder(x)  # Transformer encoder
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)  # Classifier

        return x


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
            loss = criterion(outputs.squeeze(), targets.float())
            running_loss += loss.item() * inputs.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    r2 = r2_score(all_targets, all_predictions)
    return epoch_loss, all_targets, all_predictions, r2