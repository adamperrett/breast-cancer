print("Starting imports")
import torch
from dadaptation import DAdaptAdam, DAdaptSGD
import sys
import os
import re
import pydicom
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import pad
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage import filters

on_CSF = True
if on_CSF:
    print("Running on the CSF")
    training_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, training_path)
    from models import *
else:
    from training.models import *

# time.sleep(60*60*14)
print(time.localtime())
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


if on_CSF:
    '''
    Test across:
    -n images 4
    -dataset 4
    -batch_size 6
    -optimiser 2
    -weights 2
    -transformed 2
    '''
    configurations = []
    for n_im in [8, 4, 2, 1]:
        for b_size in [128, 64, 32, 24, 16, 8]:
            for op_choice in ['adam', 'sgd']:
                for d_set in ['proc', 'log', 'histo', 'clahe']:
                    for weight_choice in [True, False]:
                        for trans_choice in [True, False]:
                            configurations.append({
                                'dataset': d_set,
                                'batch_size': b_size,
                                'optimizer': op_choice,
                                'weighted': weight_choice,
                                'transformed': trans_choice,
                                'n_images': n_im
                            })
    config = int(sys.argv[1]) - 1

    processed = True
    dataset_dir = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast-cancer/'
    config = configurations[config]
    processed_dataset_path = os.path.join(dataset_dir,
                                          'mosaics_processed/full_mosaic_dataset_{}.pth'.format(config['dataset']))
    batch_size = config['batch_size']
    op_choice = config['optimizer']
    weighted = config['weighted']
    transformed = config['transformed']
    n_images = config['n_images']

    working_dir = '/mnt/iusers01/gb01/mbaxrap7/scratch/breast-cancer/training/'
    best_model_name = 'b2f_{}_{}x{}_{}_t{}_w{}_js{}'.format(
        config['dataset'], op_choice, batch_size, n_images, transformed, weighted, int(sys.argv[1]))

    print("Config", int(sys.argv[1]) + 1, "creates test", best_model_name)
else:
    processed = True

    batch_size = 16
    op_choice = 'adam'
    weighted = False
    transformed = False
    n_images = 2

    image_directory = 'D:/mosaic_data/raw'
    csv_directory = 'C:/Users/adam_/PycharmProjects/breast-cancer/data'
    csv_name = 'full_procas_info3.csv'
    reference_csv = 'PROCAS_reference.csv'

    keyword = 'testing_prints'
    dataset = 'proc'

    processed_dataset_path = os.path.join(csv_directory, 'mosaics_processed/full_mosaic_dataset_{}.pth'.format(dataset))
    # processed_dataset_path = os.path.join(csv_directory, 'mosaics_processed/full_mosaic_dataset_log.pth')

    working_dir = 'C:/Users/adam_/PycharmProjects/breast-cancer/training/'
    best_model_name = '{}_{}_{}_{}x{}_t{}_w{}'.format(
        keyword, dataset, op_choice, batch_size, n_images, transformed, weighted)



no_study_files = []
bad_study_files = []
bad_images = []

if not processed:
    mosaic_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')
    mosaic_ids = mosaic_data['ProcID']
    vas_density_data = mosaic_data['VASCombinedAvDensity']

    reference_data = pd.read_csv(os.path.join(csv_directory, reference_csv), sep=',')
    reference_ids = reference_data['ProcID']
    if 'raw' in image_directory:
        im_ids = reference_data['ASSURE_RAW_ID']
    else:
        im_ids = reference_data['ASSURE_PROCESSED_ANON_ID']

    raw_PROC_id_dict = {}
    for ref, id in zip(reference_ids, im_ids):
        if not np.isnan(id):
            raw_PROC_id_dict[ref] = int(id)

    id_vas_dict = {}
    for id, vas in zip(mosaic_ids, vas_density_data):
        if not np.isnan(vas) and vas > 0 and id in raw_PROC_id_dict:
            raw_id = raw_PROC_id_dict[id]
            id_vas_dict["{:05}".format(raw_id)] = vas

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
        if side == 'R':
            mammographic_image = np.fliplr(mammographic_image)
        if image_type == 'raw':
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
        if np.sum(np.isnan(mammographic_image)) == 0:
            processed_images.append(mammographic_image)
        else:
            print("Bad nan image")
    return torch.stack([torch.from_numpy(img).float() for img in processed_images], dim=0)

# This function will preprocess and zip all images and return a dataset ready for saving
def preprocess_and_zip_all_images(parent_directory, id_vas_dict):
    dataset_entries = []

    # patient_dirs = [d for d in os.listdir(parent_directory) if d in id_vas_dict]
    patient_dirs = [d for d in os.listdir(parent_directory) if d[-5:] in id_vas_dict]
    patient_dirs.sort()  # Ensuring a deterministic order

    for patient_dir in tqdm(patient_dirs):
        patient_path = os.path.join(parent_directory, patient_dir)
        image_files = [f for f in os.listdir(patient_path) if f.endswith('.dcm')]

        # Load all images for the given patient/directory
        dcm_files = [pydicom.dcmread(os.path.join(patient_path, f), force=True) for f in image_files]
        try:
            all_images = [dcm.pixel_array for dcm in dcm_files]
            if np.sum([np.sum(im) == 0 for im in all_images]):
                bad_images.append([patient_dir, dcm_files])
                continue
        except:
            continue
        if not all([hasattr(dcm, 'StudyDescription') for dcm in dcm_files]):
            print("StudyDescription attribute missing")
            all_sides = ['L' if 'LCC' in f or 'LMLO' in f else 'R' for f in image_files]
            all_heights = [img.shape[0] for img in all_images]
            if any(num > 4000 for num in all_heights):
                continue
            all_widths = [img.shape[1] for img in all_images]
            if any(num > 4000 for num in all_widths):
                continue
            all_pixel_sizes = [(0.0941, 0.0941) for _ in all_images]
            all_image_types = ['raw' if ('raw' in patient_path or 'RAW' in patient_path or '_PROC' not in patient_path)
                               else 'processed' for _ in image_files]

            preprocessed_images = pre_process_mammograms(all_images, all_sides, all_heights, all_widths, all_pixel_sizes,
                                                         all_image_types)

            no_study_files.append([preprocessed_images, patient_dir, dcm_files])
        else:
            studies = [dcm.StudyDescription for dcm in dcm_files]
            if any(s != 'Breast Screening' and
                   s != 'XR MAMMOGRAM BILATERAL' and
                   s != 'BILATERAL MAMMOGRAMS 2 VIEWS' for s in studies):
                print("Skipped because not all breast screening - studies =", studies)
                all_sides = ['L' if 'LCC' in f or 'LMLO' in f else 'R' for f in image_files]
                all_heights = [img.shape[0] for img in all_images]
                if any(num > 4000 for num in all_heights):
                    continue
                all_widths = [img.shape[1] for img in all_images]
                if any(num > 4000 for num in all_widths):
                    continue
                all_pixel_sizes = [(0.0941, 0.0941) for _ in all_images]
                all_image_types = [
                    'raw' if ('raw' in patient_path or 'RAW' in patient_path or '_PROC' not in patient_path)
                    else 'processed' for _ in image_files]

                preprocessed_images = pre_process_mammograms(all_images, all_sides, all_heights, all_widths,
                                                             all_pixel_sizes,
                                                             all_image_types)

                bad_study_files.append([preprocessed_images, patient_dir, dcm_files, studies])
                continue
        # all_images = [dcm.pixel_array for dcm in dcm_files]
        all_sides = ['L' if 'LCC' in f or 'LMLO' in f else 'R' for f in image_files]
        all_heights = [img.shape[0] for img in all_images]
        if any(num > 4000 for num in all_heights):
            continue
        all_widths = [img.shape[1] for img in all_images]
        if any(num > 4000 for num in all_widths):
            continue
        all_pixel_sizes = [(0.0941, 0.0941) for _ in all_images]
        all_image_types = ['raw' if ('raw' in patient_path or 'RAW' in patient_path or '_PROC' not in patient_path)
                           else 'processed' for _ in image_files]

        preprocessed_images = pre_process_mammograms(all_images, all_sides, all_heights, all_widths, all_pixel_sizes,
                                                     all_image_types)

        dataset_entries.append((preprocessed_images, id_vas_dict[patient_dir[-5:]]))

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


class MammogramDataset(Dataset):
    def __init__(self, dataset_path, transform=None, n=8, weights=None, rand_select=True):
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


if not processed:
    print("Processing data")
    # Generate the dataset and save it
    if not os.path.exists(processed_dataset_path):
        os.mkdir(processed_dataset_path)
    dataset_entries = preprocess_and_zip_all_images(image_directory, id_vas_dict)
    torch.save(dataset_entries, processed_dataset_path)

# Load dataset from saved path
print("Creating Dataset")
dataset = MammogramDataset(processed_dataset_path,
                           n=n_images)

# Splitting the dataset
train_ratio, val_ratio = 0.7, 0.2
num_train = int(train_ratio * len(dataset))
num_val = int(val_ratio * len(dataset))
num_test = len(dataset) - num_train - num_val

train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

if transformed:
    # Define your augmentations
    data_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        # Assuming images are PIL images; if not, you'll need to adjust or implement suitable transformations
        transforms.RandomCrop(size=(10 * 64, 8 * 64), padding=4),
        # Add any other desired transforms here
    ])
else:
    data_transforms = None

# Compute weights for the training set
if weighted:
    targets = [label for _, label in train_dataset.dataset.dataset]
    sample_weights = compute_sample_weights(targets)
else:
    sample_weights = None

# Applying the transform only to the training dataset
train_dataset.dataset = MammogramDataset(processed_dataset_path,
                                         transform=data_transforms,
                                         weights=sample_weights,
                                         n=n_images)

mean, std = compute_target_statistics(train_dataset)

# from torch.utils.data import WeightedRandomSampler
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset))

# Use this sampler in your DataLoader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=custom_collate)

# Create DataLoaders
print("Creating DataLoaders")
# batch_size = 16
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
    if op_choice == 'adam':
        optimizer = DAdaptAdam(model.parameters())
    elif op_choice == 'sgd':
        optimizer = DAdaptSGD(model.parameters())
    # optimizer = optim.SGD(model.parameters(),
    #                          lr=lr, momentum=momentum)

    # best_model_name += '_{}mda_bs'.format(epsilon)

    # Training parameters
    num_epochs = 600
    patience = 150
    not_improved = 0
    not_improved_r2 = 0
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_val_l_r2 = -float('inf')
    best_test_l_r2 = -float('inf')
    best_val_r_loss = float('inf')
    best_test_r_loss = float('inf')
    best_val_r2 = -float('inf')
    best_test_r2 = -float('inf')
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(patience/10), factor=0.9, verbose=True)
    writer = SummaryWriter(working_dir+'/results/tb_'+best_model_name)

    print("Beginning training")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        all_targets = []
        all_predictions = []
        train_loss = 0.0
        scaled_train_loss = 0.0
        for inputs, targets, weights in train_loader:  # Simplified unpacking
            inputs, targets, weights = inputs.cuda(), targets.cuda(), weights.cuda()  # Send data to GPU
            if torch.sum(torch.isnan(inputs)) > 0:
                print("Image be fucked")

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
        test_loss, test_labels, test_preds, test_r2 = evaluate_model(model, test_loader, criterion,
                                                                     inverse_standardize_targets, mean, std)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"\nTrain Loss: {scaled_train_loss:.4f}, Val Loss: {val_loss:.4f}, Test loss: {test_loss:.4f}"
              f"\nTrain R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, Test R2: {test_r2:.4f}")

        writer.add_scalar('Loss/Train', scaled_train_loss, epoch)
        writer.add_scalar('R2/Train', train_r2, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('R2/Validation', val_r2, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('R2/Test', test_r2, epoch)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_test_r2 = test_r2
            best_val_r_loss = val_loss
            best_test_r_loss = test_loss
            not_improved_r2 = 0
            print("Validation R2 improved. Saving best_model.")
            torch.save(model.state_dict(), working_dir+'/models/r_'+best_model_name)
        else:
            not_improved_r2 += 1
        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_val_l_r2 = val_r2
            best_test_l_r2 = test_r2
            not_improved = 0
            print("Validation loss improved. Saving best_model.")
            print(f"From best val loss at epoch {epoch - not_improved}:\n "
                  f"val loss: {best_val_loss:.4f} test loss {best_test_loss:.4f} val r2: {best_val_l_r2:.4f} test r2 {best_test_l_r2:.4f}")
            print(f"From best val R2 at epoch {epoch - not_improved_r2}:\n "
                  f"val loss: {best_val_r_loss:.4f} test loss {best_test_r_loss:.4f} val r2: {best_val_r2:.4f} test r2 {best_test_r2:.4f}")
            torch.save(model.state_dict(), working_dir+'/models/l_'+best_model_name)
        else:
            not_improved += 1
            print(f"From best val loss at epoch {epoch - not_improved}:\n "
                  f"val loss: {best_val_loss:.4f} test loss {best_test_loss:.4f} val r2: {best_val_l_r2:.4f} test r2 {best_test_l_r2:.4f}")
            print(f"From best val R2 at epoch {epoch - not_improved_r2}:\n "
                  f"val loss: {best_val_r_loss:.4f} test loss {best_test_r_loss:.4f} val r2: {best_val_r2:.4f} test r2 {best_test_r2:.4f}")
            if not_improved >= patience:
                print("Early stopping")
                break

        writer.add_scalar('Loss/Best Validation Loss from Loss', best_val_loss, epoch)
        writer.add_scalar('Loss/Best Validation Loss from R2', best_val_r_loss, epoch)
        writer.add_scalar('R2/Best Validation R2 from R2', best_val_r2, epoch)
        writer.add_scalar('R2/Best Validation R2 from Loss', best_val_l_r2, epoch)
        writer.add_scalar('Loss/Best Test Loss from Loss', best_test_loss, epoch)
        writer.add_scalar('Loss/Best Test Loss from R2', best_test_r_loss, epoch)
        writer.add_scalar('R2/Best Test R2 from R2', best_test_r2, epoch)
        writer.add_scalar('R2/Best Test R2 from Loss', best_test_l_r2, epoch)

        # scheduler.step(val_loss)


    writer.close()
    print("Loading best model weights!")
    model.load_state_dict(torch.load(working_dir+'/models/l_'+best_model_name))

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
    plot_scatter(train_labels, train_preds, "Train Scatter Plot "+best_model_name, working_dir+'/results/')
    plot_scatter(val_labels, val_preds, "Validation Scatter Plot "+best_model_name, working_dir+'/results/')
    plot_scatter(test_labels, test_preds, "Test Scatter Plot "+best_model_name, working_dir+'/results/')

    # Error distributions
    plot_error_vs_vas(train_labels, train_preds, "Train Error vs VAS "+best_model_name, working_dir+'/results/')
    plot_error_vs_vas(val_labels, val_preds, "Validation Error vs VAS "+best_model_name, working_dir+'/results/')
    plot_error_vs_vas(test_labels, test_preds, "Test Error vs VAS "+best_model_name, working_dir+'/results/')

    # Error distributions
    plot_error_distribution(train_labels, train_preds, "Train Error Distribution "+best_model_name, working_dir+'/results/')
    plot_error_distribution(val_labels, val_preds, "Validation Error Distribution "+best_model_name, working_dir+'/results/')
    plot_error_distribution(test_labels, test_preds, "Test Error Distribution "+best_model_name, working_dir+'/results/')

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