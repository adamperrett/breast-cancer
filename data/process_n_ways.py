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
reference_csv = 'PROCAS_reference.csv'

mosaic_data = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')
mosaic_ids = mosaic_data['ProcID']
vas_density_data = mosaic_data['VASCombinedAvDensity']

reference_data = pd.read_csv(os.path.join(csv_directory, reference_csv), sep=',')
refernce_ids = reference_data['ProcID']
raw_ids = reference_data['ASSURE_RAW_ID']
raw_PROC_id_dict = {}
for ref, raw in zip(refernce_ids, raw_ids):
    if not np.isnan(raw):
        raw_PROC_id_dict[ref] = int(raw)

processed = False
processed_dataset_path = os.path.join(csv_directory, 'mosaics_processed/full_mosaic_dataset.pth')
image_statistics_pre = []
image_statistics_post = []
no_study_type = []
bad_study_type = []
process_types = ['log', 'histo', 'clahe']

best_model_name = 'rand_balanced_two_ResnetTransformer'

# save mosaic_ids which have a vas score and Study_type was Breast Screening
id_vas_dict = {}
for id, vas in zip(mosaic_ids, vas_density_data):
    if not np.isnan(vas) and vas > 0 and id in raw_PROC_id_dict:
        raw_id = raw_PROC_id_dict[id]
        id_vas_dict["PROCAS_ALL_{:05}".format(raw_id)] = vas

mosaic_ids = mosaic_ids.unique()

def pre_process_mammograms_n_ways(mammographic_images, sides, heights, widths, image_types, process_type='standard'):
    '''
    :param process_type:
        - standard = original version
        - global = rescaled based on global maximum
        - histo = histogram equalisation
        - clahe = Contrast Limited Adaptive Histogram Equalization
    '''
    processed_images = []
    print("Beginning processing", process_type, "images")
    # print(heights, widths)
    for idx, mammographic_image in enumerate(tqdm(mammographic_images)):
        # Extract parameters for each image
        side = sides[idx]
        height = heights[idx]
        width = widths[idx]
        image_type = image_types[idx]

        # Reshape and preprocess
        if side == 'R':
            mammographic_image = np.fliplr(mammographic_image)
        if image_type == 'raw':
            mammographic_image = np.log(mammographic_image+1)
            mammographic_image = np.amax(mammographic_image) - mammographic_image
            cut_off = mammographic_image > filters.threshold_otsu(mammographic_image)
            cut_off = cut_off.astype(float)
            mammographic_image = cut_off * mammographic_image
            if process_type == 'histo':
                mammographic_image = exposure.equalize_hist(mammographic_image, mask=cut_off)
            elif process_type == 'clahe':
                mammographic_image = 2.0 * (mammographic_image - np.min(mammographic_image)) / np.ptp(mammographic_image) - 1
                mammographic_image = exposure.equalize_adapthist(mammographic_image, clip_limit=0.03)
        padded_image = np.zeros((max(2995, mammographic_image.shape[0]), max(2394, mammographic_image.shape[1])))
        padded_image[:mammographic_image.shape[0], :mammographic_image.shape[1]] = mammographic_image
        mammographic_image = resize(padded_image[:2995, :2394], (10 * 64, 8 * 64))
        mammographic_image = mammographic_image / np.amax(mammographic_image)
        processed_images.append(mammographic_image)
    return torch.stack([torch.from_numpy(img).float() for img in processed_images], dim=0)


def pre_process_mammograms(mammographic_images, sides, heights, widths, pixel_sizes, image_types, process_type='standard'):
    '''
    :param process_type:
        - standard = original version
        - global = rescaled based on global maximum
        - histo = histogram equalisation
        - clahe = Contrast Limited Adaptive Histogram Equalization
    '''
    target_pixel_size = 0.0941
    processed_images = []
    print("Beginning processing", process_type, "images")
    # print(heights, widths)
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
        if 'standard' in process_type:
            max_intensity = np.amax(mammographic_image)
        else:
            max_intensity = 2**14
        mammographic_image = resize(mammographic_image, (new_height, new_width))
        mammographic_image = mammographic_image * max_intensity / np.amax(mammographic_image)
        if side == 'R':
            mammographic_image = np.fliplr(mammographic_image)
        if image_type == 'raw':
            if process_type == 'histo':
                mammographic_image = exposure.equalize_hist(mammographic_image)
            elif process_type == 'clahe':
                mammographic_image = 2.0 * (mammographic_image - np.min(mammographic_image)) / np.ptp(mammographic_image) - 1
                mammographic_image = exposure.equalize_adapthist(mammographic_image, clip_limit=0.03)
            else:
                mammographic_image = np.log(mammographic_image+1)
            mammographic_image = np.amax(mammographic_image) - mammographic_image
        cut_off = mammographic_image > filters.threshold_otsu(mammographic_image)
        cut_off = cut_off.astype(float)
        mammographic_image = cut_off * mammographic_image
        padded_image = np.zeros((max(2995, mammographic_image.shape[0]), max(2394, mammographic_image.shape[1])))
        padded_image[:mammographic_image.shape[0], :mammographic_image.shape[1]] = mammographic_image
        mammographic_image = resize(padded_image[:2995, :2394], (10 * 64, 8 * 64))
        mammographic_image = mammographic_image / np.amax(mammographic_image)
        processed_images.append(mammographic_image)
    return torch.stack([torch.from_numpy(img).float() for img in processed_images], dim=0)

# This function will preprocess and zip all images and return a dataset ready for saving
def preprocess_and_zip_all_images(parent_directory, id_vas_dict):
    dataset_entries = {p_t: [] for p_t in process_types}

    patient_dirs = [d for d in os.listdir(parent_directory) if d in id_vas_dict]
    patient_dirs.sort()  # Ensuring a deterministic order

    for patient_dir in tqdm(patient_dirs):
        patient_path = os.path.join(parent_directory, patient_dir)
        image_files = [f for f in os.listdir(patient_path) if f.endswith('.dcm')]

        # Load all images for the given patient/directory
        dcm_files = [pydicom.dcmread(os.path.join(patient_path, f), force=True) for f in image_files]
        if not all([hasattr(dcm, 'StudyDescription') for dcm in dcm_files]):
            print("StudyDescription attribute missing")
            no_study_type.append([patient_dir, dcm_files])
        else:
            studies = [dcm.StudyDescription for dcm in dcm_files]
            if any(s != 'Breast Screening' and s != 'Breast screening' and
                   s != 'XR MAMMOGRAM BILATERAL' and s != 'MAMMO 1 VIEW RT' and
                   s != 'BILATERAL MAMMOGRAMS 2 VIEWS' for s in studies):
                print("Skipped because not all breast screening - studies =", studies)
                bad_study_type.append([patient_dir, studies, dcm_files])
                continue
        all_images = [dcm.pixel_array for dcm in dcm_files]
        all_sides = ['L' if 'LCC' in f or 'LMLO' in f else 'R' for f in image_files]
        all_heights = [img.shape[0] for img in all_images]
        if any(num > 4000 for num in all_heights):
            continue
        all_widths = [img.shape[1] for img in all_images]
        if any(num > 4000 for num in all_widths):
            continue
        all_image_types = ['raw' if ('raw' in patient_path or 'RAW' in patient_path or '_PROC_' not in patient_path)
                           else 'processed' for _ in image_files]

        for process_type in process_types:
            preprocessed_images = pre_process_mammograms_n_ways(deepcopy(all_images), all_sides, all_heights, all_widths,
                                                         all_image_types, process_type=process_type)

            dataset_entries[process_type].append((preprocessed_images, id_vas_dict[patient_dir]))

    return dataset_entries

if not processed:
    # Generate the dataset and save it
    if not os.path.exists(processed_dataset_path):
        dataset_entries = preprocess_and_zip_all_images(image_directory, id_vas_dict)
        for process_type in dataset_entries:
            torch.save(dataset_entries[process_type], processed_dataset_path[:-4]+'_'+process_type+'.pth')

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
process_type = 'clahe'
dataset = MammogramDataset(processed_dataset_path[:-4]+'_'+process_type+'.pth')

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
train_dataset.dataset = MammogramDataset(processed_dataset_path, transform=data_transforms, weights=sample_weights)

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

