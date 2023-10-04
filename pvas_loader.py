from pathlib import Path
import pandas as pd
import numpy as np
import pydicom
import os

from skimage.transform import resize
import torchvision.transforms as T
import torch
import re
import matplotlib.pyplot as plt

# Debugging
from torch.utils.data import Dataset, Subset
import time


class ProcasLoader(Dataset):
    def __init__(self, csf, view_form, replicate, priors=False):
        self.csf = csf
        self.priors = priors
        if self.csf:
            self.root = '/mnt/iusers01/ct01/k16296sr/scratch/pVAS_sr'
        else:
            self.root = 'Z:/Code/pVAS_sr'
        self.replicate = replicate
        self.data = pd.read_csv(Path(self.root, 'pvas_data_sheet.csv'))
        self.view_form = view_form
        self.mamms, self.views, self.labels, self.woman_list, self.women_count = self._form_list()

    def _form_list(self):
        """
        Method to create a list of datapoints from a given folder. Automatically called during init.
        Currently returns, the paths, views and labels FIX LABELS.
        """
        data_dirs = os.listdir(Path(self.root, 'data'))
        mamm_list = []
        label_list = []
        view_list = []
        woman_list = []
        # Looping through the data folder
        for patient in data_dirs:
            # Records the patient ID and loops through the patient folder
            procas_id = int(patient[-5:])
            mamm_dirs = os.listdir(Path(self.root, 'data', patient))
            for mamm in mamm_dirs:
                # Searches for view (may be outside of these)
                match = re.search(r'-(LCC|LMLO|RMLO|RCC)', mamm)
                if match:
                    view = match.group(1)
                else:
                    continue
                if self.view_form in view:
                    # Record the image, view, source patient and the corresponding label (VAS)
                    mamm_list.append(Path(self.root, 'data', patient, mamm))
                    view_list.append(view)
                    woman_list.append(patient)
                    # Matches against the data table, GT taken at view level rather than avearge
                    label_list.append(self.data[self.data['ASSURE_PROCESSED_ANON_ID'] == procas_id][view].item())
        return mamm_list, view_list, label_list, woman_list, len(set(woman_list))

    def check_image(self, image):
        """
        This is just to check the image processing steps make sense.
        """
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()

    def preprocess_image(self, image_path, view):
        """
        Load and preprocess the given image. Note that GEO originally had code to also work on different pixel sizes
        but we didnt need to do that.
        """
        ## Read the dicom and fetch the pixel array
        image = pydicom.read_file(image_path).pixel_array

        ## Flip any right images so they are all facing the same way
        if view == 'RCC' or view == 'RMLO':
            image = np.fliplr(image)

        ## This step essentially resizes the smaller images by padding them with zeros. This preserves the pixel size between women whilst
        ## making sure all images have the same size.        
        padded_image = np.zeros((np.amax([2995, image.shape[0]]), np.amax([2394, image.shape[1]])))
        padded_image[0:image.shape[0], 0:image.shape[1]] = image[:, :]
        image = padded_image[0:2995, 0:2394]

        ## Resize to 640x512, not sure why this was chosen
        image = resize(image, (640, 512))

        ## Max min normalise
        image = image / np.amax(image)

        if self.replicate:
            image = np.stack((image, image, image), 0)

            ## We normalise to 0 mean, 1 standard deviation. 
            if self.view_form == 'CC':
                norm = T.Normalize((0.2283, 0.2283, 0.2283), (0.3108, 0.3108, 0.3108))
            elif self.view_form == 'MLO':
                norm = T.Normalize((0.2633, 0.2633, 0.2633), (0.3331, 0.3331, 0.3331))
            # norm = T.Normalize((0.56718950817, 0.56718950817 , 0.56718950817),(0.20914499727207, 0.20914499727207 , 0.20914499727207))
            image = torch.as_tensor(image.copy())
            image = norm(image)
        else:
            image = torch.as_tensor(image.copy()).unsqueeze(0)
            norm = T.Normalize((0.567189508177), (0.20914499727207))
            image = norm(image)
        return image

    def __len__(self):
        return len(self.mamms)

    def __getitem__(self, index):
        """
        Returns the array, labels and name
        """
        image = self.preprocess_image(self.mamms[index], self.views[index])
        return image, self.labels[index], self.mamms[index].stem


if __name__ == "__main__":
    ## Debugging
    dataset = ProcasLoader(csf=False, view_form='MLO', replicate=True)
    img, label, name = dataset.__getitem__(12)
