import pandas as pd
from pathlib import Path
import numpy as np
import os
from torch.utils.data import Dataset, Subset
from skimage.transform import resize
import torchvision.transforms as T
import torch
import re
import matplotlib.pyplot as plt
import time

class ProcasLoader(Dataset):
    def __init__(self, csf, view_form, replicate, priors = False):
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
        Method to create a list of datapoints from a given folder. Automatically called during init. Currently returns, the paths, views and labels FIX LABELS.
        """
        data_dirs = os.listdir(Path(self.root, 'proc_data'))
        mamm_list      = []
        label_list     = []
        view_list      = []
        woman_list     = []
        for patient in data_dirs:
            procas_id = int(patient[-5:])
            mamm_dirs = os.listdir(Path(self.root, 'proc_data', patient))
            for mamm in mamm_dirs:
                match = re.search(r'-(LCC|LMLO|RMLO|RCC)', mamm)
                if match:
                    view = match.group(1)
                else:
                    continue
                if self.view_form in view:
                    mamm_list.append(Path(self.root, 'proc_data', patient, mamm))
                    view_list.append(view)
                    woman_list.append(patient)
                    label_list.append(self.data[self.data['ASSURE_PROCESSED_ANON_ID']==procas_id][view].item())
        return mamm_list, view_list, label_list, woman_list, len(set(woman_list))   

    def check_image(self, image):
        """
        This is just to check the image processing steps make sense.
        """
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()           
    
    def preprocess_image(self, image_path, view):
        """
        Load and preprocess the given image. Note that GEO originally had code to also work on different pixel sizes but we didnt need to do that.
        """
        ## Read the dicom and fetch the pixel array
        image = np.load(image_path)

        ## Flip any right images so they are all facing the same way
        if view == 'RCC' or view == 'RMLO':
            image = np.fliplr(image)

        image = image/np.amax(image)
        
        if self.replicate:
            image = np.stack((image, image, image), 0)
            
            ## We normalise to 0 mean, 1 standard deviation. 
            if self.view_form == 'CC':
                norm = T.Normalize((0.4366, 0.4366, 0.4366),(0.2375, 0.2375, 0.2375))
            elif self.view_form == 'MLO':
                norm = T.Normalize((0.4462, 0.4462, 0.4462),(0.2467, 0.2467, 0.2467))
            image = torch.as_tensor(image.copy())
            image = norm(image)
        else:
            image = torch.as_tensor(image.copy()).unsqueeze(0)
            norm = T.Normalize((0.567189508177),(0.20914499727207))
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
    dataset = ProcasLoader(csf = False, view_form = 'CC', replicate = False)
    
    img, label, name = dataset.__getitem__(13)
    #print(dataset.mamms)
