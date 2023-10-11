import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import pydicom
from skimage import filters
from PIL import Image


root = Path('Z:/Code/For ADAM')

## Loads the mosaics data and drop the implants (probably needs other exclusions)
data = pd.read_csv(Path(root, 'PROCAS_image_information.csv'))
data = data[data['Implant']=='NO']

## Group to find patients with more than 1 view
view_counts = data.groupby(['Patient', 'View']).size().reset_index(name='Count')

## Filter patients with repeating views (Count > 1)
patients_with_repeating_views = view_counts[view_counts['Count'] > 1]
result_df = data[data['Patient'].isin(patients_with_repeating_views['Patient'])]


## Finds the largest absolute difference in size and drops the rest, number is arbitrary for now.
result_df['AbsDiff'] = result_df.groupby(['Patient', 'View'])['Size'].transform(lambda x: abs(x - x.iloc[0]))
result_df = result_df[result_df['AbsDiff']>600000]
result_df['ASSURE_PROCESSED_ANON_ID'] = result_df['Patient'].str[-5:].astype(int) # for merging later
patients = list(result_df['Patient']) # list of suspected mosaics

## Loads the procas information csv and the image reference csv.
procas = pd.read_csv(Path(root, 'full_procas_info3.csv'))
ref    = pd.read_csv(Path(root, 'PROCAS_reference.csv'))
procas = procas.merge(ref, on = 'ProcID')

## Merge with existing clinical information. It looks like none are lost
print(f'Number of detected mosaics: {len(result_df)}')
result_df = pd.merge(result_df, procas, on = 'ASSURE_PROCESSED_ANON_ID', how = 'left')
print(f'Number of matched mosaics remaining: {len(result_df)}')



def image_histogram_equalization(image, number_bins=256):
    """
    Performs hist equalisation on an image. Useful to display images
    """
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    return image_equalized.reshape(image.shape), cdf

def display_image(image ,title = None):
    """
    Display image in console
    """
    plt.imshow(image, cmap = 'gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def process_proc(bag):
    """
    Processes the image for visual clarity. Can apply any processing code you want, I just think this looks the most clear.
    """
    
    bag=bag-np.amin(bag)
    bag=bag/np.amax(bag)
    
    val = filters.threshold_otsu(bag)
    bag = np.clip(bag, val,np.amax(bag))
    
    bag=image_histogram_equalization(bag)[0]
    display_image(bag)
    image = Image.fromarray(((bag - bag.min()) / (bag.max() - bag.min()) * 255.0).astype(np.uint8))
    return image



## OPTIONAL!!!
## What this does is goes through the assure PROC folder, selects only the images that are in the patients list (so suspected mosaics),
## and saves them as PNGs in another folder. This is a cheeky way to quickly check images by hand without having to open all the dicoms. 
root = 'Y:/PROCAS_ALL_PROCESSED'
assure_mamms = os.listdir(Path(root))
root_save = 'Z:/Datasets/PROCAS_large_jpgs2' ## Change this to whereever you want to save.
Path(root_save).mkdir(exist_ok=True)

counter = 0
for mamm in assure_mamms:
    if mamm[0:4] == 'Dens':
        deeper_mamms = os.listdir(Path(root, mamm))
        for deep_mamm in deeper_mamms:
            inner = os.listdir(Path(root, mamm, deep_mamm))
            if mamm in patients:
                print(deep_mamm)
                counter += 1
                Path(root_save, deep_mamm).mkdir(exist_ok=True)
                for img in inner:
                    dcm = pydicom.read_file(Path(root, mamm, deep_mamm, img)).pixel_array
                    dcm = process_proc(dcm)
                    dcm.save(Path(root_save, deep_mamm, img.replace('.dcm', '.png')))
                    
            else:
                continue
        continue
    if mamm in patients:
        print(mamm)
        counter += 1
        Path(root_save, mamm).mkdir(exist_ok=True)
        for img in os.listdir(Path(root, mamm)):
            dcm = pydicom.read_file(Path(root, mamm, img)).pixel_array
            dcm = process_proc(dcm)
            dcm.save(Path(root_save, mamm, img.replace('.dcm', '.png')))