import pandas as pd
import pydicom
import numpy as np
import scipy.misc
from skimage.transform import resize
import matplotlib.pyplot as plt
import os


def read_data_paths(data_file, view):
    """
    Reads the data paths from the paths file filtered by mammographic view.

    Parameters
    ----------
    data_file: string
        path of the data file
    view: string
        mammographic view

    Returns
    -------
    image_paths: list of string
        list of the image paths for the specific view 
    labels: list of float
        list of the VAS labels as assessed by human readers
    client_ids: list of int
        list of client ids (e.g. assure id) to which the images belong
    sides: list of char
        list of the breast side of each image (e.g. 'L' or 'R')
    """

    # Read the data file
    df = pd.read_csv(data_file, sep=',')
    
    # Filter data by view
    df = df[df['view'] == view]

    # Parse and return the paths and labels for the correct view
    client_ids = df['client_id'].tolist()
    labels = df['label'].tolist()
    image_paths = df['image_path'].tolist()
    sides = df['side'].tolist()
    
    return image_paths, labels, client_ids, sides


def read_data_batch(data_folder, data_paths, labels, client_ids, sides, image_type):
    """
    Reads and returns a batch of pre-processed data and its meta-data. 

    Parameters
    ----------
    data_folder: string
        path to the mammographic images folder
    data_paths: list of string
        list of images paths in the batch
    labels: list of float
        list of VAS labels in the batch
    client_ids: list of int
        list of client ids in the batch
    sides: list of char
        list of breast sides of the images in the batch
    image_type: string
        type of mammographic image     

    Returns
    -------
    data_batch: numpy array
        batch of pre-processed images
    labels_batch: numpy array
        batch of labels
    client_ids_batch: numpy array
        batch of client ids
    sides_batch: numpy array
        batch of sides
    """
    data_batch = []
    labels_batch = []
    client_ids_batch = []
    sides_batch = []
    

    for i in range(len(labels)):
        image_path = os.path.join(*[data_folder, data_paths[i]])
        
        try:
            # Build the image path
            image_path = os.path.join(*[data_folder, data_paths[i]])
            #print(image_path)
            # Read the DICOM file  
            current_mammogram = pydicom.read_file(image_path) 
            #print(current_mammogram)
            # Extract the image from the DICOM file
            print(image_path+'.npy')
            mammographic_image = np.load(image_path+'.npy').astype(np.uint16)
            # mammographic_image = np.fromstring(current_mammogram.PixelData, dtype=np.uint16)   
            
            # Define meta-parameters
            side = sides[i]
            height = current_mammogram.Rows
            width = current_mammogram.Columns
            pixel_size = current_mammogram.ImagerPixelSpacing
            
            # Preprocess image
            mammographic_image = pre_process_mammogram(mammographic_image, side, height, width, pixel_size, image_type)
            #print(mammographic_image)
            # Add data to the batch
            data_batch.append(mammographic_image.flatten())
            labels_batch.append(labels[i])
            client_ids_batch.append(client_ids[i])
            sides_batch.append(sides[i])
        except:
            print("An exception occurred")


    # Make lists as arrays
    data_batch = np.asarray(data_batch)
    labels_batch = np.asarray(labels_batch).reshape(-1,1)  
    client_ids_batch = np.asarray(client_ids_batch).reshape(-1,1)  
    sides_batch = np.asarray(sides_batch).reshape(-1,1)  

    return data_batch, labels_batch, client_ids_batch, sides_batch



def pre_process_mammogram(mammographic_image, side, height, width, pixel_size, image_type):
    """
    Preprocesses images to match those on which the models were trained.

    Parameters
    ----------
    mammographic_image: numpy array
        image extracted from the dicom file
    side: string
        breast side (e.g. 'L' or 'R')
    height: int
        height of the mammographic image
    width: int
        width of the mammographic image
    pixel_size: float
        pixel size of the dicom image
    image_type: string
        type of the mammographic image (i.e. raw or procesed)

    Returns
    -------
    mammographic_image: numpy array NXM
        pre-processed mamographic image
    """
    
    # Reshape image array to the 2D shape 
    mammographic_image = np.reshape(mammographic_image,(height,width))
    
    # DO NOT CHANGE THIS!!!! 
    target_pixel_size = 0.0941 # All models have been trained on images with this pixel size

    new_height = int(np.ceil(mammographic_image.shape[0] * pixel_size[0]/target_pixel_size))
    new_width = int(np.ceil(mammographic_image.shape[1] * pixel_size[1]/target_pixel_size))

    max_intensity = np.amax(mammographic_image)
    mammographic_image = resize(mammographic_image, (new_height,new_width))    
    
    # Rescale intensity values to their original range
    mammographic_image = mammographic_image * max_intensity/np.amax(mammographic_image)

    # FLIP IMAGE!
    if side == 'R':
        mammographic_image = np.fliplr(mammographic_image)

       
    if image_type == 'raw':
        # Apply log transform and inverse pixel intensities
        mammographic_image = np.log(mammographic_image)
        mammographic_image = np.amax(mammographic_image) - mammographic_image
    
    # Pad images
    padded_image = np.zeros((np.amax([2995, mammographic_image.shape[0]]), np.amax([2394,mammographic_image.shape[1]])))
    padded_image[0:mammographic_image.shape[0],0:mammographic_image.shape[1]] = mammographic_image[:,:]
    mammographic_image = padded_image[0:2995, 0:2394]
    
    # Resize to 640X512
    mammographic_image = resize(mammographic_image, (10*64,8*64))
    
    # Rescale intensities to [0, 1]
    mammographic_image = mammographic_image/np.amax(mammographic_image)
    
    return  mammographic_image   