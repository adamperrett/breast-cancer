import os
import shutil
from tqdm import tqdm


def copy_folders_with_min_images(src_dir, dest_dir, min_images=4):
    # Check if source directory exists
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist.")
        return

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Loop through each folder in the source directory
    for folder_name in tqdm(os.listdir(src_dir)):
        folder_path = os.path.join(src_dir, folder_name)

        # Ensure that it's indeed a directory
        if os.path.isdir(folder_path):
            # Count the number of images (files) in the folder
            num_images = len(os.listdir(folder_path))
            dest_folder_path = os.path.join(dest_dir, folder_name)  # for if it gets cancelled part way through
            # If the number of images is greater than the threshold and the file has not already been copied
            if num_images > min_images and not os.path.exists(dest_folder_path):
                dest_folder_path = os.path.join(dest_dir, folder_name)
                # Copy the entire folder to the destination directory
                shutil.copytree(folder_path, dest_folder_path)
                print(f"Copied {folder_name} to {dest_folder_path}")


# Usage
src_directory = "Z:/PROCAS_ALL_RAW"
dest_directory = "D:/mosaic_data/raw"
copy_folders_with_min_images(src_directory, dest_directory)