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

    all_dirs = os.listdir(src_dir)
    mosaic_dirs = []
    for dir in all_dirs:
        if 'Densitas' in dir:
            for sub_dir in os.listdir(os.path.join(src_dir, dir)):
                mosaic_dirs.append([os.path.join(src_dir, dir), sub_dir])
        else:
            mosaic_dirs.append([src_dir, dir])

    # Loop through each folder in the source directory
    for folder_path, folder_name in tqdm(mosaic_dirs):
        folder_path = os.path.join(folder_path, folder_name)

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




look_for_sub = ['7e', '10f']
for i in range(8, 11):
    look_for_sub.append('{}a'.format(i))
    look_for_sub.append('{}b'.format(i))
    look_for_sub.append('{}c'.format(i))
    look_for_sub.append('{}d'.format(i))
    look_for_sub.append('{}e'.format(i))

# Usage
src_directory = "Z:/PROCAS_ALL_PROCESSED"
dest_directory = "D:/mosaic_data/processed"
copy_folders_with_min_images(src_directory, dest_directory)