import os
import random
import shutil
from skimage import io
from skimage.color import rgb2gray
import pandas as pd


# Path to the dataset
dataset_path = "/Users/solozobovavaleria/dataset/14212/1"
output_path = "dataset/subset_images_1_14212"

# Get all image filenames
all_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.png')]


num_images_to_select = min(50, len(all_images))
# Randomly select 50 images
selected_images = random.sample(all_images, num_images_to_select)

# Copy selected images to a new folder
os.makedirs(output_path, exist_ok=True)
for img in selected_images:
    shutil.copy(img, output_path)


 # Set the path to your image folder
#image_folder_0 = "dataset/subset_images_0"

# Get the list of image files in the folder
#image_files = [os.path.join(image_folder_1, f) for f in os.listdir(image_folder_1) if f.endswith(".png")]

#print(image_files)  # Check the list of files   


