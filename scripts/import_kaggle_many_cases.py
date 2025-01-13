import os
import random
import shutil
from skimage import io
from skimage.color import rgb2gray
import pandas as pd


# Path to the dataset
dataset_path_1= "/Users/solozobovavaleria/dataset/8950/0"
dataset_path_2= "/Users/solozobovavaleria/dataset/12749/0"
dataset_path_3= "/Users/solozobovavaleria/dataset/13462/0"
dataset_path_4= "/Users/solozobovavaleria/dataset/10285/0"
dataset_path_5= "/Users/solozobovavaleria/dataset/12947/0"
dataset_path_6= "/Users/solozobovavaleria/dataset/14155/0"
dataset_path_7= "/Users/solozobovavaleria/dataset/13693/0"
dataset_path_8= "/Users/solozobovavaleria/dataset/14305/0"
dataset_path_9= "/Users/solozobovavaleria/dataset/9044/0"
dataset_path_10= "/Users/solozobovavaleria/dataset/12894/0"

output_path = "dataset/set_images_0"
list_dataset=[dataset_path_1, dataset_path_2, dataset_path_3,dataset_path_4,dataset_path_5,dataset_path_6,dataset_path_7, dataset_path_8, dataset_path_9, dataset_path_10]
# Get all image filenames

all_images = []
for dataset_path in list_dataset:
    all_images.extend([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.png')])

num_images_to_select = min(500, len(all_images))
# Randomly select 50 images
selected_images = random.sample(all_images, num_images_to_select)

# Copy selected images to a new folder
os.makedirs(output_path, exist_ok=True)

for img in selected_images:
    shutil.copy(img, output_path)



print(f"Successfully copied {len(selected_images)} images to {output_path}.")

# Get the list of image files in the folder
#image_files = [os.path.join(image_folder_1, f) for f in os.listdir(image_folder_1) if f.endswith(".png")]

#print(image_files)  # Check the list of files   


