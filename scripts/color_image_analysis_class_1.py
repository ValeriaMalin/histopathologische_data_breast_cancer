import os
from skimage import io
from skimage.color import rgb2gray
import pandas as pd
import numpy as np
from skimage import io, measure, filters, feature
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import label
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
from skimage import io, measure
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
import os
import numpy as np
import pandas as pd
import cv2
from skimage import color, feature
from skimage.filters import gabor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm


# Set the path to your image folder
#!!!!!!!!!! Vety important - chech that the directory is correct !!!!!!!
image_folder = "dataset/subset_images_1"

# Get the list of image files in the folder
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]

def extract_color_features(image):
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Mean, StdDev for each channel
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])
    std_r = np.std(image[:, :, 0])
    std_g = np.std(image[:, :, 1])
    std_b = np.std(image[:, :, 2])

    # Flatten and normalize histogram
    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
    hist_r = hist_r / sum(hist_r)
    hist_g = hist_g / sum(hist_g)
    hist_b = hist_b / sum(hist_b)

    # Concatenate all features
    return [mean_r, mean_g, mean_b, std_r, std_g, std_b] + hist_r.tolist() + hist_g.tolist() + hist_b.tolist()

#Function to extract gabor features
def extract_gabor_features(image, frequencies=[0.6, 1.0], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    gabor_features = []
    for freq in frequencies:
        for angle in angles:
            filt_real, filt_imag = gabor(image, frequency=freq, theta=angle)
            gabor_features.append(filt_real.mean())
            gabor_features.append(filt_real.var())
    
    return gabor_features

#Function to define intensity:
def extract_intensity_features(image):
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    # Intensity statistics
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    skewness = ((image - mean_intensity)**3).mean()
    kurtosis = ((image - mean_intensity)**4).mean()
    
    return [mean_intensity, std_intensity, min_intensity, max_intensity, skewness, kurtosis]

# Example of combining LBP and GLCM features
def extract_all_features(image):

    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract features
    color_feats = extract_color_features(image)
    gabor_feats = extract_gabor_features(image)
    intensity_feats = extract_intensity_features(image)
    
    return color_feats + gabor_feats + intensity_feats

# Paths to images

image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]


# Debugging: Check feature lengths
for image_path in tqdm(image_files):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    color_feats = extract_color_features(image)
    print("Color features length:", len(color_feats))  # Should match the color-related columns in your DataFrame

    gabor_feats = extract_gabor_features(image)
    print("Gabor features length:", len(gabor_feats))  # Should match the Gabor-related columns

    intensity_feats = extract_intensity_features(image)
    print("Intensity features length:", len(intensity_feats))  # Should match the intensity-related columns

    features = color_feats + gabor_feats + intensity_feats
    print("Total features length:", len(features))  # Ensure this matches len(columns)
    break  # Debugging for one sample

columns = (
    ["image_name"] +  # Add a column for the image name
    ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"] + 
    [f"hist_r_{i}" for i in range(256)] + 
    [f"hist_g_{i}" for i in range(256)] + 
    [f"hist_b_{i}" for i in range(256)] +
    [f"gabor_{i}" for i in range(16)] +  # Adjusted for 16 Gabor features
    ["mean_intensity", "std_intensity", "min_intensity", "max_intensity", "skewness", "kurtosis"]
)


all_features = []
# Extract features for each image
for image_path in tqdm(image_files):
    features = extract_all_features(image_path)
    
    # Extract the file name from the image path
    image_name = os.path.basename(image_path)
    
    # Prepend the image name to the features
    features = [image_name] + features
    all_features.append(features)
    # Append to the DataFrame
   # features_df = features_df._append(pd.Series(features, index=columns), ignore_index=True)

features_df = pd.DataFrame(all_features, columns=columns)
# Save to CSV
#!!!!!!! Check the name of the file !!!!!!!
features_df.to_csv("dataset/graycomatrix_image_analysis_class_1_color.csv", index=False)
