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


# Set the path to your image folder
#!!!!! Very important: check the correct directory
image_folder_0 = "dataset/set_images_0"

# Get the list of image files in the folder
image_files = [os.path.join(image_folder_0, f) for f in os.listdir(image_folder_0) if f.endswith(".png")]

# Function to extract LBP features
def extract_lbp_features(image, radius=3, n_points=24):
    # Compute the Local Binary Pattern
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    # Generate the histogram
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist / lbp_hist.sum()  # Normalize the histogram
    
    return lbp_hist.tolist()


# Function to extract GLCM features
def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    # Compute the grey-level co-occurrence matrix
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    
    # Extract GLCM properties
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    
    return [contrast, correlation, energy, homogeneity]


# Example of combining LBP and GLCM features
def extract_features(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:  # Check if the image is RGB
        image = rgb2gray(image)
    
    # Ensure image is of integer type for GLCM
    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image

    # Extract GLCM features
    glcm_features = extract_glcm_features(image)
    
    # Extract LBP features
    lbp_features = extract_lbp_features(image)
    
    # Combine both feature sets
    return glcm_features + lbp_features


# Main loop to process images and extract features
features_list = []  # Create an empty list to store features
image_names = []

for image_path in image_files:
    try:
        # Read the image
        img = imread(image_path)
        img_gray = rgb2gray(img)  # Convert to grayscale
        
        # Apply segmentation (e.g., thresholding)
        img_threshold = img_gray > 0.7  # Threshold at 50%
        
        # Extract features
        features = extract_features(img_gray)
        
        # Add features to the list
        features_list.append(features)
        
        # Add the image name to the list
        image_names.append(os.path.basename(image_path))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Define feature names
if features_list:  # Check if features_list is not empty
    feature_names = ['GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy', 'GLCM_Homogeneity'] + [f'LBP_{i}' for i in range(len(features_list[0]) - 4)]

    # Convert features_list to a DataFrame
    features_df = pd.DataFrame(features_list, columns=feature_names, index=image_names)

    # Drop columns where the variance is zero (constant features)
    features_df = features_df.loc[:, features_df.var() != 0]
    
    # Save the DataFrame to a CSV file
    #!!!!! Very importnat - correct name of the file !!!!!!!!!!!!!!!!
    features_df.to_csv("dataset/graycomatrix_image_analysis_class_0_set.csv", index=True)
else:
    print("No features were extracted.")