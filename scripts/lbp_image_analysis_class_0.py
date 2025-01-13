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
import mahotas
from skimage.io import imread


 # Set the path to your image folder
image_folder_0 = "dataset/subset_images_0_10277"

# Get the list of image files in the folder
image_files = [os.path.join(image_folder_0, f) for f in os.listdir(image_folder_0) if f.endswith(".png")]

#print(image_files)  # Check the list of files   

#Function to extract morphological features:
def extract_morphological_features(segmented_img):
    labeled_img = label(segmented_img)  # Label the connected regions
    regions = measure.regionprops(labeled_img)  # Get properties for each region
    
    features = []
    
    #for region in regions:
    for idx, region in enumerate(regions, start=1):     
        # Area
        area = region.area
        
        # Perimeter
        perimeter = region.perimeter
        
        # Eccentricity
        eccentricity = region.eccentricity
        
        # Major and minor axes length
        major_axis_length = region.major_axis_length
        minor_axis_length = region.minor_axis_length
        
        features.extend([area, perimeter, eccentricity, major_axis_length, minor_axis_length])
    
    # If no region was found, we set a default value to prevent errors
    if len(features) == 0:
        features = [0] * 5  # Adjust the length based on the number of features you extract
    
    return features

# Function to extract texture features (e.g., using Local Binary Patterns)
def extract_lbp_features(img_gray):
    # Apply Local Binary Pattern (LBP)
    radius = 1  # Radius of the neighborhood
    n_points = 8 * radius  # Number of points in the LBP pattern
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    
    # Calculate the histogram of LBP values
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    # Normalize the histogram to have values between 0 and 1
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Avoid division by zero
    
    return lbp_hist

# Function to extract texture features using GLCM (using mahotas)
def extract_glcm_features(img_gray):
    # Compute the GLCM using mahotas
    img_uint8 = (img_gray * 255).astype(np.uint8)  # Convert grayscale to uint8 format
    glcm = mahotas.features.haralick(img_uint8)  # Extract the Haralick features (GLCM)
    
    # Extract specific features from the Haralick matrix
    contrast = glcm[:, 0].mean()  # Contrast is in the first column
    correlation = glcm[:, 1].mean()  # Correlation is in the second column
    energy = glcm[:, 4].mean()  # Energy is in the fifth column
    homogeneity = glcm[:, 3].mean()  # Homogeneity is in the fourth column
    
    return [contrast, correlation, energy, homogeneity]

# Ensure equal length for all feature arrays
def ensure_equal_length(features_array, target_length):
    current_length = len(features_array)
    if current_length < target_length:
        features_array = np.pad(features_array, (0, target_length - current_length), 'constant')
    elif current_length > target_length:
        features_array = features_array[:target_length]
    return features_array

# Set target length (choose the largest length in your case)
target_length = 200  # Choose based on the largest feature set length



# Main loop to process images and extract features
features_list = []  # Create an empty list to store features
image_names = []

for image_path in image_files:
    # Read the image
    img = imread(image_path)
    img_gray = rgb2gray(img)  # Convert to grayscale
    
    # Apply segmentation (e.g., thresholding)
    img_threshold = img_gray > 0.5  # Threshold at 50%
    
    # Extract texture features using GLCM
    glcm_features = extract_glcm_features(img_gray)
    print(len(glcm_features))
    glcm_features = ensure_equal_length(glcm_features, target_length)
    
    # Extract morphological features
    morphological_features = extract_morphological_features(img_threshold)
    print(len(morphological_features))
    morphological_features = ensure_equal_length(morphological_features, target_length)
    
    # Extract texture features using LBP
    lbp_features = extract_lbp_features(img_gray)
    print(len(lbp_features))
    lbp_features = ensure_equal_length(lbp_features, target_length)
    
    
    # Combine all features into a single list
    features = np.concatenate([glcm_features, morphological_features, lbp_features])
    
    # Store the features for this image
    features_list.append(features)

    # Extract the image name (e.g., 'image1.png' from 'path/to/image1.png')
    image_name = os.path.basename(image_path)
    image_names.append(image_name)


# Dynamically generate column names based on the total number of features
#num_features = target_length * 3  # 3 feature sets of length 'target_length'
# Dynamically generate column names based on feature type
#morphological_features = ["Morphological_" + str(i) for i in range(target_length)]
morphological_names = []
for idx in range(1, len(morphological_features) // 5 + 1):
    morphological_names.extend([
        f"morphological_{idx}_area",
        f"morphological_{idx}_perimeter",
        f"morphological_{idx}_eccentricity",
        f"morphological_{idx}_major_axis_length",
        f"morphological_{idx}_minor_axis_length"
    ])

GLCM_names = []
for idx in range(1, len(glcm_features) // 4 + 1):
    GLCM_names.extend([
        f"GLCM_{idx}_Contrast",
        f"GLCM_{idx}_Correlation",
        f"GLCM_{idx}_Energy",
        f"GLCM_{idx}_Homogeneity",
    ])   

# This will give you the list of feature names, which you can use to create a DataFrame
#print(morphological_names)

GLCM_features = GLCM_names

morphological_features=morphological_names

lbp_features = ["LBP_" + str(i) for i in range(target_length)]
#glcm_features = ["GLCM_" + str(i) for i in range(target_length)]


# Combine all feature names
columns = GLCM_features + morphological_features + lbp_features 

# Convert features_list to a DataFrame and save as CSV
features_df = pd.DataFrame(features_list, columns=columns, index=image_names)

# Drop columns where the variance is zero (constant features)
features_df = features_df.loc[:, features_df.var() != 0]
# Save the DataFrame to a CSV file
features_df.to_csv("dataset/features_subset_0_10277_lbp.csv", index=True)

