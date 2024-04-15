import cv2
import numpy as np
from skimage.feature import hog
from skimage import color, exposure, io
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
from upload import load_data


def resize_image(image, height=512):
    ratio = height / image.shape[0]
    width = int(image.shape[1] * ratio)
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def extract_patches(image, patch_size):
    patches = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append((patch, (i + patch_size // 2, j + patch_size // 2)))
    return patches

def calculate_hsv_histograms(patch, bins):
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_patch], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv_patch], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_patch], [2], None, [bins], [0, 256]).flatten()
    return np.concatenate([hist_h, hist_s, hist_v])

def calculate_hog_features(patch):
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_patch, orientations=8, pixels_per_cell=(patch.shape[0] // 2, patch.shape[1] // 2),
                       cells_per_block=(1, 1), visualize=False, feature_vector=True)
    return hog_features

def relative_location(center, dimensions):
    return np.array([center[0] / dimensions[0], center[1] / dimensions[1]])

def feature_vector(patch, bins, image_dims):
    color_features = calculate_hsv_histograms(patch, bins)
    shape_features = calculate_hog_features(patch)
    spatial_features = relative_location((patch.shape[0]//2, patch.shape[1]//2), image_dims)
    return np.concatenate([color_features, shape_features, spatial_features])

# Example usage
images, labels = load_data()
resized_image = resize_image(images[0])
patches = extract_patches(resized_image, 32)  # Example: 32x32 patches

bins = 32  # Number of bins in the histogram
feature_vectors = [feature_vector(patch, bins, resized_image.shape[:2]) for patch, center in patches]
print(feature_vectors[:5])
# Now feature_vectors contains the feature vector for each patch
