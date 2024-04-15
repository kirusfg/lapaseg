import os 
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split






def extract_hsv_histogram(image, nbins):
    """Extract concatenated HSV histograms for the given patch."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = [cv2.calcHist([hsv_image], [i], None, [nbins], [0, 256 if i != 0 else 180]).flatten() 
            for i in range(3)]
    hist = np.concatenate(hist)
    return hist / hist.sum()  # Normalize histogram

def extract_hog_features(image, pixels_per_cell):
    """Extract HOG features for the given patch."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray_image, orientations=8, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                      cells_per_block=(1, 1), visualize=True, feature_vector=True)
    return features

def classify_patches(image, patch_size, classifier, nbins=32, pixels_per_cell=16):
    H, W = image.shape[:2]
    labels_image = np.zeros((H, W), dtype=np.uint8)  # Placeholder for the labeled image

    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            hsv_hist = extract_hsv_histogram(patch, nbins)
            hog_feat = extract_hog_features(patch, pixels_per_cell)
            features = np.concatenate([hsv_hist, hog_feat])
            
            # Predict the label for the patch
            label = classifier.predict([features])[0]
            
            # Assign the label to the center pixel of the patch
            center_y, center_x = y + patch_size // 2, x + patch_size // 2
            labels_image[center_y, center_x] = label

    return labels_image

def resize_image(image, height=512):
    ratio = height / image.shape[0]
    width = int(image.shape[1] * ratio)
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def img_to_numpy(img):
    img = cv2.imread(img)
    return img




# Example features and labels loading
# features, labels = load_your_data()  # You need to implement this

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train SVM classifier
classifier = svm.SVC(kernel='linear', C=1.0)
classifier.fit(X_train, y_train)

# Now use the classifier to label the patches
image = cv2.imread('path_to_your_test_image.jpg')
images = [img_to_numpy(image) for image in path_images]
images_resized = [resize_image(image) for image in images]
labeled_image = classify_patches(image, patch_size=32, classifier=classifier)

# You can save or display the labeled_image as needed
cv2.imwrite('labeled_image.png', labeled_image)

