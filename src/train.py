import code
import sys
from upload import load_data
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def extract_features(image, patch_size=32):
    """
    Extract features using color histograms and HOG features from patches of the image.
    Returns the array of features and the corresponding labels for each patch.
    """
    img_h, img_w = image.shape[:2]
    features = []
    labels = [] 

    for y in range(0, img_h - patch_size + 1, patch_size):
        for x in range(0, img_w - patch_size + 1, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            # print(patch.shape)
            # Color histogram in HSV space
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_patch], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
            # print("Hist:", hist.shape)

            hist = cv2.normalize(hist, hist).flatten()
            # HOG features
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            print("gray pathch shape: ", gray_patch.shape)
            hog_features = hog(gray_patch, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=False, feature_vector=True)
            # print("HOG:", hog_features.shape)
            feature_vector = np.hstack([hist, hog_features])
            features.append(feature_vector)

            

    return np.array(features)

def resize_image(image, height=512):
    ratio = height / image.shape[0]
    width = int(image.shape[1] * ratio)
    resized_image = cv2.resize(image, (width, height))
    return resized_image

train_images, train_labels = load_data('train')

resized_imgs = [resize_image(img) for img in tqdm(train_images)]
resized_lbls = [resize_image(lbl) for lbl in tqdm(train_labels)]

features = [extract_features(image) for image in tqdm(resized_imgs)]
sys.exit(0)
code.interact(local=locals())

X_train, X_test, y_train, y_test = train_test_split(features, train_labels, test_size=0.2, random_state=42)

# Train CRF and predict on test set
# Assume crf_training_function is implemented
print("The  model is going to train")
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', n_jobs=32)
model.fit(X_train, y_train)

print("The  model is going to test")
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
