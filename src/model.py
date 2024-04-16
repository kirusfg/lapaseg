import code
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from upload import load_data


def resize_image(image, height=512):
    ratio = height / image.shape[0]
    width = int(image.shape[1] * ratio)
    resized_image = cv2.resize(image, (width, height))
    return resized_image


def adjust_landmarks(landmarks, width_ratio, height_ratio):
    adjusted_landmarks = []
    for i in range(len(landmarks)):
        adjusted_landmarks = (
            landmarks[i][0] * width_ratio[i],
            landmarks[i][1] * height_ratio[i],
        )
    return adjusted_landmarks


def extract_patches(image, patch_size):
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    patches = []
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):
        for j in range(0, image.shape[1] - patch_size + 1, patch_size):
            patch = image[i : i + patch_size, j : j + patch_size]
            # Ensure the patch is exactly the right size (it should always be if images are correctly dimensioned)
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                center_x = i + patch_size // 2
                center_y = j + patch_size // 2
                patches.append((patch, (center_x, center_y)))
    return patches


def calculate_hsv_histograms(patch, bins):
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_patch], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv_patch], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_patch], [2], None, [bins], [0, 256]).flatten()
    return np.concatenate([hist_h, hist_s, hist_v])


def calculate_hog_features(patch):
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray_patch,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True,
    )
    return hog_features


def relative_location(center, dimensions):
    return np.array([center[0] / dimensions[0], center[1] / dimensions[1]])


def landmarks_distance(location, landmarks):
    distance = np.array(landmarks - location)
    return distance


def feature_vector(patch, bins, image_dims, landmarks_array):
    color_features = calculate_hsv_histograms(patch, bins)
    shape_features = calculate_hog_features(patch)
    spatial_features = relative_location(
        (patch.shape[0] // 2, patch.shape[1] // 2), image_dims
    )
    distance_features = landmarks_distance(spatial_features, landmarks_array)
    color_features = color_features.flatten()
    shape_features = shape_features.flatten()
    spatial_features = spatial_features.flatten()
    distance_features = distance_features.flatten()

    return np.concatenate(
        (color_features, shape_features, spatial_features, distance_features)
    )


def get_resize_ratios(images, target_height=512):
    width_ratios = []
    height_ratios = []
    for image in images:
        original_height, original_width = image.shape[:2]
        height_ratio = target_height / float(original_height)
        width_ratio = (target_height / float(original_height)) * (
            original_width / float(original_height)
        )
        width_ratios.append(width_ratio)
        height_ratios.append(height_ratio)
    return height_ratios, width_ratios


bins = 16
images, labels, landmarks = load_data("train")

resized_image = [resize_image(img) for img in images]
resized_labels = [resize_image(lbl) for lbl in labels]
height_ratios, width_ratios = get_resize_ratios(images)
resized_landmarks = adjust_landmarks(landmarks, height_ratios, width_ratios)

print((np.array(resized_image).shape))
print((np.array(resized_labels).shape))
print((np.array(resized_landmarks).shape))
patches_list_locations = [
    extract_patches(resized_img, 32) for resized_img in resized_image
]  # 32x32 patches

patches_list = []
for patches in patches_list_locations[:1]:
    for i in patches:
        patches_list.append(i[0])


feature_vectors = []
for patches in tqdm(patches_list):
    print(patches.shape)
    feature_vectors.append(
        feature_vector(patches, bins, resized_image.shape[:2], resized_landmarks)
    )


print(feature_vectors[:5])

# code.interact(local=locals())

# X_train, X_test, y_train, y_test = train_test_split(
#     feature_vectors, resized_labels, test_size=0.2, random_state=42
# )
# print("The  model is going to train")
# model = LogisticRegression(
#     max_iter=1000, solver="lbfgs", multi_class="multinomial", n_jobs=32
# )
# model.fit(X_train, y_train)

# print("The  model is going to test")
# y_pred = model.predict(X_test)

# # Evaluation
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))
