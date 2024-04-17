import cv2
import os
import numpy as np
from tqdm import tqdm


DATASET_DIR = "LaPa"


def resize_image(image, height=512):
    ratio = height / image.shape[0]
    width = int(image.shape[1] * ratio)
    resized_image = cv2.resize(image, (width, height))
    return resized_image


def get_resize_ratios(image_dims: tuple[int, int, int], target_height=512):
    original_height, original_width, _ = image_dims
    height_ratio = target_height / float(original_height)
    width_ratio = (target_height / float(original_height)) * (
        original_width / float(original_height)
    )
    return height_ratio, width_ratio


def adjust_landmarks(landmarks: np.ndarray, ratio: tuple[int, int]):
    adjusted_landmarks = []
    for i in range(len(landmarks)):
        adjusted_landmarks.append(
            (
                landmarks[i][0] * ratio[0],
                landmarks[i][1] * ratio[1],
            )
        )
    return np.array(adjusted_landmarks)


def load_images_and_labels(image_paths, label_paths, landmark_paths):
    images = [cv2.imread(path) for path in tqdm(image_paths)]
    labels = [cv2.imread(path, 0) for path in tqdm(label_paths)]
    landmarks = []

    for i in range(len(images)):
        landmarks.append(np.loadtxt(landmark_paths[i], skiprows=1))

    return images, labels, landmarks


def load_data(mode, n=-1, resize: bool = True):
    path_to_imgs = os.path.join(DATASET_DIR, f"{mode}/images")
    path_to_labels = os.path.join(DATASET_DIR, f"{mode}/labels")
    path_to_landmarks = os.path.join(DATASET_DIR, f"{mode}/landmarks")

    imgs_list = os.listdir(path_to_imgs)
    labels_list = os.listdir(path_to_labels)
    landmarks_list = os.listdir(path_to_landmarks)

    image_paths = []
    label_paths = []
    landmark_paths = []

    for i in tqdm(range(len(imgs_list))):
        image_paths.append(os.path.join(path_to_imgs, imgs_list[i]))
        label_paths.append(os.path.join(path_to_labels, labels_list[i]))
        landmark_paths.append(os.path.join(path_to_landmarks, landmarks_list[i]))

    image_paths = sorted(image_paths)[1337 : 1337 + n]
    label_paths = sorted(label_paths)[1337 : 1337 + n]
    landmark_paths = sorted(landmark_paths)[1337 : 1337 + n]

    images, labels, landmarks = load_images_and_labels(
        image_paths, label_paths, landmark_paths
    )

    if resize:
        images = [resize_image(img) for img in images]
        labels = [resize_image(lbl) for lbl in labels]
        ratios = [get_resize_ratios(img.shape) for img in images]
        landmarks = [
            adjust_landmarks(landmark, ratio)
            for landmark, ratio in zip(landmarks, ratios)
        ]
        for image, label in zip(images, labels):
            assert image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1]

    return images, labels, landmarks


if __name__ == "__main__":
    images, labels, landmarks = load_data("train")
    # code.interact(local=locals())
