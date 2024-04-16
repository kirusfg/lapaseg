import cv2
import os
import numpy as np
import code
from tqdm import tqdm

dataset_dir = "/home/zhuldyz/Code/lapaseg/LaPa"


def load_images_and_labels(image_paths, label_paths, landmarks_path):
    images = [cv2.imread(path) for path in tqdm(image_paths)]
    labels = [cv2.imread(path, 0) for path in tqdm(label_paths)]
    landmarks = []

    for i in range(len(images)):
        landmarks.append(np.loadtxt(landmarks_path[i], skiprows=1))

    return images, labels, landmarks


def load_data(mode):
    path_to_imgs = os.path.join(dataset_dir, f"{mode}/images")
    path_to_labels = os.path.join(dataset_dir, f"{mode}/labels")
    path_to_landmarks = os.path.join(dataset_dir, f"{mode}/landmarks")

    imgs_list = os.listdir(path_to_imgs)
    labels_list = os.listdir(path_to_labels)
    landmarks_list = os.listdir(path_to_landmarks)

    path_images = []
    path_labels = []
    path_landmarks = []

    for i in tqdm(range(len(imgs_list))):
        path_images.append(os.path.join(path_to_imgs, imgs_list[i]))
        path_labels.append(os.path.join(path_to_labels, labels_list[i]))
        path_landmarks.append(os.path.join(path_to_landmarks, landmarks_list[i]))

    images = sorted(path_images)[:200]
    labels = sorted(path_labels)[:200]
    landmarks = sorted(path_landmarks)[:200]

    images, labels, landmarks = load_images_and_labels(images, labels, landmarks)
    # for i in range(10):
    #     print("Shape oof the image:", images[i].shape, "Shape of the label:", labels[i].shape)
    return images, labels, landmarks


images, labels, landmarks = load_data("train")

# code.interact(local=locals())
