import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import code
from tqdm import tqdm

def load_images_and_labels(image_paths, label_paths):
    images = [cv2.imread(path) for path in tqdm(image_paths)]
    labels = [cv2.imread(path, 0) for path in tqdm(label_paths)]  # Load as grayscale if labels are masks
    return images, labels

def load_data(mode):
    
    path_to_imgs = f"/raid/kirill_kirillov/585-cv/project/data/LaPa/{mode}/images"
    path_to_labels = f"/raid/kirill_kirillov/585-cv/project/data/LaPa/{mode}/labels"
    imgs_list = os.listdir(path_to_imgs)
    labels_list = os.listdir(path_to_labels)
    path_images = []
    path_labels = []

    for i in tqdm(range(len(imgs_list))):
        path_images.append(os.path.join(path_to_imgs, imgs_list[i]))
        path_labels.append(os.path.join(path_to_labels, labels_list[i]))
    
    images = sorted(path_images)[:200]
    labels = sorted(path_labels)[:200]
    images, labels = load_images_and_labels(images, labels)
    # for i in range(10): 
    #     print("Shape oof the image:", images[i].shape, "Shape of the label:", labels[i].shape) 
    return images, labels

# images, labels = load_data('train')

# code.interact(local=locals())

# print(images[0])
