import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .features import extract_features_pixel
from .loader import load_data


class FeatureExtractionDataset(Dataset):
    def __init__(self, images, labels, landmarks):
        self.images = images
        self.landmarks = landmarks
        self.labels = labels

        _len = 0
        for image in images:
            _len += image.shape[0] * image.shape[1]
        self.size = _len

        _image_sizes = [0]
        for image in images:
            _image_sizes.append(image.shape[0] * image.shape[1])
        self._image_sizes = np.cumsum(_image_sizes)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image_index = 0
        for i, sz in enumerate(self._image_sizes):
            if index >= sz:
                image_index = i
        image = self.images[image_index]
        landmark = self.landmarks[image_index]

        image_width = image.shape[1]
        n = index - self._image_sizes[image_index]
        y = n % image_width
        x = n // image_width
        
        features = extract_features_pixel(image, landmark, x=x, y=y)
        label = self.labels[image_index][x][y]

        return features, label


if __name__ == "__main__":
    images, labels, landmarks = load_data("train", 1)

    train_dataset = FeatureExtractionDataset(images, labels, landmarks)

    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False)

    for image, label in tqdm(train_dataloader):
        print(image.shape)
        print(label.shape)
        print(np.unique(label))
