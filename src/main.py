import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FeatureExtractionDataset
from data.loader import load_data


if __name__ == "__main__":
    # How many images to extract per-pixel features from
    # Beware that features from one image might take up 5GiB of RAM
    n = 5
    features, labels, landmarks = load_data(
        "train", n
    )  # Choose either "train" or "test"

    data = []
    train_dataset = FeatureExtractionDataset(features, labels, landmarks)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=64,  # Adjust according to your spec
    )

    for features, labels in tqdm(train_dataloader):
        features, labels = features.numpy(), labels.numpy()
        labels = labels.reshape((labels.shape[0], 1))
        combined = np.hstack([features, labels])
        data.append(combined)

    data = np.vstack(data)

    np.savez_compressed(f"{n}.npz", data)
