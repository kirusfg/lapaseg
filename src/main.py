import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FeatureExtractionDataset
from data.loader import load_data
from models.mlp import mlp


def main():
    # os.environ['KERAS_BACKEND'] = 'torch'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    images, labels, landmarks = load_data("train", 5)

    train_dataset = FeatureExtractionDataset(images, labels, landmarks)

    train_dataloader = DataLoader(
        train_dataset, batch_size=1024, shuffle=True, num_workers=24
    )

    model = mlp(device=device)
    model.summary()

    model.fit(train_dataloader, epochs=2)

    model.save("mlp.keras")


if __name__ == "__main__":
    # main()
    features, labels, landmarks = load_data("train", 3)

    data = []
    train_dataset = FeatureExtractionDataset(features, labels, landmarks)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=24,
    )

    for features, labels in tqdm(train_dataloader):
        features, labels = features.numpy(), labels.numpy()
        labels = labels.reshape((labels.shape[0], 1))
        combined = np.hstack([features, labels])
        data.append(combined)

    data = np.vstack(data)

    np.savez_compressed("3.npz", data)
