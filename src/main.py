import os

import torch
from torch.utils.data import DataLoader

from data.dataset import FeatureExtractionDataset
from data.loader import load_data
from models.mlp import mlp


def main():
    # os.environ['KERAS_BACKEND'] = 'torch'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(device)

    images, labels, landmarks = load_data("train", 1)

    train_dataset = FeatureExtractionDataset(images, labels, landmarks)

    train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=False)

    model = mlp(device=device)
    model.summary()

    model.fit(train_dataloader, epochs=1)

    model.save('mlp.keras')


if __name__ == "__main__":
    main()