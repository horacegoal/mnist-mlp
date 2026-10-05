import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = "./data"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")


def _transform():
    # return transforms.Compose([ ... , ... ])
    # two steps: ToTensor(), then Normalize((MNIST_MEAN,), (MNIST_STD,))
    pass


def get_loaders(batch_size=128, val_size=5000, seed=42, num_workers=2):
    # 1. train_full = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=_transform())
    # 2. test_set   = datasets.MNIST(DATA_DIR, train=False, download=True, transform=_transform())
    # 3. split train_full into train_set + val_set with random_split(...)
    # 4. wrap train_set / val_set / test_set in DataLoader(...)
    # 5. return train_loader, val_loader, test_loader
    pass


if __name__ == "__main__":
    d = device()
    print(d)
    # get loaders, grab one batch, print device / shapes / min-max