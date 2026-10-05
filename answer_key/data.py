import os

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = "./data"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def get_loaders(batch_size=128, val_size=5000, seed=42, num_workers=2):
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=_transform())
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=_transform())

    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tr, va, te = get_loaders()
    x, y = next(iter(tr))
    print("device:", device())
    print("train batches:", len(tr), "val batches:", len(va), "test batches:", len(te))
    print("batch x:", tuple(x.shape), "dtype:", x.dtype)
    print("batch y:", tuple(y.shape), "dtype:", y.dtype)
    print("x range:", float(x.min()), float(x.max()))
