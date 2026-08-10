"""Datasets: dynamically binarized MNIST and Omniglot (T1/T2 anchors)."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# On cluster nodes, point VMF_DATA_ROOT at node-local /scratch (shared
# constraints: hot data must not stream from net_scratch).
DATA_ROOT = Path(
    os.environ.get("VMF_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")
)


class DynamicBinarized(Dataset):
    """Wraps an image dataset, re-binarizing on every access (Salakhutdinov &
    Murray 2008, as used by Davidson et al.)."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        return (torch.rand_like(x) < x).float(), 0


def get_mnist(root: Path | None = None, dynamic: bool = True):
    root = root or DATA_ROOT
    tf = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    if dynamic:
        train, test = DynamicBinarized(train), DynamicBinarized(test)
    return train, test


def get_omniglot(root: Path | None = None, dynamic: bool = True):
    """Omniglot (background split train, evaluation split test), 28x28."""
    root = root or DATA_ROOT
    tf = transforms.Compose(
        [transforms.Resize(28), transforms.ToTensor(), transforms.Lambda(lambda x: 1.0 - x)]
    )
    train = datasets.Omniglot(root, background=True, download=True, transform=tf)
    test = datasets.Omniglot(root, background=False, download=True, transform=tf)
    if dynamic:
        train, test = DynamicBinarized(train), DynamicBinarized(test)
    return train, test


DATASETS = {
    "mnist": (get_mnist, 784),
    "omniglot": (get_omniglot, 784),
}
