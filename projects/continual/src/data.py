"""CIFAR-10 5+5 class-incremental data handling.

Everything lives on-device as flat tensors; batching is index permutation.
No augmentation (normalization only) — this is a mechanism replication,
not a SOTA run; noted in the report.
"""

import numpy as np
import torch
import torchvision

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_CACHE = {}


def load_cifar10(data_dir, device):
    key = (data_dir, str(device))
    if key in _CACHE:
        return _CACHE[key]
    # trigger download only; read the raw arrays directly (fast path)
    tr = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
    te = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)

    def to_flat(ds):
        x = torch.from_numpy(ds.data).float().div_(255.0)  # (N,32,32,3) uint8
        x = x.permute(0, 3, 1, 2).reshape(len(x), -1)  # (N, 3072) CHW-flat
        mean = torch.tensor(CIFAR10_MEAN).repeat_interleave(32 * 32)
        std = torch.tensor(CIFAR10_STD).repeat_interleave(32 * 32)
        return x.sub_(mean).div_(std)

    xtr = to_flat(tr).to(device)
    ytr = torch.tensor(tr.targets, dtype=torch.long, device=device)
    xte = to_flat(te).to(device)
    yte = torch.tensor(te.targets, dtype=torch.long, device=device)
    _CACHE[key] = (xtr, ytr, xte, yte)
    return _CACHE[key]


def make_partitions(n_partitions=5, base_seed=1000):
    """n_partitions random permutations of the 10 classes; first 5 are
    phase 1, last 5 are phase 2."""
    rng = np.random.default_rng(base_seed)
    return [rng.permutation(10).tolist() for _ in range(n_partitions)]


class SplitData:
    def __init__(self, data_dir, device, classes):
        xtr, ytr, xte, yte = load_cifar10(data_dir, device)
        mtr = torch.isin(ytr, torch.tensor(classes, device=device))
        mte = torch.isin(yte, torch.tensor(classes, device=device))
        self.xtr, self.ytr = xtr[mtr], ytr[mtr]
        self.xte, self.yte = xte[mte], yte[mte]
