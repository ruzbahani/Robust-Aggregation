from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def load_cifar100(root: str = "./data", augment: bool = True) -> Tuple[Dataset, Dataset]:
    import torchvision
    import torchvision.transforms as T
    norm = T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    tr = T.Compose(([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []) + [T.ToTensor(), norm])
    te = T.Compose([T.ToTensor(), norm])
    train = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=tr)
    test = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=te)
    return train, test


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(root: str = "./data", augment: bool = True) -> Tuple[Dataset, Dataset]:
    import torchvision
    import torchvision.transforms as T
    norm = T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    tr = T.Compose(([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []) + [T.ToTensor(), norm])
    te = T.Compose([T.ToTensor(), norm])
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tr)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=te)
    return train, test


def load_dataset(name: str, root: str = "./data", augment: bool = True):
    if name == "cifar100":
        return (*load_cifar100(root, augment), 100)
    if name == "cifar10":
        return (*load_cifar10(root, augment), 10)
    raise ValueError(f"unknown dataset {name!r}")


def load_synthetic(n_train: int = 2000, n_test: int = 500, num_classes: int = 100, seed: int = 0) -> Tuple[Dataset, Dataset]:
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, 3, 32, 32, generator=g) * 0.6

    def make(n):
        y = torch.randint(0, num_classes, (n,), generator=g)
        x = centers[y] + 0.5 * torch.randn(n, 3, 32, 32, generator=g)
        return TensorDataset(x, y)

    return make(n_train), make(n_test)


def labels_of(ds: Dataset) -> np.ndarray:
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets)
    if isinstance(ds, TensorDataset):
        return ds.tensors[1].numpy()
    return np.asarray([ds[i][1] for i in range(len(ds))])


def dirichlet_partition(ds: Dataset, n_clients: int, alpha: float, num_classes: int, seed: int = 0,
                        min_size: int = 10) -> List[Subset]:
    y = labels_of(ds)
    rng = np.random.default_rng(seed)
    while True:
        idx_per_client = [[] for _ in range(n_clients)]
        for k in range(num_classes):
            idx_k = np.where(y == k)[0]
            rng.shuffle(idx_k)
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p) * len(idx_k)).astype(int)[:-1]
            for c, part in enumerate(np.split(idx_k, cuts)):
                idx_per_client[c].extend(part.tolist())
        if min(len(i) for i in idx_per_client) >= min_size:
            break
    return [Subset(ds, sorted(i)) for i in idx_per_client]
