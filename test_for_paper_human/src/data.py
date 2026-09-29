# -*- coding: utf-8 -*-
# src/data.py
from __future__ import annotations
import torch
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

# preprocessing (64x64)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

tfm_tr = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

tfm_te = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

tfm_ood = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

#  Dataset loading & splitting
def get_human_split(human_root: str, seed: int, tfm_tr, tfm_te):
    """
    Load full human dataset with train/test transforms and split 80/20 using given seed.
    Returns (tr_human, te_human) as Subset objects.
    """
    full_human_tr = torchvision.datasets.ImageFolder(human_root, transform=tfm_tr)
    full_human_te = torchvision.datasets.ImageFolder(human_root, transform=tfm_te)

    N = len(full_human_tr)
    n_train = int(0.8 * N)
    n_test = N - n_train

    split_g = torch.Generator().manual_seed(seed)
    tr_idx_split, te_idx_split = random_split(range(N), [n_train, n_test], generator=split_g)

    tr_human = Subset(full_human_tr, tr_idx_split.indices)  
    te_human = Subset(full_human_te, te_idx_split.indices)  
    return tr_human, te_human

def load_ood_datasets(tfm_ood):
    """
    Load all OOD datasets (CIFAR-10, CIFAR-100, Tiny-ImageNet, SVHN).
    Returns (ood_c10, ood_c100, ood_tin, ood_svhn).
    """
    ood_c10  = torchvision.datasets.CIFAR10("./data", False, download=True, transform=tfm_ood)
    ood_c100 = torchvision.datasets.CIFAR100("./data", False, download=True, transform=tfm_ood)
    ood_tin  = torchvision.datasets.ImageFolder("./tiny-imagenet-200/val", tfm_ood)
    ood_svhn = torchvision.datasets.SVHN("./data", split="test", download=True, transform=tfm_ood)
    return ood_c10, ood_c100, ood_tin, ood_svhn

#  DataLoaders                                                                 
def _seed_worker(worker_id: int):
    # Keeps per-worker RNG reproducible across runs when using multiple workers
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np
    import random
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loaders_for_seed(seed, tr_ds, te_ds, batch=64):
    """
    Create DataLoaders for a given seed and datasets.
    IMPORTANT: students want reproducible loaders, but NOT hard-deterministic ops.
    """
    # Reproducible RNG state (Python/NumPy/PyTorch)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(
        tr_ds, batch, True,
        num_workers=6, worker_init_fn=_seed_worker,
        generator=g, pin_memory=True
    )
    te_loader = DataLoader(
        te_ds, batch, False,
        num_workers=6, worker_init_fn=_seed_worker,
        generator=g, pin_memory=True
    )
    return tr_loader, te_loader