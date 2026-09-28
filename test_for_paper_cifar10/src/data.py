import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from utils import _seed_worker

# ImageNet statistics (mean/std) since the backbone is ResNet-18.

def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tfm_tr10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    tfm_te10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    tfm_ood = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tfm_tr10, tfm_te10, tfm_ood

def get_datasets():
    tfm_tr10, tfm_te10, tfm_ood = get_transforms()

    tr_c10 = torchvision.datasets.CIFAR10("./data", True, download=True, transform=tfm_tr10)
    te_c10 = torchvision.datasets.CIFAR10("./data", False, download=True, transform=tfm_te10)

    ood_c100 = torchvision.datasets.CIFAR100("./data", False, download=True, transform=tfm_ood)
    ood_tin = torchvision.datasets.ImageFolder("./tiny-imagenet-200/val", tfm_ood)
    ood_svhn = torchvision.datasets.SVHN("./data", split="test", download=True, transform=tfm_ood)
    ood_human = torchvision.datasets.ImageFolder("./human detection dataset", tfm_ood)

    return tr_c10, te_c10, ood_c100, ood_tin, ood_svhn, ood_human

def make_loaders_for_seed(seed, tr_ds, te_ds, batch=128, num_workers=6):
    from utils import set_global_seed  
    set_global_seed(seed)              
    g = torch.Generator().manual_seed(seed)

    tr_loader = DataLoader(
        tr_ds, batch_size=batch, shuffle=True,
        num_workers=num_workers, worker_init_fn=_seed_worker,
        generator=g, pin_memory=True
    )
    te_loader = DataLoader(
        te_ds, batch_size=batch, shuffle=False,
        num_workers=num_workers, worker_init_fn=_seed_worker,
        generator=g, pin_memory=True
    )
    return tr_loader, te_loader
