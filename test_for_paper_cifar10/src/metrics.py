import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, roc_curve

def _ood_loader(in_ds, ood_ds):
    concat = ConcatDataset([in_ds, ood_ds])
    targets = torch.cat([torch.zeros(len(in_ds)), torch.ones(len(ood_ds))])
    return DataLoader(concat, 512, False, num_workers=2), targets

def auroc_entropy_single(in_ds, ood_ds, model, dev):
    dl, tgt = _ood_loader(in_ds, ood_ds)
    scr = []
    with torch.no_grad():
        for x, _ in dl:
            p = F.softmax(model(x.to(dev)), 1)
            scr.append((-torch.sum(p * torch.log(p + 1e-10), 1)).cpu().numpy())
    return roc_auc_score(tgt, np.concatenate(scr))

def auroc_entropy_ensemble(in_ds, ood_ds, ens, dev):
    dl, tgt = _ood_loader(in_ds, ood_ds)
    scr = []
    with torch.no_grad():
        for x, _ in dl:
            p = torch.stack([F.softmax(m(x.to(dev)), 1) for m in ens.models], 0).mean(0)
            scr.append((-torch.sum(p * torch.log(p + 1e-10), 1)).cpu().numpy())
    return roc_auc_score(tgt, np.concatenate(scr))

def auroc_kunc_ensemble(in_ds, ood_ds, ens, dev):
    dl, tgt = _ood_loader(in_ds, ood_ds)
    scr = []
    with torch.no_grad():
        for x, _ in dl:
            stack = torch.stack([F.softmax(m(x.to(dev)), 1) for m in ens.models], 0)
            avg = stack.mean(0)
            total = -torch.sum(avg * torch.log(avg + 1e-10), 1)
            per_e = -torch.sum(stack * torch.log(stack + 1e-10), 2).mean(0)
            scr.append((total - per_e).cpu().numpy())
    return roc_auc_score(tgt, np.concatenate(scr))

def _fpr_at_95_tpr(scores, labels, *, pos_label=1):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=pos_label)
    if np.all(tpr < 0.95):
        return float(fpr[np.argmax(tpr)])
    if np.all(tpr >= 0.95):
        return float(np.min(fpr))
    return float(np.interp(0.95, tpr, fpr))

def fpr95_entropy_single(in_ds, ood_ds, model, dev):
    dl, tgt = _ood_loader(in_ds, ood_ds)
    scr = []
    with torch.no_grad():
        for x, _ in dl:
            p = F.softmax(model(x.to(dev)), 1)
            ent = -torch.sum(p * torch.log(p + 1e-10), 1)
            scr.append(ent.cpu().numpy())
    return _fpr_at_95_tpr(np.concatenate(scr), tgt.numpy(), pos_label=1)

def fpr95_entropy_ensemble(in_ds, ood_ds, ens, dev):
    dl, tgt = _ood_loader(in_ds, ood_ds)
    scr = []
    with torch.no_grad():
        for x, _ in dl:
            p = torch.stack([F.softmax(m(x.to(dev)), 1) for m in ens.models], 0).mean(0)
            ent = -torch.sum(p * torch.log(p + 1e-10), 1)
            scr.append(ent.cpu().numpy())
    return _fpr_at_95_tpr(np.concatenate(scr), tgt.numpy(), pos_label=1)