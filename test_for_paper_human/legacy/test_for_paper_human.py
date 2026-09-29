from __future__ import annotations
# -*- coding: utf-8 -*-


import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np, torch, torchvision, random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from itertools import product
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve



def seed_everything(seed: int):
    """Reproducible RNG state (Python/NumPy/PyTorch)"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_determinism(enabled: bool):
    """
    If enabled=True: hard deterministic (same exact results run-to-run).
    If enabled=False: allow fast non-deterministic kernels (students), but keep seeds for reproducibility.
    """
    try:
        torch.use_deterministic_algorithms(enabled)
    except Exception:
        pass

    torch.backends.cudnn.deterministic = bool(enabled)
    torch.backends.cudnn.benchmark = (not enabled)

    if enabled:
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

def seed_and_optionally_deterministic(seed: int, *, deterministic: bool):
    seed_everything(seed)
    set_determinism(deterministic)

def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _dash(x): return "–" if (x is None or x == "–") else x
def mean_std(a): return float(np.mean(a)), float(np.std(a))



class ResNet18Backbone(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        try:
            base = torchvision.models.resnet18(weights=None)
        except TypeError:
            base = torchvision.models.resnet18(pretrained=False)

        self.features = nn.Sequential(*list(base.children())[:-1])  
        in_feats = base.fc.in_features
        self.fc = nn.Linear(in_feats, n_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)

    def get_features(self, x):
        x = self.features(x).flatten(1)
        return x

def resnet18_backbone(n_classes=2):
    return ResNet18Backbone(n_classes=n_classes)


class TeacherEnsemble(nn.Module):
    def __init__(self, n=5, n_classes=2, base_seed=42):
        super().__init__()
        self.models = nn.ModuleList()
        for i in range(n):
            local_seed = base_seed + i
            seed_and_optionally_deterministic(local_seed, deterministic=True)
            self.models.append(resnet18_backbone(n_classes=n_classes))

    def forward(self, x):
        return torch.stack([m(x) for m in self.models], 0).mean(0)

    def get_features(self, x):
        return torch.stack([m.get_features(x) for m in self.models], 0).mean(0)



def train_one_epoch(m, ldr, opt, crit, dev):
    m.train(); loss=cor=tot=0
    for x,y in ldr:
        x,y = x.to(dev), y.to(dev)
        opt.zero_grad(); o = m(x); l = crit(o,y); l.backward(); opt.step()
        loss += l.item()*x.size(0); cor += o.argmax(1).eq(y).sum().item(); tot += y.size(0)
    return loss/tot, 100.*cor/tot

def validate(m, ldr, crit, dev):
    m.eval(); loss=cor=tot=0
    with torch.no_grad():
        for x,y in ldr:
            x,y = x.to(dev), y.to(dev); o = m(x); l = crit(o,y)
            loss += l.item()*x.size(0); cor += o.argmax(1).eq(y).sum().item(); tot += y.size(0)
    return loss/tot, 100.*cor/tot



def kd_loss(s, t, *, T, alpha):
    kl = F.kl_div(F.log_softmax(s/T,1), F.softmax(t/T,1), reduction="batchmean")*(T*T)
    return kl

def kd_loss1(s_logits, t_logits, labels, *, T, alpha):
    ce = F.cross_entropy(s_logits, labels)
    kl = F.kl_div(
        F.log_softmax(s_logits / T, 1),
        F.softmax(t_logits / T, 1),
        reduction="batchmean"
    ) * (T*T)
    return alpha*ce + (1-alpha)*kl

def cosine_similarity_loss(o, t, eps=1e-7):
    o = F.normalize(o, dim=1); t = F.normalize(t, dim=1)
    so = (torch.mm(o,o.t())+1)/2; st = (torch.mm(t,t.t())+1)/2
    so /= so.sum(1,keepdim=True)+eps; st /= st.sum(1,keepdim=True)+eps
    return torch.sum(st * torch.log((st+eps)/(so+eps)))


class Encoder32(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        try: base = torchvision.models.resnet18(weights=None)
        except TypeError: base = torchvision.models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))

class Decoder32(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d( 64, 64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d( 64, 64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d( 64, 64,4,2,1), nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(z.size(0),512,1,1))

class CAE2(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = Encoder32(latent_dim)
        self.decoder = Decoder32(latent_dim)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class General_Class_Network(nn.Module):
    def __init__(self, input_size=2, output_size=2, noise_std=1e-4):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.linear.weight.data = torch.eye(input_size, output_size) + noise_std * torch.randn(input_size, output_size)
    def forward(self, x):
        return self.linear(x)


def _sgd_optimizer(params, lr=0.005, momentum=0.9, weight_decay=5e-4):
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

def _step_scheduler(opt):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.1)

def train_student_normal(m, tr, te, *, device, epochs):
    crit = nn.CrossEntropyLoss()
    opt = _sgd_optimizer(m.parameters(), lr=0.005)
    scheduler = _step_scheduler(opt)

    m.to(device)
    for ep in range(epochs):
        m.train(); loss=cor=tot=0
        for x,y in tr:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            o = m(x); l = crit(o,y)
            l.backward(); opt.step()
            loss += l.item()*x.size(0); cor += o.argmax(1).eq(y).sum().item(); tot += y.size(0)

        scheduler.step()
        vl_l, vl_a = validate(m, te, crit, device)
        print(f"[Student] Ep {ep+1:02}/{epochs} | Train {loss/tot:.4f}/{100.*cor/tot:.2f}% | "
              f"Val {vl_l:.4f}/{vl_a:.2f}% | LR {scheduler.get_last_lr()[0]:.6f}")
    return m

def train_student_kd(m, teacher, tr, te, *, device, epochs, T, alpha):
    crit = nn.CrossEntropyLoss()
    opt = _sgd_optimizer(m.parameters(), lr=0.005)
    scheduler = _step_scheduler(opt)

    m.to(device); teacher.eval().to(device)
    for p in teacher.parameters(): p.requires_grad=False

    for ep in range(epochs):
        m.train(); loss=cor=tot=0
        for x,y in tr:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            s=m(x); t=teacher(x)
            l=kd_loss1(s,t,y,T=T,alpha=alpha)
            l.backward(); opt.step()
            loss+=l.item()*x.size(0); cor+=s.argmax(1).eq(y).sum().item(); tot+=y.size(0)

        scheduler.step()
        vl_l,vl_a=validate(m,te,crit,device)
        print(f"[KD Student] Ep {ep+1:02}/{epochs} | Train {loss/tot:.4f}/{100.*cor/tot:.2f}% | "
              f"Val {vl_l:.4f}/{vl_a:.2f}% | LR {scheduler.get_last_lr()[0]:.6f}")
    return m

def train_student_pkt(m, teacher, tr, te, *, device, epochs, lamb):
    crit = nn.CrossEntropyLoss()
    opt = _sgd_optimizer(m.parameters(), lr=0.005)
    scheduler = _step_scheduler(opt)

    m.to(device); teacher.eval().to(device)
    for p in teacher.parameters(): p.requires_grad=False

    for ep in range(epochs):
        m.train(); loss=cor=tot=0
        for x,y in tr:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            pkt=cosine_similarity_loss(m.get_features(x),teacher.get_features(x))
            logits=m(x); ce=crit(logits,y); l=lamb*pkt+ce
            l.backward(); opt.step()
            loss+=l.item()*x.size(0); cor+=logits.argmax(1).eq(y).sum().item(); tot+=y.size(0)

        scheduler.step()
        vl_l,vl_a=validate(m,te,crit,device)
        print(f"[PKT Student] Ep {ep+1:02}/{epochs} | Train {loss/tot:.4f}/{100.*cor/tot:.2f}% | "
              f"Val {vl_l:.4f}/{vl_a:.2f}% | LR {scheduler.get_last_lr()[0]:.6f}")
    return m


def train_student_kd_asl(
    m, teacher, ae, ae_t, gcn_s, gcn_t,
    tr, te, * ,
    device, epochs,
    T_kd, alpha_kd,
    alpha_gcn, beta_ae, T_ae,
    lam_rec=2.,
    alpha_gcn_t=0.1, beta_ae_t=0.1, T_ae_t=5.0
):
    def kd_kl_from_probs(s_logits, t_probs, *, T):
        t_probs = t_probs.detach()
        t_probs = t_probs / t_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return F.kl_div(
            F.log_softmax(s_logits / T, dim=1),
            t_probs,
            reduction="batchmean"
        ) * (T * T)

    crit_ce      = nn.CrossEntropyLoss()
    crit_bce     = nn.BCELoss()
    crit_mse     = nn.MSELoss()

    opt = _sgd_optimizer(m.parameters(), lr=0.005)
    scheduler = _step_scheduler(opt)

    gopt_s   = optim.Adam(gcn_s.parameters(),  lr=1e-5)
    gopt_t   = optim.Adam(gcn_t.parameters(),  lr=1e-5)
    opt_ae   = optim.Adam(ae.parameters(),     lr=1e-5)
    opt_ae_t = optim.Adam(ae_t.parameters(),   lr=1e-5)

    m.to(device)
    teacher.eval().to(device)
    ae.train().to(device)
    ae_t.train().to(device)
    gcn_s.train().to(device)
    gcn_t.train().to(device)

    for p in teacher.parameters():
        p.requires_grad = False

    for ep in range(epochs):
        m.train()
        loss = cor = tot = 0
        sum_y_hat = sum_y_hat_t = 0.0
        sum_ae_s = sum_ae_t = 0.0
        sum_teacher = sum_student = 0.0
        total_samples = 0

        for images, labels in tr:
            x, y = images.to(device), labels.to(device)
            labels_idx = y
            C = m.fc.out_features
            y_onehot = F.one_hot(y, num_classes=C).float()

            opt.zero_grad()
            opt_ae.zero_grad()
            opt_ae_t.zero_grad()
            gopt_s.zero_grad()
            gopt_t.zero_grad()

            t_logits = teacher(x)
            y_t = F.softmax(t_logits / T_kd, dim=1)
            s_logits = m(x)

            z = ae.encoder(x)
            x_rec = ae.decoder(z)
            p_ae = F.softmax(z / T_ae, dim=1)

            gcn_s_logits = gcn_s(y_onehot)
            g_s_y = F.softmax(gcn_s_logits, dim=1)

            y_hat = (
                alpha_gcn * g_s_y +
                beta_ae   * p_ae +
                (1.0 - alpha_gcn - beta_ae) * y_onehot
            )

            y_hat_gcn = (
                alpha_gcn * g_s_y +
                beta_ae   * p_ae.detach() +
                (1.0 - alpha_gcn - beta_ae) * y_onehot
            )

            p_student = F.log_softmax(s_logits, dim=1)
            loss_cls = -(y_hat.detach() * p_student).sum(dim=1).mean()

            loss_ae_s = crit_ce(z, labels_idx) + lam_rec * crit_mse(x_rec, x)
            loss_gcn_s = crit_bce(y_hat_gcn, F.softmax(s_logits, dim=1).detach())

            z_t = ae_t.encoder(x)
            x_rec_t = ae_t.decoder(z_t)
            p_ae_t = F.softmax(z_t / T_ae_t, dim=1)

            gcn_t_logits = gcn_t(y_t)
            g_t_yt = F.softmax(gcn_t_logits, dim=1)

            y_hat_t = (
                alpha_gcn_t * g_t_yt +
                beta_ae_t   * p_ae_t +
                (1.0 - alpha_gcn_t - beta_ae_t) * y_t
            )

            y_hat_gcn_t = (
                alpha_gcn_t * g_t_yt +
                beta_ae_t   * p_ae_t.detach() +
                (1.0 - alpha_gcn_t - beta_ae_t) * y_t
            )

            loss_gcn_t = crit_bce(y_hat_gcn_t, F.softmax(s_logits, dim=1).detach())

            kd_student = kd_kl_from_probs(s_logits, y_hat_t, T=T_kd)
            kd_teacher_ae = kd_loss(z_t, t_logits, T=T_kd, alpha=alpha_kd)

            loss_distill = kd_student
            loss_ae_t = kd_teacher_ae + lam_rec * crit_mse(x_rec_t, x)

            total_loss = alpha_kd * loss_cls + (1 - alpha_kd) * loss_distill + loss_ae_t + loss_gcn_s + loss_gcn_t + loss_ae_s
            total_loss.backward()

            opt.step()
            gopt_s.step()
            gopt_t.step()
            opt_ae.step()
            opt_ae_t.step()

            with torch.no_grad():
                max_y_hat, _     = y_hat.max(dim=1)
                max_y_hat_t, _   = y_hat_t.max(dim=1)
                max_ae_s, _      = p_ae.max(dim=1)
                max_ae_t, _      = p_ae_t.max(dim=1)
                max_teacher, _   = y_t.max(dim=1)
                max_student, _   = F.softmax(s_logits, dim=1).max(dim=1)

                sum_y_hat     += max_y_hat.sum().item()
                sum_y_hat_t   += max_y_hat_t.sum().item()
                sum_ae_s      += max_ae_s.sum().item()
                sum_ae_t      += max_ae_t.sum().item()
                sum_teacher   += max_teacher.sum().item()
                sum_student   += max_student.sum().item()
                total_samples += y.size(0)

            loss += total_loss.item() * x.size(0)
            cor  += s_logits.argmax(1).eq(labels_idx).sum().item()
            tot  += labels_idx.size(0)

        scheduler.step()

        tr_l, tr_a = loss / tot, 100. * cor / tot
        vl_l, vl_a = validate(m, te, crit_ce, device)

        mean_y_hat     = sum_y_hat     / max(1, total_samples)
        mean_y_hat_t   = sum_y_hat_t   / max(1, total_samples)
        mean_ae_s      = sum_ae_s      / max(1, total_samples)
        mean_ae_t      = sum_ae_t      / max(1, total_samples)
        mean_teacher   = sum_teacher   / max(1, total_samples)
        mean_student   = sum_student   / max(1, total_samples)

        print(
            f"[KD+ASL] Ep {ep+1:02}/{epochs} | "
            f"Train {tr_l:.4f}/{tr_a:.2f}% | "
            f"Val {vl_l:.4f}/{vl_a:.2f}% | "
            f"LR {scheduler.get_last_lr()[0]:.6f} | "
            f"soft(y_hat)={mean_y_hat:.3f}, "
            f"soft(y_hat_t)={mean_y_hat_t:.3f}, "
            f"soft(AE_s)={mean_ae_s:.3f}, "
            f"soft(AE_t)={mean_ae_t:.3f}, "
            f"soft(teacher)={mean_teacher:.3f}, "
            f"soft(student)={mean_student:.3f}"
        )

    return m


def _ood_loader(in_ds, ood_ds):
    concat = ConcatDataset([in_ds, ood_ds])
    targets = torch.cat([torch.zeros(len(in_ds)), torch.ones(len(ood_ds))])
    return DataLoader(concat, 512, False, num_workers=2), targets

def auroc_entropy_single(in_ds, ood_ds, model, dev):
    dl, tgt = _ood_loader(in_ds, ood_ds); scr = []
    with torch.no_grad():
        for x, _ in dl:
            p = F.softmax(model(x.to(dev)), 1)
            scr.append((-torch.sum(p * torch.log(p + 1e-10), 1)).cpu().numpy())
    return roc_auc_score(tgt, np.concatenate(scr))

def auroc_entropy_ensemble(in_ds, ood_ds, ens, dev):
    dl, tgt = _ood_loader(in_ds, ood_ds); scr = []
    with torch.no_grad():
        for x, _ in dl:
            p = torch.stack([F.softmax(m(x.to(dev)), 1) for m in ens.models], 0).mean(0)
            scr.append((-torch.sum(p * torch.log(p + 1e-10), 1)).cpu().numpy())
    return roc_auc_score(tgt, np.concatenate(scr))

def auroc_kunc_ensemble(in_ds, ood_ds, ens, dev):
    dl, tgt = _ood_loader(in_ds, ood_ds); scr = []
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
    dl, tgt = _ood_loader(in_ds, ood_ds); scr = []
    with torch.no_grad():
        for x, _ in dl:
            p = F.softmax(model(x.to(dev)), 1)
            ent = -torch.sum(p * torch.log(p + 1e-10), 1)
            scr.append(ent.cpu().numpy())
    return _fpr_at_95_tpr(np.concatenate(scr), tgt.numpy(), pos_label=1)

def fpr95_entropy_ensemble(in_ds, ood_ds, ens, dev):
    dl, tgt = _ood_loader(in_ds, ood_ds); scr = []
    with torch.no_grad():
        for x, _ in dl:
            p = torch.stack([F.softmax(m(x.to(dev)), 1) for m in ens.models], 0).mean(0)
            ent = -torch.sum(p * torch.log(p + 1e-10), 1)
            scr.append(ent.cpu().numpy())
    return _fpr_at_95_tpr(np.concatenate(scr), tgt.numpy(), pos_label=1)


def make_loaders_for_seed(seed, tr_ds, te_ds, batch=64):
    seed_everything(seed)  
    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(tr_ds, batch, True,
                           num_workers=6, worker_init_fn=_seed_worker,
                           generator=g, pin_memory=True)
    te_loader = DataLoader(te_ds, batch, False,
                           num_workers=6, worker_init_fn=_seed_worker,
                           generator=g, pin_memory=True)
    return tr_loader, te_loader



def main():
    base_seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    C = 2
    t_epochs = 50
    s_epochs = 50

    alpha_kd, T_kd = 0.5, 2.5
    lambda_pkt = 1.0
    n_runs = 5

    from torchvision import transforms
    from torch.utils.data import Subset

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

    
    human_root = "./human detection dataset"

    
    full_human_tr = torchvision.datasets.ImageFolder(human_root, transform=tfm_tr)
    full_human_te = torchvision.datasets.ImageFolder(human_root, transform=tfm_te)

    
    N = len(full_human_tr)
    n_train = int(0.8 * N)
    n_test = N - n_train

    split_g = torch.Generator().manual_seed(base_seed)
    tr_idx_split, te_idx_split = random_split(range(N), [n_train, n_test], generator=split_g)

    tr_human = Subset(full_human_tr, tr_idx_split.indices)  
    te_human = Subset(full_human_te, te_idx_split.indices)  

    
    ood_c10  = torchvision.datasets.CIFAR10("./data", False, download=True, transform=tfm_ood)
    ood_c100 = torchvision.datasets.CIFAR100("./data", False, download=True, transform=tfm_ood)
    ood_tin  = torchvision.datasets.ImageFolder("./tiny-imagenet-200/val", tfm_ood)
    ood_svhn = torchvision.datasets.SVHN("./data", split="test", download=True, transform=tfm_ood)

    crit = nn.CrossEntropyLoss()

    
    print("=== TEACHER ENSEMBLE SETUP (DETERMINISTIC) ===")
    seed_and_optionally_deterministic(base_seed, deterministic=True)

    gT = torch.Generator().manual_seed(base_seed)
    tr_loader_T = DataLoader(
        tr_human, 64, True,
        num_workers=6, worker_init_fn=_seed_worker, generator=gT, pin_memory=True
    )
    te_loader_T = DataLoader(
        te_human, 64, False,
        num_workers=6, worker_init_fn=_seed_worker, generator=gT, pin_memory=True
    )

    teacher_ckpt = "./teacher_human_ens5.pth"
    if os.path.dirname(teacher_ckpt):
        os.makedirs(os.path.dirname(teacher_ckpt), exist_ok=True)

    teacher = TeacherEnsemble(n=5, n_classes=C, base_seed=base_seed).to(device)

    if os.path.exists(teacher_ckpt):
        print(f"Loading teacher ensemble weights from: {teacher_ckpt}")
        state = torch.load(teacher_ckpt, map_location=device)
        teacher.load_state_dict(state)
    else:
        print("No pretrained teacher found. Training teacher ensemble from scratch (DETERMINISTIC)...")
        for i, net in enumerate(teacher.models, 1):
            print(f"[Teacher {i}/5] Training ResNet-18 on HUMAN(ID) …")
            train_student_normal(net, tr_loader_T, te_loader_T, device=device, epochs=t_epochs)
        torch.save(teacher.state_dict(), teacher_ckpt)
        print(f"Saved teacher ensemble weights to: {teacher_ckpt}")

    
    teacher_accs = []
    teacher_au_c10 = []; teacher_kunc_c10 = []; teacher_fpr_c10 = []
    teacher_au_c100 = []; teacher_kunc_c100 = []; teacher_fpr_c100 = []
    teacher_au_tin = []; teacher_kunc_tin = []; teacher_fpr_tin = []
    teacher_au_svhn = []; teacher_kunc_svhn = []; teacher_fpr_svhn = []

    _, acc = validate(teacher, te_loader_T, crit, device)
    teacher_accs.append(acc)

    teacher_au_c10.append(auroc_entropy_ensemble(te_human, ood_c10, teacher, device))
    teacher_kunc_c10.append(auroc_kunc_ensemble(te_human, ood_c10, teacher, device))
    teacher_fpr_c10.append(fpr95_entropy_ensemble(te_human, ood_c10, teacher, device))

    teacher_au_c100.append(auroc_entropy_ensemble(te_human, ood_c100, teacher, device))
    teacher_kunc_c100.append(auroc_kunc_ensemble(te_human, ood_c100, teacher, device))
    teacher_fpr_c100.append(fpr95_entropy_ensemble(te_human, ood_c100, teacher, device))

    teacher_au_tin.append(auroc_entropy_ensemble(te_human, ood_tin, teacher, device))
    teacher_kunc_tin.append(auroc_kunc_ensemble(te_human, ood_tin, teacher, device))
    teacher_fpr_tin.append(fpr95_entropy_ensemble(te_human, ood_tin, teacher, device))

    teacher_au_svhn.append(auroc_entropy_ensemble(te_human, ood_svhn, teacher, device))
    teacher_kunc_svhn.append(auroc_kunc_ensemble(te_human, ood_svhn, teacher, device))
    teacher_fpr_svhn.append(fpr95_entropy_ensemble(te_human, ood_svhn, teacher, device))

    
    print("\n=== SWITCHING TO STUDENT MODE (seeded, non-deterministic ops) ===")
    set_determinism(False)  

    
    print("[Autoencoder] Preparing CAE2(C) on HUMAN(ID) …")
    seed_everything(base_seed)

    lamda = 2.0
    ae_ckpt_full = "./ae_human_full.pth"
    ae_ckpt_enc  = "./ae_human_encoder.pth"

    ae_base = CAE2(latent_dim=C).to(device)

    if os.path.exists(ae_ckpt_full):
        print(f"[AE] Found checkpoint: {ae_ckpt_full} -> loading weights.")
        state = torch.load(ae_ckpt_full, map_location=device)
        ae_base.load_state_dict(state)
    else:
        print("[AE] No checkpoint found. Pretraining CAE2(C) from scratch and saving weights …")

        opt_ae = torch.optim.Adam(ae_base.parameters(), lr=1e-4)
        mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        num_epochs = 100

        ae_base.train()
        for ep in range(num_epochs):
            run_loss = 0.0
            for x, y in tr_loader_T:
                x, y = x.to(device), y.to(device)
                opt_ae.zero_grad()

                z = ae_base.encoder(x)
                x_hat = ae_base.decoder(z)

                loss = lamda * mse(x_hat, x) + ce(softmax(z), y)
                loss.backward()
                opt_ae.step()

                run_loss += loss.item()

            print(f"[AE] Epoch {ep + 1:03}/{num_epochs} | Loss {run_loss / len(tr_loader_T):.4f}")

        ae_base.eval()
        torch.save(ae_base.state_dict(), ae_ckpt_full)
        torch.save(ae_base.encoder.state_dict(), ae_ckpt_enc)
        print(f"[AE] Saved full AE to {ae_ckpt_full} and encoder to {ae_ckpt_enc}")

    ae_base.eval()

    # ---------- metric containers (students) -------------------------------
    sn_accs = []; sn_au_c10 = []; sn_au_c100 = []; sn_au_tin = []; sn_au_svhn = []
    sn_fpr_c10 = []; sn_fpr_c100 = []; sn_fpr_tin = []; sn_fpr_svhn = []

    kd_accs = []; kd_au_c10 = []; kd_au_c100 = []; kd_au_tin = []; kd_au_svhn = []
    kd_fpr_c10 = []; kd_fpr_c100 = []; kd_fpr_tin = []; kd_fpr_svhn = []

    pkt_accs = []; pkt_au_c10 = []; pkt_au_c100 = []; pkt_au_tin = []; pkt_au_svhn = []
    pkt_fpr_c10 = []; pkt_fpr_c100 = []; pkt_fpr_tin = []; pkt_fpr_svhn = []

    # ---------- KD-ASL grid parameters ------------------------------------
    alpha_gcn_vals = [0.0]
    beta_ae_vals = [0.0]
    T_ae_vals = [5.0]
    alpha_gcn_t_vals = [0.0]
    beta_ae_t_vals = [0.0]
    T_ae_t_vals = [5.0]
    grid_params = list(product(
        alpha_gcn_vals, beta_ae_vals, T_ae_vals,
        alpha_gcn_t_vals, beta_ae_t_vals, T_ae_t_vals
    ))

    print(f"[GRID] Will run {len(grid_params)} KD+ASL configs:")
    for p in grid_params:
        print("  ", p)

    kd_asl_accs_map      = defaultdict(list)
    kd_asl_au_c10_map    = defaultdict(list)
    kd_asl_au_c100_map   = defaultdict(list)
    kd_asl_au_tin_map    = defaultdict(list)
    kd_asl_au_svhn_map   = defaultdict(list)
    kd_asl_fpr_c10_map   = defaultdict(list)
    kd_asl_fpr_c100_map  = defaultdict(list)
    kd_asl_fpr_tin_map   = defaultdict(list)
    kd_asl_fpr_svhn_map  = defaultdict(list)

    # =================== runs =============================================
    for run in range(n_runs):
        print(f"\n=== RUN {run+1}/{n_runs} ===")
        run_seed = base_seed + run * 1000

        seed_everything(run_seed)

        full_human_run_tr = torchvision.datasets.ImageFolder(human_root, transform=tfm_tr)
        full_human_run_te = torchvision.datasets.ImageFolder(human_root, transform=tfm_te)

        N_run = len(full_human_run_tr)
        n_train_run = int(0.8 * N_run)
        n_test_run = N_run - n_train_run

        split_g = torch.Generator().manual_seed(run_seed)
        tr_idx_split, te_idx_split = random_split(range(N_run), [n_train_run, n_test_run], generator=split_g)

        tr_human_run = Subset(full_human_run_tr, tr_idx_split.indices)  
        te_human_run = Subset(full_human_run_te, te_idx_split.indices)  

        tr_loader, te_loader = make_loaders_for_seed(run_seed, tr_human_run, te_human_run, batch=64)

        # -------- Student baseline ----------------------------------------
        print("Training Student Baseline...")
        seed_everything(run_seed)  # ensures same init each time for this run
        s = resnet18_backbone(n_classes=C)
        train_student_normal(s, tr_loader, te_loader, device=device, epochs=s_epochs)
        _, acc = validate(s, te_loader, crit, device)
        sn_accs.append(acc)

        sn_au_c10.append(auroc_entropy_single(te_human_run, ood_c10, s, device))
        sn_au_c100.append(auroc_entropy_single(te_human_run, ood_c100, s, device))
        sn_au_tin.append(auroc_entropy_single(te_human_run, ood_tin, s, device))
        sn_au_svhn.append(auroc_entropy_single(te_human_run, ood_svhn, s, device))

        sn_fpr_c10.append(fpr95_entropy_single(te_human_run, ood_c10, s, device))
        sn_fpr_c100.append(fpr95_entropy_single(te_human_run, ood_c100, s, device))
        sn_fpr_tin.append(fpr95_entropy_single(te_human_run, ood_tin, s, device))
        sn_fpr_svhn.append(fpr95_entropy_single(te_human_run, ood_svhn, s, device))

        # -------- Student KD ----------------------------------------------
        print("Training Student with KD...")
        seed_everything(run_seed)
        s = resnet18_backbone(n_classes=C)
        train_student_kd(
            s, teacher, tr_loader, te_loader,
            device=device, epochs=s_epochs,
            T=T_kd, alpha=alpha_kd
        )
        _, acc = validate(s, te_loader, crit, device)
        kd_accs.append(acc)

        kd_au_c10.append(auroc_entropy_single(te_human_run, ood_c10, s, device))
        kd_au_c100.append(auroc_entropy_single(te_human_run, ood_c100, s, device))
        kd_au_tin.append(auroc_entropy_single(te_human_run, ood_tin, s, device))
        kd_au_svhn.append(auroc_entropy_single(te_human_run, ood_svhn, s, device))

        kd_fpr_c10.append(fpr95_entropy_single(te_human_run, ood_c10, s, device))
        kd_fpr_c100.append(fpr95_entropy_single(te_human_run, ood_c100, s, device))
        kd_fpr_tin.append(fpr95_entropy_single(te_human_run, ood_tin, s, device))
        kd_fpr_svhn.append(fpr95_entropy_single(te_human_run, ood_svhn, s, device))

        # -------- Student PKT ---------------------------------------------
        print("Training Student with PKT...")
        seed_everything(run_seed)
        s = resnet18_backbone(n_classes=C)
        train_student_pkt(
            s, teacher, tr_loader, te_loader,
            device=device, epochs=s_epochs,
            lamb=lambda_pkt
        )
        _, acc = validate(s, te_loader, crit, device)
        pkt_accs.append(acc)

        pkt_au_c10.append(auroc_entropy_single(te_human_run, ood_c10, s, device))
        pkt_au_c100.append(auroc_entropy_single(te_human_run, ood_c100, s, device))
        pkt_au_tin.append(auroc_entropy_single(te_human_run, ood_tin, s, device))
        pkt_au_svhn.append(auroc_entropy_single(te_human_run, ood_svhn, s, device))

        pkt_fpr_c10.append(fpr95_entropy_single(te_human_run, ood_c10, s, device))
        pkt_fpr_c100.append(fpr95_entropy_single(te_human_run, ood_c100, s, device))
        pkt_fpr_tin.append(fpr95_entropy_single(te_human_run, ood_tin, s, device))
        pkt_fpr_svhn.append(fpr95_entropy_single(te_human_run, ood_svhn, s, device))

        # -------- KD+ASL grid ---------------------------------------------
        print("Training KD+ASL variants...")
        print("[Autoencoder] Using pretrained CAE2(C) weights for this run …")

        for (a, b, t, a_t, b_t, t_t) in grid_params:
            print(f"[GRID KD+ASL] a={a},b={b},T={t}|a_t={a_t},b_t={b_t},T_t={t_t}")
            tr_loader_cfg, te_loader_cfg = make_loaders_for_seed(run_seed, tr_human_run, te_human_run, batch=64)

            seed_everything(run_seed)
            gcn_s = General_Class_Network(input_size=C, output_size=C).to(device)
            gcn_t = General_Class_Network(input_size=C, output_size=C).to(device)

            ae = CAE2(latent_dim=C).to(device)
            ae.load_state_dict(ae_base.state_dict())
            ae_t = CAE2(latent_dim=C).to(device)
            ae_t.load_state_dict(ae_base.state_dict())

            s = resnet18_backbone(n_classes=C).to(device)

            train_student_kd_asl(
                s, teacher, ae, ae_t, gcn_s, gcn_t,
                tr_loader_cfg, te_loader_cfg,
                device=device, epochs=s_epochs,
                T_kd=T_kd, alpha_kd=alpha_kd,
                alpha_gcn=a, beta_ae=b, T_ae=t,
                lam_rec=2.0,
                alpha_gcn_t=a_t, beta_ae_t=b_t, T_ae_t=t_t
            )

            _, acc = validate(s, te_loader_cfg, crit, device)
            key = (a, b, t, a_t, b_t, t_t)

            kd_asl_accs_map[key].append(acc)
            kd_asl_au_c10_map[key].append(auroc_entropy_single(te_human_run, ood_c10, s, device))
            kd_asl_au_c100_map[key].append(auroc_entropy_single(te_human_run, ood_c100, s, device))
            kd_asl_au_tin_map[key].append(auroc_entropy_single(te_human_run, ood_tin, s, device))
            kd_asl_au_svhn_map[key].append(auroc_entropy_single(te_human_run, ood_svhn, s, device))

            kd_asl_fpr_c10_map[key].append(fpr95_entropy_single(te_human_run, ood_c10, s, device))
            kd_asl_fpr_c100_map[key].append(fpr95_entropy_single(te_human_run, ood_c100, s, device))
            kd_asl_fpr_tin_map[key].append(fpr95_entropy_single(te_human_run, ood_tin, s, device))
            kd_asl_fpr_svhn_map[key].append(fpr95_entropy_single(te_human_run, ood_svhn, s, device))

    # =================== summary ===========================================
    hdr = "{:<60} {:>7}  {:>8} {:>8} {:>8} {:>8}  {:>8} {:>8} {:>8} {:>8}"
    print("\n" + "="*120)
    print("SUMMARY (ID=HUMAN; OOD: C10 / C100 / TIN / SVHN)")
    print("="*120)
    print(hdr.format(
        "Model", "Acc",
        "AU-C10", "AU-C100", "AU-TIN", "AU-SVHN",
        "FPR-C10", "FPR-C100", "FPR-TIN", "FPR-SVHN"
    ))
    print("-"*120)

    def _ms(x): return f"{np.mean(x):.3f}±{np.std(x):.3f}"

    print(hdr.format(
        "Ensemble-5 (Teacher, deterministic)",
        _ms(teacher_accs),
        _ms(teacher_au_c10), _ms(teacher_au_c100), _ms(teacher_au_tin), _ms(teacher_au_svhn),
        _ms(teacher_fpr_c10), _ms(teacher_fpr_c100), _ms(teacher_fpr_tin), _ms(teacher_fpr_svhn)
    ))

    print(hdr.format(
        "ResNet-18 (Student)",
        _ms(sn_accs),
        _ms(sn_au_c10), _ms(sn_au_c100), _ms(sn_au_tin), _ms(sn_au_svhn),
        _ms(sn_fpr_c10), _ms(sn_fpr_c100), _ms(sn_fpr_tin), _ms(sn_fpr_svhn)
    ))

    print(hdr.format(
        "ResNet-18 (KD)",
        _ms(kd_accs),
        _ms(kd_au_c10), _ms(kd_au_c100), _ms(kd_au_tin), _ms(kd_au_svhn),
        _ms(kd_fpr_c10), _ms(kd_fpr_c100), _ms(kd_fpr_tin), _ms(kd_fpr_svhn)
    ))

    print(hdr.format(
        "ResNet-18 (PKT)",
        _ms(pkt_accs),
        _ms(pkt_au_c10), _ms(pkt_au_c100), _ms(pkt_au_tin), _ms(pkt_au_svhn),
        _ms(pkt_fpr_c10), _ms(pkt_fpr_c100), _ms(pkt_fpr_tin), _ms(pkt_fpr_svhn)
    ))

    def _fmt_params(a, b, t, a_t, b_t, t_t):
        return f"a={a:.2f},b={b:.2f},T={t:.1f} | a_t={a_t:.2f},b_t={b_t:.2f},T_t={t_t:.1f}"

    for key in sorted(kd_asl_accs_map.keys()):
        a, b, t, a_t, b_t, t_t = key
        label = "KD+ASL " + _fmt_params(a, b, t, a_t, b_t, t_t)

        print(hdr.format(
            label[:60],
            _ms(kd_asl_accs_map[key]),
            _ms(kd_asl_au_c10_map[key]), _ms(kd_asl_au_c100_map[key]),
            _ms(kd_asl_au_tin_map[key]), _ms(kd_asl_au_svhn_map[key]),
            _ms(kd_asl_fpr_c10_map[key]), _ms(kd_asl_fpr_c100_map[key]),
            _ms(kd_asl_fpr_tin_map[key]), _ms(kd_asl_fpr_svhn_map[key])
        ))
        print(" " * 2 + _fmt_params(a, b, t, a_t, b_t, t_t))

    print("="*120)
    print("Done.")

if __name__ == "__main__":
    main()
