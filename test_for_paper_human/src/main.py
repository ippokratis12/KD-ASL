# -*- coding: utf-8 -*-
# src/main.py
from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import product
from collections import defaultdict

from utils import (
    seed_everything,
    set_determinism,
    seed_and_optionally_deterministic,
    _seed_worker,
    _dash,
    mean_std
)
from models import (
    ResNet18Backbone,
    TeacherEnsemble,
    resnet18_backbone,
    CAE2,
    General_Class_Network
)
from losses import kd_loss, kd_loss1, cosine_similarity_loss
from train import (
    train_one_epoch,
    validate,
    train_student_normal,
    train_student_kd,
    train_student_pkt,
    train_student_kd_asl
)
from metrics import (
    auroc_entropy_single,
    auroc_entropy_ensemble,
    auroc_kunc_ensemble,
    fpr95_entropy_single,
    fpr95_entropy_ensemble
)
from data import (
    mean, std, tfm_tr, tfm_te, tfm_ood,
    get_human_split,
    load_ood_datasets,
    make_loaders_for_seed
)
import config

# -------- Main experiment ----------------------------------------                                                             
def main():
    # Use config values
    base_seed = config.base_seed
    device = config.device
    C = 2
    t_epochs = config.t_epochs
    s_epochs = config.s_epochs

    alpha_kd = config.alpha_kd
    T_kd = config.T_kd
    lambda_pkt = config.lambda_pkt
    n_runs = config.n_runs

    human_root = config.human_root
    teacher_ckpt = config.teacher_ckpt
    ae_ckpt_full = config.ae_ckpt_full
    ae_ckpt_enc = config.ae_ckpt_enc

    print("Device:", device)

    # Datasets
    # Fixed 80/20 split for teacher (base_seed)
    tr_human_teacher, te_human_teacher = get_human_split(
        human_root, base_seed, tfm_tr, tfm_te
    )

    # OOD datasets
    ood_c10, ood_c100, ood_tin, ood_svhn = load_ood_datasets(tfm_ood)

    crit = nn.CrossEntropyLoss()

    
    # -------- TEACHER ----------------------------------------                                          
    
    print("=== TEACHER ENSEMBLE SETUP (DETERMINISTIC) ===")
    seed_and_optionally_deterministic(base_seed, deterministic=True)

    gT = torch.Generator().manual_seed(base_seed)
    tr_loader_T = DataLoader(
        tr_human_teacher, 64, True,
        num_workers=6, worker_init_fn=_seed_worker, generator=gT, pin_memory=True
    )
    te_loader_T = DataLoader(
        te_human_teacher, 64, False,
        num_workers=6, worker_init_fn=_seed_worker, generator=gT, pin_memory=True
    )

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

    # teacher metrics (deterministic as well)
    teacher_accs = []
    teacher_au_c10 = []; teacher_kunc_c10 = []; teacher_fpr_c10 = []
    teacher_au_c100 = []; teacher_kunc_c100 = []; teacher_fpr_c100 = []
    teacher_au_tin = []; teacher_kunc_tin = []; teacher_fpr_tin = []
    teacher_au_svhn = []; teacher_kunc_svhn = []; teacher_fpr_svhn = []

    _, acc = validate(teacher, te_loader_T, crit, device)
    teacher_accs.append(acc)

    teacher_au_c10.append(auroc_entropy_ensemble(te_human_teacher, ood_c10, teacher, device))
    teacher_kunc_c10.append(auroc_kunc_ensemble(te_human_teacher, ood_c10, teacher, device))
    teacher_fpr_c10.append(fpr95_entropy_ensemble(te_human_teacher, ood_c10, teacher, device))

    teacher_au_c100.append(auroc_entropy_ensemble(te_human_teacher, ood_c100, teacher, device))
    teacher_kunc_c100.append(auroc_kunc_ensemble(te_human_teacher, ood_c100, teacher, device))
    teacher_fpr_c100.append(fpr95_entropy_ensemble(te_human_teacher, ood_c100, teacher, device))

    teacher_au_tin.append(auroc_entropy_ensemble(te_human_teacher, ood_tin, teacher, device))
    teacher_kunc_tin.append(auroc_kunc_ensemble(te_human_teacher, ood_tin, teacher, device))
    teacher_fpr_tin.append(fpr95_entropy_ensemble(te_human_teacher, ood_tin, teacher, device))

    teacher_au_svhn.append(auroc_entropy_ensemble(te_human_teacher, ood_svhn, teacher, device))
    teacher_kunc_svhn.append(auroc_kunc_ensemble(te_human_teacher, ood_svhn, teacher, device))
    teacher_fpr_svhn.append(fpr95_entropy_ensemble(te_human_teacher, ood_svhn, teacher, device))

    
    # -------- STUDENTS ----------------------------------------        
    
    print("\n=== SWITCHING TO STUDENT MODE (seeded, non-deterministic ops) ===")
    set_determinism(False)

    
    # AE Pretrain                                                     
    
    print("[Autoencoder] Preparing CAE2(C) on HUMAN(ID) …")
    seed_everything(base_seed)

    lamda = config.lamda
    ae_epochs = config.ae_epochs

    ae_base = CAE2(latent_dim=C).to(device)

    if os.path.exists(ae_ckpt_full):
        print(f"[AE] Found checkpoint: {ae_ckpt_full} -> loading weights.")
        state = torch.load(ae_ckpt_full, map_location=device)
        ae_base.load_state_dict(state)
    else:
        print("[AE] No checkpoint found. Pretraining CAE2(C) from scratch and saving weights …")

        opt_ae = torch.optim.Adam(ae_base.parameters(), lr=config.ae_lr)
        mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)

        ae_base.train()
        for ep in range(ae_epochs):
            run_loss = 0.0
            for x, y in tr_loader_T:   # using teacher loader for pretraining
                x, y = x.to(device), y.to(device)
                opt_ae.zero_grad()

                z = ae_base.encoder(x)
                x_hat = ae_base.decoder(z)

                loss = lamda * mse(x_hat, x) + ce(softmax(z), y)
                loss.backward()
                opt_ae.step()

                run_loss += loss.item()

            print(f"[AE] Epoch {ep + 1:03}/{ae_epochs} | Loss {run_loss / len(tr_loader_T):.4f}")

        ae_base.eval()
        torch.save(ae_base.state_dict(), ae_ckpt_full)
        torch.save(ae_base.encoder.state_dict(), ae_ckpt_enc)
        print(f"[AE] Saved full AE to {ae_ckpt_full} and encoder to {ae_ckpt_enc}")

    ae_base.eval()

    # metric containers (students) 
    sn_accs = []; sn_au_c10 = []; sn_au_c100 = []; sn_au_tin = []; sn_au_svhn = []
    sn_fpr_c10 = []; sn_fpr_c100 = []; sn_fpr_tin = []; sn_fpr_svhn = []

    kd_accs = []; kd_au_c10 = []; kd_au_c100 = []; kd_au_tin = []; kd_au_svhn = []
    kd_fpr_c10 = []; kd_fpr_c100 = []; kd_fpr_tin = []; kd_fpr_svhn = []

    pkt_accs = []; pkt_au_c10 = []; pkt_au_c100 = []; pkt_au_tin = []; pkt_au_svhn = []
    pkt_fpr_c10 = []; pkt_fpr_c100 = []; pkt_fpr_tin = []; pkt_fpr_svhn = []

    # KD-ASL grid parameters 
    alpha_gcn_vals = config.alpha_gcn_vals
    beta_ae_vals = config.beta_ae_vals
    T_ae_vals = config.T_ae_vals
    alpha_gcn_t_vals = config.alpha_gcn_t_vals
    beta_ae_t_vals = config.beta_ae_t_vals
    T_ae_t_vals = config.T_ae_t_vals

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

        # per-run split (train/test transforms + index split)
        tr_human_run, te_human_run = get_human_split(
            human_root, run_seed, tfm_tr, tfm_te
        )

        # loaders: reproducible shuffle/order by run_seed
        tr_loader, te_loader = make_loaders_for_seed(run_seed, tr_human_run, te_human_run, batch=64)

        # -------- Student baseline ----------------------------------------
        print("Training Student Baseline...")
        seed_everything(run_seed)
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

            # build models AFTER seed reset (reproducible init per config)
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
