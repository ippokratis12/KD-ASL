import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from config import *
from utils import set_global_seed, _seed_worker, mean_std
from models import resnet18_backbone, TeacherEnsemble, CAE2, General_Class_Network
from train import train_student_normal, train_student_kd, train_student_pkt, train_student_kd_asl, validate
from metrics import (auroc_entropy_single, auroc_entropy_ensemble,
                     auroc_kunc_ensemble, fpr95_entropy_single, fpr95_entropy_ensemble)
from data import get_datasets, make_loaders_for_seed, get_transforms

def main():
    device = DEVICE
    print("Device:", device)
    C = 10

    # ---------- Load OOD datasets only (once) -----------------------------
    _, _, ood_c100, ood_tin, ood_svhn, ood_human = get_datasets()

    # ---------- Initial datasets for teacher & AE pretraining (fixed seed) -
    tfm_tr10, tfm_te10, _ = get_transforms()
    tr_c10_pretrain = torchvision.datasets.CIFAR10("./data", True, download=True, transform=tfm_tr10)
    te_c10_pretrain = torchvision.datasets.CIFAR10("./data", False, download=True, transform=tfm_te10)

    g = torch.Generator()
    g.manual_seed(BASE_SEED)
    tr_loader_pretrain = torch.utils.data.DataLoader(
        tr_c10_pretrain, 128, True,
        num_workers=6, worker_init_fn=_seed_worker, generator=g, pin_memory=True
    )
    te_loader_pretrain = torch.utils.data.DataLoader(
        te_c10_pretrain, 128, False,
        num_workers=6, worker_init_fn=_seed_worker, generator=g, pin_memory=True
    )

    # ---------- metric containers -----------------------------------------
    teacher_accs = []; teacher_au_c100 = []; teacher_kunc_c100 = []; teacher_fpr_c100 = []
    teacher_au_tin = []; teacher_kunc_tin = []; teacher_fpr_tin = []
    teacher_au_svhn = []; teacher_kunc_svhn = []; teacher_fpr_svhn = []
    teacher_au_human = []; teacher_kunc_human = []; teacher_fpr_human = []

    sn_accs = []; sn_au_c100 = []; sn_au_tin = []; sn_au_svhn = []; sn_au_human = []
    sn_fpr_c100 = []; sn_fpr_tin = []; sn_fpr_svhn = []; sn_fpr_human = []
    kd_accs = []; kd_au_c100 = []; kd_au_tin = []; kd_au_svhn = []; kd_au_human = []
    kd_fpr_c100 = []; kd_fpr_tin = []; kd_fpr_svhn = []; kd_fpr_human = []
    pkt_accs = []; pkt_au_c100 = []; pkt_au_tin = []; pkt_au_svhn = []; pkt_au_human = []
    pkt_fpr_c100 = []; pkt_fpr_tin = []; pkt_fpr_svhn = []; pkt_fpr_human = []

    crit = nn.CrossEntropyLoss()

    # KD+ASL containers
    kd_asl_accs_map      = defaultdict(list)
    kd_asl_au_c100_map    = defaultdict(list)
    kd_asl_au_tin_map    = defaultdict(list)
    kd_asl_au_svhn_map   = defaultdict(list)
    kd_asl_au_human_map  = defaultdict(list)
    kd_asl_fpr_c100_map   = defaultdict(list)
    kd_asl_fpr_tin_map   = defaultdict(list)
    kd_asl_fpr_svhn_map  = defaultdict(list)
    kd_asl_fpr_human_map = defaultdict(list)

    # -------- Teacher ------------------------------------------------------
    print("=== TEACHER ENSEMBLE SETUP ===")
    teacher_ckpt = TEACHER_CKPT
    os.makedirs(os.path.dirname(teacher_ckpt), exist_ok=True)

    teacher = TeacherEnsemble(n=5, n_classes=C, base_seed=BASE_SEED).to(device)

    if os.path.exists(teacher_ckpt):
        print(f"Loading teacher ensemble weights from: {teacher_ckpt}")
        state = torch.load(teacher_ckpt, map_location=device)
        teacher.load_state_dict(state)
    else:
        print("No pretrained teacher found. Training teacher ensemble from scratch...")
        for i, net in enumerate(teacher.models, 1):
            print(f"[Teacher {i}/5] Training ResNet-18 on CIFAR-10 …")
            train_student_normal(net, tr_loader_pretrain, te_loader_pretrain,
                                 device=device, epochs=T_EPOCHS)
        torch.save(teacher.state_dict(), teacher_ckpt)
        print(f"Saved teacher ensemble weights to: {teacher_ckpt}")

    _, acc = validate(teacher, te_loader_pretrain, crit, device)
    teacher_accs.append(acc)

    # OOD evaluation for teacher
    teacher_au_c100.append(auroc_entropy_ensemble(te_c10_pretrain, ood_c100, teacher, device))
    teacher_kunc_c100.append(auroc_kunc_ensemble(te_c10_pretrain, ood_c100, teacher, device))
    teacher_fpr_c100.append(fpr95_entropy_ensemble(te_c10_pretrain, ood_c100, teacher, device))

    teacher_au_tin.append(auroc_entropy_ensemble(te_c10_pretrain, ood_tin, teacher, device))
    teacher_kunc_tin.append(auroc_kunc_ensemble(te_c10_pretrain, ood_tin, teacher, device))
    teacher_fpr_tin.append(fpr95_entropy_ensemble(te_c10_pretrain, ood_tin, teacher, device))

    teacher_au_svhn.append(auroc_entropy_ensemble(te_c10_pretrain, ood_svhn, teacher, device))
    teacher_kunc_svhn.append(auroc_kunc_ensemble(te_c10_pretrain, ood_svhn, teacher, device))
    teacher_fpr_svhn.append(fpr95_entropy_ensemble(te_c10_pretrain, ood_svhn, teacher, device))

    teacher_au_human.append(auroc_entropy_ensemble(te_c10_pretrain, ood_human, teacher, device))
    teacher_kunc_human.append(auroc_kunc_ensemble(te_c10_pretrain, ood_human, teacher, device))
    teacher_fpr_human.append(fpr95_entropy_ensemble(te_c10_pretrain, ood_human, teacher, device))

    # --------------------------------------------------------------------- #
    # AE Pretrain (once)                                                    #
    # --------------------------------------------------------------------- #
    print("[Autoencoder] Preparing CAE2(10) on CIFAR-10 …")
    lamda = 2.0

    ae_ckpt_full = AE_CKPT_FULL
    ae_ckpt_enc = AE_CKPT_ENCODER

    ae_base = CAE2(latent_dim=C).to(device)

    if os.path.exists(ae_ckpt_full):
        print(f"[AE] Found checkpoint: {ae_ckpt_full} -> loading weights.")
        state = torch.load(ae_ckpt_full, map_location=device)
        ae_base.load_state_dict(state)
    else:
        print("[AE] No checkpoint found. Pretraining CAE2(10) from scratch and saving weights …")
        opt_ae = torch.optim.Adam(ae_base.parameters(), lr=1e-4)
        mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        num_epochs = 100

        ae_base.train()
        for ep in range(num_epochs):
            run_loss = 0.0
            for x, y in tr_loader_pretrain:
                x, y = x.to(device), y.to(device)
                opt_ae.zero_grad()

                z = ae_base.encoder(x)
                x_hat = ae_base.decoder(z)

                loss = lamda * mse(x_hat, x) + ce(softmax(z), y)
                loss.backward()
                opt_ae.step()

                run_loss += loss.item()

            print(f"[AE] Epoch {ep + 1:03}/{num_epochs} | Loss {run_loss / len(tr_loader_pretrain):.4f}")

        ae_base.eval()
        torch.save(ae_base.state_dict(), ae_ckpt_full)
        torch.save(ae_base.encoder.state_dict(), ae_ckpt_enc)
        print(f"[AE] Saved full AE to {ae_ckpt_full} and encoder to {ae_ckpt_enc}")

    ae_base.eval()

    # =================== runs =============================================
    for run in range(N_RUNS):
        print(f"\n=== RUN {run+1}/{N_RUNS} ===")
        run_seed = BASE_SEED + run * 1000
        set_global_seed(run_seed)

        # Fresh CIFAR-10 datasets with the current run seed
        tfm_tr10, tfm_te10, _ = get_transforms()
        tr_c10 = torchvision.datasets.CIFAR10("./data", True, download=True, transform=tfm_tr10)
        te_c10 = torchvision.datasets.CIFAR10("./data", False, download=True, transform=tfm_te10)

        # Loaders for baseline / KD / PKT (no extra reset)
        g = torch.Generator().manual_seed(run_seed)
        tr_loader = torch.utils.data.DataLoader(
            tr_c10, 128, True,
            num_workers=6, worker_init_fn=_seed_worker, generator=g, pin_memory=True
        )
        te_loader = torch.utils.data.DataLoader(
            te_c10, 128, False,
            num_workers=6, worker_init_fn=_seed_worker, generator=g, pin_memory=True
        )

        # -------- Student baseline ----------------------------------------
        print("Training Student Baseline...")
        s = resnet18_backbone(n_classes=C)
        train_student_normal(s, tr_loader, te_loader, device=device, epochs=S_EPOCHS)
        _, acc = validate(s, te_loader, crit, device)
        sn_accs.append(acc)

        sn_au_c100.append(auroc_entropy_single(te_c10, ood_c100, s, device))
        sn_au_tin.append(auroc_entropy_single(te_c10, ood_tin, s, device))
        sn_au_svhn.append(auroc_entropy_single(te_c10, ood_svhn, s, device))
        sn_au_human.append(auroc_entropy_single(te_c10, ood_human, s, device))

        sn_fpr_c100.append(fpr95_entropy_single(te_c10, ood_c100, s, device))
        sn_fpr_tin.append(fpr95_entropy_single(te_c10, ood_tin, s, device))
        sn_fpr_svhn.append(fpr95_entropy_single(te_c10, ood_svhn, s, device))
        sn_fpr_human.append(fpr95_entropy_single(te_c10, ood_human, s, device))

        # -------- Student KD ----------------------------------------------
        print("Training Student with KD...")
        s = resnet18_backbone(n_classes=C)
        train_student_kd(s, teacher, tr_loader, te_loader,
                         device=device, epochs=S_EPOCHS,
                         T=T_KD, alpha=ALPHA_KD)
        _, acc = validate(s, te_loader, crit, device)
        kd_accs.append(acc)

        kd_au_c100.append(auroc_entropy_single(te_c10, ood_c100, s, device))
        kd_au_tin.append(auroc_entropy_single(te_c10, ood_tin, s, device))
        kd_au_svhn.append(auroc_entropy_single(te_c10, ood_svhn, s, device))
        kd_au_human.append(auroc_entropy_single(te_c10, ood_human, s, device))

        kd_fpr_c100.append(fpr95_entropy_single(te_c10, ood_c100, s, device))
        kd_fpr_tin.append(fpr95_entropy_single(te_c10, ood_tin, s, device))
        kd_fpr_svhn.append(fpr95_entropy_single(te_c10, ood_svhn, s, device))
        kd_fpr_human.append(fpr95_entropy_single(te_c10, ood_human, s, device))

        # -------- Student PKT ---------------------------------------------
        print("Training Student with PKT...")
        s = resnet18_backbone(n_classes=C)
        train_student_pkt(s, teacher, tr_loader, te_loader,
                          device=device, epochs=S_EPOCHS,
                          lamb=LAMBDA_PKT)
        _, acc = validate(s, te_loader, crit, device)
        pkt_accs.append(acc)

        pkt_au_c100.append(auroc_entropy_single(te_c10, ood_c100, s, device))
        pkt_au_tin.append(auroc_entropy_single(te_c10, ood_tin, s, device))
        pkt_au_svhn.append(auroc_entropy_single(te_c10, ood_svhn, s, device))
        pkt_au_human.append(auroc_entropy_single(te_c10, ood_human, s, device))

        pkt_fpr_c100.append(fpr95_entropy_single(te_c10, ood_c100, s, device))
        pkt_fpr_tin.append(fpr95_entropy_single(te_c10, ood_tin, s, device))
        pkt_fpr_svhn.append(fpr95_entropy_single(te_c10, ood_svhn, s, device))
        pkt_fpr_human.append(fpr95_entropy_single(te_c10, ood_human, s, device))

        # -------- KD+ASL grid ---------------------------------------------
        print("Training KD+ASL variants...")
        print("[Autoencoder] Using pretrained CAE2(10) weights for this run …")

        for a, b, t, a_t, b_t, t_t in GRID_PARAMS:
            print(f"[GRID KD+ASL] a={a},b={b},T={t}|a_t={a_t},b_t={b_t},T_t={t_t}")
            
            tr_loader, te_loader = make_loaders_for_seed(run_seed, tr_c10, te_c10, batch=128)

            gcn_s = General_Class_Network(input_size=C, output_size=C).to(device)
            gcn_t = General_Class_Network(input_size=C, output_size=C).to(device)

            ae = CAE2(latent_dim=C).to(device)
            ae_t = CAE2(latent_dim=C).to(device)

            ae.load_state_dict(ae_base.state_dict())
            ae_t.load_state_dict(ae_base.state_dict())

            s = resnet18_backbone(n_classes=C).to(device)

            train_student_kd_asl(
                s, teacher, ae, ae_t, gcn_s, gcn_t,
                tr_loader, te_loader,
                device=device, epochs=S_EPOCHS,
                T_kd=T_KD, alpha_kd=ALPHA_KD,
                alpha_gcn=a, beta_ae=b, T_ae=t,
                lam_rec=2.,
                alpha_gcn_t=a_t, beta_ae_t=b_t, T_ae_t=t_t
            )

            _, acc = validate(s, te_loader, crit, device)
            key = (a, b, t, a_t, b_t, t_t)

            kd_asl_accs_map[key].append(acc)
            kd_asl_au_c100_map[key].append(auroc_entropy_single(te_c10, ood_c100, s, device))
            kd_asl_au_tin_map[key].append(auroc_entropy_single(te_c10, ood_tin, s, device))
            kd_asl_au_svhn_map[key].append(auroc_entropy_single(te_c10, ood_svhn, s, device))
            kd_asl_au_human_map[key].append(auroc_entropy_single(te_c10, ood_human, s, device))

            kd_asl_fpr_c100_map[key].append(fpr95_entropy_single(te_c10, ood_c100, s, device))
            kd_asl_fpr_tin_map[key].append(fpr95_entropy_single(te_c10, ood_tin, s, device))
            kd_asl_fpr_svhn_map[key].append(fpr95_entropy_single(te_c10, ood_svhn, s, device))
            kd_asl_fpr_human_map[key].append(fpr95_entropy_single(te_c10, ood_human, s, device))

    # =================== summary ===========================================
    hdr = "{:<60} {:>7}  {:>8} {:>8} {:>8} {:>8}  {:>8} {:>8} {:>8} {:>8}"
    print("\n" + "="*120)
    print("SUMMARY (ID=CIFAR-10; OOD: C100 / TIN / SVHN / HUMAN)")
    print("="*120)
    print(hdr.format(
        "Model", "Acc",
        "AU-C100", "AU-TIN", "AU-SVHN", "AU-HUMAN",
        "FPR-C100", "FPR-TIN", "FPR-SVHN", "FPR-HUMAN"
    ))
    print("-"*120)

    def _ms(x): return f"{np.mean(x):.3f}±{np.std(x):.3f}"

    # Teacher row
    print(hdr.format(
        "Ensemble-5 (Teacher)",
        _ms(teacher_accs),
        _ms(teacher_au_c100), _ms(teacher_au_tin), _ms(teacher_au_svhn), _ms(teacher_au_human),
        _ms(teacher_fpr_c100), _ms(teacher_fpr_tin), _ms(teacher_fpr_svhn), _ms(teacher_fpr_human)
    ))

    # Student baseline row
    print(hdr.format(
        "ResNet-18 (Student)",
        _ms(sn_accs),
        _ms(sn_au_c100), _ms(sn_au_tin), _ms(sn_au_svhn), _ms(sn_au_human),
        _ms(sn_fpr_c100), _ms(sn_fpr_tin), _ms(sn_fpr_svhn), _ms(sn_fpr_human)
    ))

    # KD row
    print(hdr.format(
        "ResNet-18 (KD)",
        _ms(kd_accs),
        _ms(kd_au_c100), _ms(kd_au_tin), _ms(kd_au_svhn), _ms(kd_au_human),
        _ms(kd_fpr_c100), _ms(kd_fpr_tin), _ms(kd_fpr_svhn), _ms(kd_fpr_human)
    ))

    # PKT row
    print(hdr.format(
        "ResNet-18 (PKT)",
        _ms(pkt_accs),
        _ms(pkt_au_c100), _ms(pkt_au_tin), _ms(pkt_au_svhn), _ms(pkt_au_human),
        _ms(pkt_fpr_c100), _ms(pkt_fpr_tin), _ms(pkt_fpr_svhn), _ms(pkt_fpr_human)
    ))

    # KD+ASL rows
    def _fmt_params(a, b, t, a_t, b_t, t_t):
        return f"a={a:.2f},b={b:.2f},T={t:.1f} | a_t={a_t:.2f},b_t={b_t:.2f},T_t={t_t:.1f}"

    for key in sorted(kd_asl_accs_map.keys()):
        acc = kd_asl_accs_map[key]
        a, b, t, a_t, b_t, t_t = key
        label = "KD+ASL " + _fmt_params(a, b, t, a_t, b_t, t_t)

        print(hdr.format(
            label[:60],
            _ms(acc),
            _ms(kd_asl_au_c100_map[key]), _ms(kd_asl_au_tin_map[key]),
            _ms(kd_asl_au_svhn_map[key]), _ms(kd_asl_au_human_map[key]),
            _ms(kd_asl_fpr_c100_map[key]), _ms(kd_asl_fpr_tin_map[key]),
            _ms(kd_asl_fpr_svhn_map[key]), _ms(kd_asl_fpr_human_map[key])
        ))
        print(" " * 2 + _fmt_params(a, b, t, a_t, b_t, t_t))

    print("="*120)
    print("Done.")

if __name__ == "__main__":
    main()
