# -*- coding: utf-8 -*-
# src/train.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from losses import kd_loss, kd_loss1, cosine_similarity_loss

# train/validate helpers                                                 
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


# training wrappers (SGD + StepLR)                                        
def _sgd_optimizer(params, lr=0.005, momentum=0.9, weight_decay=5e-4):
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

def _step_scheduler(opt):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.1)

# training_students
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



# KD+ASL training loop                                   

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