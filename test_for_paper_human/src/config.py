# src/config.py
from __future__ import annotations
import torch

# Global settings
base_seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# Teacher ensemble
n_teachers = 5
t_epochs = 50

# Students
s_epochs = 50
n_runs = 5

# KD parameters
alpha_kd = 0.5
T_kd = 2.5

# PKT parameter
lambda_pkt = 1.0

# AE pretraining
ae_epochs = 100
lamda = 2.0
ae_lr = 1e-4

# KD+ASL grid parameters 
alpha_gcn_vals = [0.0]
beta_ae_vals = [0.0]
T_ae_vals = [5.0]
alpha_gcn_t_vals = [0.0]
beta_ae_t_vals = [0.0]
T_ae_t_vals = [5.0]

# Dataset paths
human_root = "./human detection dataset"
teacher_ckpt = "./checkpoints/teacher_human_ens5.pth"
ae_ckpt_full = "./checkpoints/ae_human_full.pth"
ae_ckpt_enc = "./checkpoints/ae_human_encoder.pth"
