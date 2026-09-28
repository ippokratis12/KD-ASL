import os

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
BASE_SEED = 42
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
C = 10  # CIFAR-10

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
T_EPOCHS = 45
S_EPOCHS = 90
ALPHA_KD = 0.5
T_KD = 2.5
LAMBDA_PKT = 1.0
N_RUNS = 5

# ---------------------------------------------------------------------------
# Dataset / paths
# ---------------------------------------------------------------------------
DATA_DIR = "./data"
TINY_IMAGENET_VAL = "./tiny-imagenet-200/val"
HUMAN_DETECTION_DIR = "./human detection dataset"

TEACHER_CKPT = "./teacher_c10_ens5.pth"
AE_CKPT_FULL = "./ae_c10_ae_full.pth"
AE_CKPT_ENCODER = "./ae_c10_ae_encoder.pth"

# ---------------------------------------------------------------------------
# KD+ASL grid (list of tuples)
# ---------------------------------------------------------------------------

# Each tuple: (alpha_gcn, beta_ae, T_ae, alpha_gcn_t, beta_ae_t, T_ae_t)
GRID_PARAMS = [
    
    (0.00, 0.00, 5.0, 0.00, 0.00, 5.0),
]
