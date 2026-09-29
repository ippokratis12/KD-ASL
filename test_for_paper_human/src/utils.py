# -*- coding: utf-8 -*-
# src/utils.py
from __future__ import annotations
import os
import numpy as np
import random
import torch

def seed_everything(seed: int):
    """Reproducible RNG state (Python/NumPy/PyTorch). Does NOT force deterministic ops."""
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
    # Keeps per-worker RNG reproducible across runs when using multiple workers
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _dash(x): return "–" if (x is None or x == "–") else x
def mean_std(a): return float(np.mean(a)), float(np.std(a))