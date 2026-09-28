import numpy as np
import random
import torch

# Some ops (e.g. certain CUDA kernels) don't support deterministic mode; ignore errors.
def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _dash(x):
    return "–" if (x is None or x == "–") else x

def mean_std(a):
    return float(np.mean(a)), float(np.std(a))