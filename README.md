# KD-ASL: Trustworthy Uncertainty Estimation via Bidirectional Knowledge Distillation

Code for the paper:

> **Trustworthy Uncertainty Estimation via Bidirectional Knowledge Distillation**
> Ippokratis Raptis, Dimitrios Spanos, Nikolaos Passalis, Anastasios Tefas
> *Artificial Intelligence Applications and Innovations (AIAI 2026)*
> DOI: [10.1007/978-3-032-30805-4_22](https://doi.org/10.1007/978-3-032-30805-4_22)

---

## Structure

This repository contains two independent experiments:

- `test_for_paper_cifar10/` — ID: CIFAR-10, OOD: CIFAR-100 / Tiny-ImageNet / SVHN / Human Detection
- `test_for_paper_human/` — ID: Human Detection, OOD: CIFAR-10 / CIFAR-100 / Tiny-ImageNet / SVHN

Each experiment folder contains **two implementations** of the same method:

- `src/` — modular codebase (`config.py`, `data.py`, `models.py`, `losses.py`, `train.py`, `metrics.py`, `main.py`, `utils.py`).
- `legacy/` — an earlier monolithic version of the code, kept for reference.

Both implementations produce identical results when configured with the values reported in the paper.

---

## Setup

```bash
git clone https://github.com/ippokratis12/KD-ASL.git
cd KD-ASL
pip install -r requirements.txt
```

**Note:** The code was developed and run on a shared Linux server with Python 3.8, PyTorch 1.12.1 and torchvision 0.6.1. Any recent PyTorch installation should work. If you encounter version-specific issues, adjust `requirements.txt` accordingly.

---

## Data preparation

CIFAR-10, CIFAR-100 and SVHN are downloaded automatically on first run. For the remaining datasets, place them as follows:

```
test_for_paper_cifar10/
├── data/
│   ├── cifar-10-batches-py/        # auto-downloaded
│   ├── cifar-100-python/           # auto-downloaded
│   ├── svhn/                       # auto-downloaded
│   ├── tiny-imagenet-200/val/      # manual download
│   └── human detection dataset/    # manual download
└── src/

test_for_paper_human/
├── data/
│   ├── human detection dataset/    # manual download
│   ├── cifar-10-batches-py/        # auto-downloaded
│   ├── cifar-100-python/           # auto-downloaded
│   ├── svhn/                       # auto-downloaded
│   └── tiny-imagenet-200/val/      # manual download
└── src/
```

---

## Run

You can run each experiment in **two equivalent ways**. Both produce the same results if the hyperparameters match the values reported in the paper.

### Option A — Modular code (`src/`)

```bash
cd test_for_paper_cifar10/src
python main.py
```

```bash
cd test_for_paper_human/src
python main.py
```

### Option B — Monolithic code (`legacy/`)

```bash
cd test_for_paper_cifar10/legacy
python test_for_paper_cifar10.py
```

```bash
cd test_for_paper_human/legacy
python test_for_paper_human.py
```

Each script trains (or loads) the teacher ensemble and the autoencoder, then trains all student variants (baseline, KD, PKT, KD+ASL) and prints a summary table with accuracy, AUROC and FPR@95 for all OOD datasets.

---

## Reproducibility

To reproduce the exact numbers reported in the paper, you **must** set the hyperparameters to the values listed in the paper (Section *Experimental Setup*).

- **Modular version (`src/`)** — edit `config.py`:
  - `t_epochs`, `s_epochs`, `n_runs`
  - `alpha_kd`, `T_kd`, `lambda_pkt`
  - `ae_epochs`, `lamda`, `ae_lr`
  - the KD+ASL grid (`alpha_gcn_vals`, `beta_ae_vals`, `T_ae_vals`, `alpha_gcn_t_vals`, `beta_ae_t_vals`, `T_ae_t_vals`)

- **Legacy version (`legacy/`)** — the same values appear as constants at the top of the script.

With the paper's values, both implementations produce the same results.

---

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{raptis2026trustworthy,
  title     = {Trustworthy Uncertainty Estimation via Bidirectional Knowledge Distillation},
  author    = {Raptis, Ippokratis and Spanos, Dimitrios and Passalis, Nikolaos and Tefas, Anastasios},
  booktitle = {Artificial Intelligence Applications and Innovations (AIAI 2026)},
  year      = {2026},
  publisher = {Springer},
  doi       = {10.1007/978-3-032-30805-4_22}
}
```

---

## Acknowledgments

This paper has received partial funding from the Hellenic Foundation for Research & Innovation (H.F.R.I.) scholarship under grant agreement No 20490 (Deep Learning Methodologies for Trustworthy Intelligent Systems). This paper has been partially supported by the research project "Robotic Safe Adaptation In Unprecedented Situations (RoboSAPIENS)", which is implemented in the framework of Horizon Europe 2021–2027 research and innovation programme under grant agreement No 101133807. This publication reflects the authors' views only. The H.F.R.I. and European Commission are not responsible for any use that may be made of the information it contains.

<p align="center">
  <img src="assets/eu-flag.png" alt="Funded by the European Union" width="180">
</p>

<p align="center">
  <b>Funded by<br>the European Union</b>
</p>

<p align="center">
  Learn more about <a href="https://robosapiens.eu/">RoboSAPIENS</a>.
</p>

<p align="center">
  <img src="assets/robosapiens.png" alt="RoboSAPIENS" width="120">
</p>
