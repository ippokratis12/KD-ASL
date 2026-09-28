# KD-ASL: Trustworthy Uncertainty Estimation via Bidirectional Knowledge Distillation

Code for the paper:

> **Trustworthy Uncertainty Estimation via Bidirectional Knowledge Distillation**
> Ippokratis Raptis, Dimitrios Spanos, Nikolaos Passalis, Anastasios Tefas
> *Artificial Intelligence Applications and Innovations (AIAI 2026)*
> DOI: [10.1007/978-3-032-30805-4_22](https://doi.org/10.1007/978-3-032-30805-4_22)

## Structure

- `test_for_paper_cifar10/legacy/` — ID: CIFAR-10, OOD: CIFAR-100 / Tiny-ImageNet / SVHN / Human Detection
- `test_for_paper_human/legacy/` — ID: Human Detection, OOD: CIFAR-10 / CIFAR-100 / Tiny-ImageNet / SVHN

## Run

### CIFAR-10 experiment

```bash
cd test_for_paper_cifar10/legacy
python main.py
