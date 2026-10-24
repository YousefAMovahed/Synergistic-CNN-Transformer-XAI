# SecondLook: Grad-CAM Consensus Review of Uncertain Predictions in a ConvNeXt–Swin Hybrid for Imbalanced Medical Image Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the manuscript *"SecondLook: Grad-CAM Consensus Review of Uncertain Predictions in a ConvNeXt–Swin Hybrid for Imbalanced Medical Image Classification"* (Y. Azizimovahed, A. Jalili, H. Sajedi; University of Tehran).

The code implements a **two-phase pipeline**:

1. **Phase 1 – ConvNeXt–Swin hybrid.** ImageNet-pretrained `convnext_small` and `swin_small_patch4_window7_224` (timm) process the same 224×224 input. Their final 7×7×768 feature maps are projected to 49 tokens of dimension 384 and fused by **one cross-attention block** (Swin tokens = queries, ConvNeXt tokens = keys/values; 12 heads, FFN width 384, dropout 0.1). The pooled embedding feeds a linear classifier and a 384–384–128 projection head. Training objective: `0.5 · Focal (α=0.25, γ=2, label smoothing 0.1) + 0.5 · Supervised Contrastive (τ=0.07)`; all parameters are trained jointly with AdamW (lr 5e-5, batch 16). No data augmentation is used.
2. **Phase 2 – SecondLook review layer (training-free).**
   * *Consensus masks:* for every correctly classified **validation** image, the Grad-CAM map (last depthwise convolution of the final ConvNeXt stage) is binarised at its `activation_percentile`-th percentile and dilated (3×3); masks are averaged per class and thresholded at `consensus_threshold`. A class mask is valid only if it has ≥ 5 contributing images and > 50 pixels.
   * *Review rule:* a test prediction with top-1 probability `p1 < τ`, where `τ = min(confidence_threshold, τ_err)` and `τ_err` is the mean confidence of misclassified validation images, is reviewed. The Grad-CAM mask of the predicted class is compared with every valid consensus mask by the **unnormalised pixel intersection** `S(x,c) = Σ M_x · C_c`. The label is switched to the second-ranked class **only if** that class also has the largest intersection (dual confirmation). Logits and attention are never modified.
   * Thresholds are tuned with Optuna (50 trials) by maximising **validation** balanced accuracy; the test split is used once.

> **Rule used in the reported experiments.** Review is triggered by `p1 < τ` alone. An earlier version of this repository additionally required the predicted class to have a valid consensus mask; this stricter variant is still available as `require_valid_pred_mask: true` but is **off by default** so that the code reproduces the manuscript.

---

## Results reported in the manuscript

Balanced accuracy (BA) and macro F1 in %, held-out test splits. *Short budget* = 5 epochs, seed 42, single run (Phase 1 keeps the epoch with the best validation accuracy, baselines the fifth epoch). Best baseline = strongest of ResNet-18/50, EfficientNetV2-S, Swin-T and VGG16 (scratch or ImageNet) on BA.

| Dataset | Best baseline (BA / F1) | Phase 1 (BA / F1) | Phase 1 + SecondLook (BA / F1) | SecondLook: corrected / harmed |
| :-- | :--: | :--: | :--: | :--: |
| OASIS (slice level, image-level split)\* | EfficientNetV2-S IN: 75.10 / 74.25 | 83.08 / 87.78 | **83.19 / 88.08** | 3 / 0 |
| AD5 (5-class Kaggle Alzheimer MRI) | EfficientNetV2-S IN: 86.83 / 74.61 | 91.30 / 79.90 | **92.48 / 85.95** | 60 / 2 (McNemar p = 8.5e-16) |
| BreastMNIST | **EfficientNetV2-S IN: 81.08 / 79.70** | 76.75 / 76.55 | 76.69 / 73.51 | 5 / 12 |
| RetinaMNIST | ResNet-50 scratch: 34.22 / 30.58 | 34.49 / 30.99 | **35.86 / 33.35** | 20 / 11 |
| PneumoniaMNIST | **ResNet-18 IN: 91.58 / 91.19** | 88.16 / 89.24 | 88.29 / 89.40 | 2 / 1 |
| DermaMNIST | EfficientNetV2-S IN: 40.64 / 44.65 | 44.30 / 47.72 | **45.41 / 48.45** | 8 / 5 |

\*OASIS slices of one participant can occur in both training and test splits, so OASIS numbers are slice-level, not subject-independent.

**Convergence protocol** (≤ 100 epochs, early stopping with patience 15, seeds 42/123/2026): the hybrid had the highest mean BA on AD5 (87.24%, tied with VGG16 and Swin-T within the between-seed spread) but was matched or beaten by single backbones on BreastMNIST, RetinaMNIST, PneumoniaMNIST and DermaMNIST. At convergence SecondLook changed at most 18 labels per run and gave no consistent benefit.

**Caveat (read before using SecondLook).** Because the mask score is an unnormalised intersection, accepted changes tend to flow into one "sink" class whose consensus mask is largest (e.g., all 62 AD5 changes went from Very Mild to Severe; 16 of 17 BreastMNIST changes went to Malignant). The effect is therefore dataset-dependent. Enable the layer only if, **on the validation split**, the net number of corrections and the change in balanced accuracy are both positive and at least two classes have valid masks. The pipeline is a research tool and is not clinically validated.

---

## Installation

```bash
git clone https://github.com/YousefAMovahed/Synergistic-CNN-Transformer-XAI.git
cd Synergistic-CNN-Transformer-XAI
pip install -r requirements.txt
```

## Data preparation

Use the standard `ImageFolder` layout (`data/train/<class>/...`, `data/val/<class>/...`, `data/test/<class>/...`). Official MedMNIST v2 splits were used unchanged; for AD5 the original training directory was split 80/20 into train/val and the original test directory was kept; for OASIS a stratified 80/10/10 image-level split was built.

Datasets: OASIS-1 slice derivative (https://www.kaggle.com/datasets/ninadaithal/imagesoasis), AD5 (https://www.kaggle.com/datasets/phamnguyenduytien/alzheimer-datasets-5-classes), MedMNIST v2 (https://medmnist.com/).

## Usage

```bash
python train.py       # Phase 1: trains the hybrid, saves best_hybrid_model.pth (best validation accuracy)
python optimize.py    # optional: Optuna search of the SecondLook thresholds on the validation split
python inference.py   # Phase 2: consensus masks (validation) + SecondLook review (test)
```

`inference.py` prints the operating threshold, the number of reviewed images, corrected (b), harmed (c), wrong-to-wrong and unchanged counts, the exact McNemar p-value, and the classification report. It also writes a per-image audit file (`results/secondlook_audit.csv`) with the true label, top-two classes, p1, whether review was triggered, the mask scores S(x,c), the mask-proposed class, the final label and the outcome, so that every change can be audited.

Key settings in `configs/config.yaml`:

```yaml
inference:
  activation_percentile: 95        # p_act
  consensus_threshold: 0.90        # tau_cons
  confidence_threshold: 0.70       # tau_conf
  use_error_threshold: true        # tau = min(tau_conf, tau_err)
  require_valid_pred_mask: false   # stricter variant, not used in the paper
```

## Citation

```bibtex
@unpublished{Azizimovahed2026SecondLook,
  title  = {SecondLook: Grad-CAM Consensus Review of Uncertain Predictions in a ConvNeXt--Swin Hybrid for Imbalanced Medical Image Classification},
  author = {Azizimovahed, Yousef and Jalili, Amirreza and Sajedi, Hedieh},
  note   = {Manuscript, Department of Computer Science, University of Tehran},
  year   = {2026}
}
```

## Acknowledgments

This work was funded by the Iran National Science Foundation (INSF), project No. 4037260. OASIS-1 data were provided by OASIS: Cross-Sectional (PIs D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382). Libraries: `timm`, `pytorch-grad-cam`, `optuna`.
