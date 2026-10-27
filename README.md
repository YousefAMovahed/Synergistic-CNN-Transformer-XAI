# SecondLook: Grad-CAM Consensus Review of Uncertain Predictions in a ConvNeXt–Swin Hybrid for Imbalanced Medical Image Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the manuscript *"SecondLook: Grad-CAM Consensus Review of Uncertain Predictions in a ConvNeXt–Swin Hybrid for Imbalanced Medical Image Classification"* (Y. Azizi Movahed, A. Jalili, H. Sajedi; University of Tehran).

The code implements a **two-phase pipeline**:

1. **Phase 1 – ConvNeXt–Swin hybrid.** ImageNet-pretrained `convnext_small` and `swin_small_patch4_window7_224` (timm) process the same 224×224 input. Their final 7×7×768 feature maps are projected to 49 tokens of dimension 384 and fused by **one cross-attention block** (Swin tokens = queries, ConvNeXt tokens = keys/values; 12 heads, FFN width 384, dropout 0.1). The pooled embedding feeds a linear classifier and a 384–384–128 projection head. Training objective: `0.5 · Focal (α=0.25, γ=2, label smoothing 0.1) + 0.5 · Supervised Contrastive (τ=0.07)`, AdamW (lr 5e-5, batch 16), no data augmentation. **Short budget** (5 epochs): the pretrained backbones are frozen and only the projections, the fusion block and the heads (≈1.7 M parameters) are trained; the epoch with the best validation balanced accuracy is kept. **Convergence protocol**: all parameters are trained for up to 100 epochs with early stopping on the validation loss.
2. **Phase 2 – SecondLook review layer (training-free).**
   * *Consensus masks:* for every correctly classified **validation** image, the Grad-CAM map (last depthwise convolution of the final ConvNeXt stage) is binarised at its `activation_percentile`-th percentile and dilated (3×3); masks are averaged per class and thresholded at `consensus_threshold`. A class mask is valid only if it has ≥ 5 contributing images and ≥ 100 pixels.
   * *Review rule:* a test prediction is reviewed if its top-1 probability `p1 < τ` **and its predicted class has a valid consensus mask**. The Grad-CAM mask of the predicted class is compared with every valid consensus mask by the **unnormalised pixel intersection** `S(x,c) = Σ M_x · C_c`. The label is switched to the second-ranked class **only if** that class also has the largest intersection (dual confirmation). Logits and attention are never modified.
   * *Thresholds:* masks are first built with the defaults (`activation_percentile` 95, `consensus_threshold` 0.90). If this gives at most one valid mask, Optuna (50 trials) tunes the three thresholds on the **validation** split and `τ = min(confidence_threshold, τ_err)`, where `τ_err` is the mean confidence of misclassified validation images. In the AD5 run the defaults gave two valid masks and `τ` was fixed at 0.6585 (`fixed_tau`).

> **Correction (release v1.1-secondlook).** Release v1.0-secondlook triggered review on `p1 < τ` alone and used a 50-pixel mask threshold. The archived notebook that produced the reported results also requires the predicted class to have a valid mask and uses 100 pixels; this rule is now the default (`require_valid_pred_mask: true`, `min_mask_pixels: 100`).

---

## Results reported in the manuscript

Balanced accuracy (BA) and macro F1 in %, held-out test splits. *Short budget* = 5 epochs, seed 42, single run; every model keeps its best validation epoch (Phase 1 by balanced accuracy, baselines by accuracy). Baselines: torchvision, fully fine-tuned with cross-entropy and Adam (lr 1e-3). Best baseline = strongest of ResNet-18/50, EfficientNetV2-S, Swin-T and VGG16 (scratch or ImageNet) on BA.

| Dataset | Best baseline (BA / F1) | Phase 1 (BA / F1) | Phase 1 + SecondLook (BA / F1) | SecondLook: corrected / harmed |
| :-- | :--: | :--: | :--: | :--: |
| OASIS (slice level, image-level split)\* | EfficientNetV2-S IN: 75.10 / 74.25 | 83.08 / 87.78 | **83.19 / 88.08** | 3 / 0 |
| AD5 (5-class Kaggle Alzheimer MRI) | EfficientNetV2-S IN: 86.83 / 74.61 | 91.30 / 79.90 | **92.48 / 85.95** | 60 / 2 (McNemar p = 8.5e-16) |
| BreastMNIST | **EfficientNetV2-S IN: 81.08 / 79.70** | 76.75 / 76.55 | 76.69 / 73.51 | 5 / 12 |
| RetinaMNIST | ResNet-50 scratch: 34.22 / 30.58 | 34.49 / 30.99 | **35.86 / 33.35** | 19 / 10 |
| PneumoniaMNIST† | **ResNet-18 IN: 91.58 / 92.19** | 88.16 / 89.24 | 88.29 / 89.40 | 2 / 1 |
| DermaMNIST† | EfficientNetV2-S IN: 40.64 / 44.65 | 44.30 / 47.72 | **45.41 / 48.45** | 8 / 5 |

\*OASIS slices of one participant can occur in both training and test splits, so OASIS numbers are slice-level, not subject-independent.

†Rule adapted after a first test run with the standard rule (PneumoniaMNIST: 5 corrected / 8 harmed; DermaMNIST: 8 / 11). PneumoniaMNIST: τ = 0.506 and the class proposed by the masks adopted directly; DermaMNIST: additional condition p2 > 0.20. Both runs are in `reproducibility/audit_tables/`.

**Exact controls** on the same reviewed images (`reproducibility/controls.py`): over the five audited datasets SecondLook changed the number of correct predictions by +64, a class gate by −90 and switching every reviewed prediction by −97. On AD5: +58 vs −42 (class gate) and −32 (no filter); a random filter of the same size harms ≤ 2 images with probability 4.8e-33.

**AD5 re-runs with the archived code** (5 seeds, τ = 0.6585): 58/2 with seed 42; net +41 to +77 in four seeds and −41 in one (Very Mild had no valid mask); total 254 corrected / 73 harmed. Label-shift corrections without test labels (EM prior re-weighting, balanced logit adjustment) lowered accuracy on every seed. On the validation split the rule lowered accuracy in every seed, because the AD5 test directory has very different class proportions (65 % Severe) from the training directory.

**Convergence protocol** (≤ 100 epochs, early stopping with patience 15, seeds 42/123/2026): the hybrid had the highest mean BA on AD5 (87.24%, tied with VGG16 and Swin-T within the between-seed spread) but was matched or beaten by single backbones on BreastMNIST, RetinaMNIST, PneumoniaMNIST and DermaMNIST. At convergence SecondLook changed at most 18 labels per run and gave no consistent benefit on MedMNIST; on three retrained AD5 checkpoints it made 14 corrections and no harmful change.

**Caveat (read before using SecondLook).** Because the mask score is an unnormalised intersection, accepted changes tend to flow into one "sink" class whose consensus mask is largest (e.g., all 62 AD5 changes went from Very Mild to Severe; 16 of 17 BreastMNIST changes went to Malignant). The effect is therefore dataset-dependent. Enable the layer only if, **on the validation split**, the net number of corrections and the change in balanced accuracy are both positive and at least two classes have valid masks. The pipeline is a research tool and is not clinically validated.

---

## Installation

```bash
git clone https://github.com/YousefAMovahed/Synergistic-CNN-Transformer-XAI.git
cd Synergistic-CNN-Transformer-XAI
pip install -r requirements.txt
```

## Data preparation

Use the standard `ImageFolder` layout (`data/train/<class>/...`, `data/val/<class>/...`, `data/test/<class>/...`). Official MedMNIST v2 splits were used unchanged; for AD5 the original training directory was split at random 80/20 into train/val (`random_split` after seeding with 42: 1,785 / 447 images) and the original test directory was kept; for OASIS a stratified 80/10/10 image-level split was built.

Datasets: OASIS-1 slice derivative (https://www.kaggle.com/datasets/ninadaithal/imagesoasis), AD5 (https://www.kaggle.com/datasets/phamnguyenduytien/alzheimer-datasets-5-classes), MedMNIST v2 (https://medmnist.com/).

## Usage

```bash
python train.py       # Phase 1: trains the hybrid, saves best_hybrid_model.pth (best validation balanced accuracy)
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
  fixed_tau: null                  # AD5 run: 0.6585
  min_mask_pixels: 100
  require_valid_pred_mask: true    # rule of the reported experiments
```

## Reproducibility material

* `reproducibility/audit_tables/` – per-image tables of every reviewed test image of the archived short-budget runs (predicted and second-ranked class with probabilities, class chosen by the masks, true label, logged outcome).
* `reproducibility/controls.py` – recomputes the exact controls of the paper from these tables.
* `reproducibility/kaggle/` – the Kaggle scripts and result files of the AD5 re-runs (short budget, five seeds; convergence protocol, three seeds), of the additional checks (IoU/Dice/coverage scores, Grad-CAM from the Swin branch and the fusion block, directional rule, temperature scaling with a top-2 switch, validation-tuned class bias, duplicate exclusion; BreastMNIST re-runs) and of the Grad-CAM sanity check on AD5 (deletion/insertion against random pixel order, parameter randomization).
* `reproducibility/kaggle/secondlook_ad5_round2.py` – threshold sweep (0.50–0.95) on the five AD5 re-runs and a comparison in which SecondLook and post-hoc corrections (class bias, temperature scaling with a top-2 switch, re-weighting, BCTS + EM) are tuned on half of the AD5 test directory and evaluated on the other half (200 splits). With labeled target data a tuned class bias gains more correct predictions, while SecondLook with its enablement rule keeps balanced accuracy with the smallest variation between splits (+0.48, SD 0.30, against −0.01, SD 1.14).
* `reproducibility/kaggle/secondlook_matched_heads.py` – matched comparison of cross-attention, late-fusion and single-backbone heads (combined loss and cross-entropy) on frozen ConvNeXt-Small and Swin-Small features, AD5 and four MedMNIST tasks, three seeds.
* `reproducibility/directional_control.py` – SecondLook against a fixed rule that switches only toward the class with the largest consensus mask, on the audit tables.
* `reproducibility/split_half_enablement.py` – the enablement rule evaluated with half of the AD5 test directory as validation data (exact enumeration).
* `reproducibility/ad5_duplicate_check.py` – pixel-level comparison of the AD5 training and test directories (102 test images duplicate training images; indices in `ad5_test_duplicate_indices.json`).
* `reproducibility/round3/` – third round of experiments:
  * `kaggle/secondlook_round3_ad5.py` – the hybrid, ConvNeXt-Small alone (same head and recipe), ResNet-50 and EfficientNetV2-S on the same five AD5 splits; saves probabilities and raw Grad-CAM maps (`outputs/ad5_arrays/`). The hybrid had the highest balanced accuracy of the four in every seed (89.20 vs 85.53 for ConvNeXt alone, paired t-test p = 0.008).
  * `analysis/analysis_r3_ad5.py` – offline controls computed from these arrays: masks permuted among classes (+181 → −324), masks rebuilt from shuffled labels (no change), masks moved, random pixels and centred discs of the same size; test set re-weighted to the training-directory class proportions; paired bootstrap intervals; target-half comparison with spreads; Monte Carlo dropout, selective prediction and conformal sets.
  * `kaggle/secondlook_round3_newdata.py` and `analysis/analysis_r3_newdata2.py` – OASIS-1 with a participant-level split (347 participants, CDR 0 / 0.5 / ≥1) and RetinaMNIST / DermaMNIST at 224×224; documented threshold search and cross-fitted enablement rule on validation data only.
  * `kaggle/secondlook_matched_heads_r3.py`, `kaggle/secondlook_heads_lr_{a,b}.py` – ten-seed matched comparison of heads, loss weights, linear probes and learning-rate-tuned heads (`outputs/`).

* `reproducibility/round4/` – fourth round: Phase 1 trained with cross-entropy and frozen ResNet-50 / EfficientNetV2-S backbones with the Phase-1 head on the same AD5 splits (the hybrid keeps the highest balanced accuracy in every seed); Phase 1 trained on part of the AD5 training directory and evaluated on held-out images of that directory, under synthetic intensity changes and on the test directory (the corrected Severe→Very Mild errors arise only on the test directory, where the gain recurs: +74, +89, +62, +36); one learning-rate grid for all heads and linear probes with self-attention capacity controls (cross-attention beats same-loss self-attention heads in 39 and 40 of 50 runs and late fusion in 35 of 50); late fusion with cross-entropy and self-attention heads on OASIS-1 and full-resolution MedMNIST; deep ensemble, label-shift-weighted conformal sets, participant-cluster bootstrap and age-matched evaluation on OASIS-1.

* `reproducibility/round5/` – fifth round: equal-area templates (shape and location only; the layer still gains in all five AD5 seeds, +58 in total), RetinaMNIST-224 → APTOS 2019 external test set (the cross-attention head stays the best of eight heads; the enablement rule keeps SecondLook off where the errors have no consistent direction), and training-time imbalance remedies (weighted cross-entropy, balanced sampling, logit-adjusted loss) for the cross-attention head on five datasets. The manuscript is split into a main text and a Supplementary Material (`analysis/split_supp.py`).

## Citation

```bibtex
@unpublished{AziziMovahed2026SecondLook,
  title  = {SecondLook: Auditable Grad-CAM-Guided Review of Uncertain Predictions in a ConvNeXt--Swin Hybrid under Class Imbalance and Dataset Shift},
  author = {Azizi Movahed, Yousef and Jalili, Amirreza and Sajedi, Hedieh},
  note   = {Manuscript, Department of Computer Science, University of Tehran},
  year   = {2026}
}
```

## Acknowledgments

This work was funded by the Iran National Science Foundation (INSF), project No. 4037260. OASIS-1 data were provided by OASIS: Cross-Sectional (PIs D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382). Libraries: `timm`, `pytorch-grad-cam`, `optuna`.
