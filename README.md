# A Synergistic CNN-Transformer Architecture with XAI-Guided Correction for Robust Multi-Class Medical Image Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://doi.org/YOUR_DOI_HERE)

> **Official implementation** of the paper: *"A Synergistic CNN-Transformer Architecture with XAI-Guided Correction for Robust Multi-Class Medical Image Classification"*.

This repository provides a robust framework for medical image classification, particularly effective for imbalanced multi-class datasets (e.g., Alzheimer's MRI, Skin Lesions). It combines a hybrid **ConvNeXt + Swin Transformer** backbone with an **XAI-guided Intelligent Inference** layer that corrects low-confidence predictions using Grad-CAM consensus masks.

---

## 🧩 Architecture Overview

The framework consists of three main phases:

1. **Phase 1: Hybrid Representation Learning**

   * **Backbone:** Parallel branches of **ConvNeXt** (Local features) and **Swin Transformer** (Global context).
   * **Fusion:** A **Cross-Attention Fusion Module** merges features, using Swin as *Query* and ConvNeXt as *Key/Value*.
   * **Training Objective:** A composite loss function: `Focal Loss` (for imbalance) + `Supervised Contrastive Loss` (for feature discrimination).

2. **Phase 2: XAI-Guided Consensus Mask Generation**

   * For correctly classified validation samples, **Grad-CAM** heatmaps are generated.
   * These are aggregated to create **Consensus Masks**—prototypical visual patterns for each class.

3. **Phase 3: Intelligent Inference (Test Time)**

   * Standard prediction is performed.
   * **Intervention Logic:** If confidence is low, the system compares the test image's Grad-CAM against the Consensus Masks (using IoU).
   * **Dual Confirmation:** A correction is applied *only* if visual similarity aligns with the model's second-best probabilistic guess.

---

## 📊 Results

The model was evaluated on **Alzheimer's MRI datasets** and **MedMNIST v2** benchmarks.

| Dataset                   | Metric        | Baseline (Hybrid) | **Proposed (Hybrid + XAI)** |
| :------------------------ | :------------ | :---------------: | :-------------------------: |
| **Alzheimer's Dataset 1** | Balanced Acc. |       91.30%      |          **92.48%**         |
| **Alzheimer's Dataset 2** | Balanced Acc. |       95.51%      |          **95.92%**         |
| **RetinaMNIST**           | Balanced Acc. |       34.49%      |          **35.86%**         |
| **PneumoniaMNIST**        | Balanced Acc. |       88.16%      |          **88.29%**         |
| **DermaMNIST**            | Balanced Acc. |       44.30%      |          **45.41%**         |

> *Note: The proposed method consistently outperforms standard backbones like ResNet-50, EfficientNet-V2, and Swin-T, especially in balanced accuracy metrics.*

---

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone [https://github.com/YourUsername/Synergistic-CNN-Transformer-XAI.git](https://github.com/YourUsername/Synergistic-CNN-Transformer-XAI.git)
cd Synergistic-CNN-Transformer-XAI
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Data Preparation

Organize your dataset in the standard `ImageFolder` structure:

```text
data/
├── train/
│   ├── class_1/
│   └── class_2/
├── val/
│   ├── class_1/
│   └── class_2/
└── test/
    ├── class_1/
    └── class_2/
```

### 2. Configuration

Modify `configs/config.yaml` to set your dataset path and hyperparameters:

```yaml
dataset_name: "Alzheimer5Class"
data_dir: "./data"
batch_size: 16
learning_rate: 5e-5
alpha: 0.5  # Weight for Focal Loss vs Contrastive Loss
```

### 3. Training & Inference

To run the full pipeline (Training -> Mask Generation -> Intelligent Inference):

```bash
python main.py
```

Or run specific phases programmatically:

```python
# Example logic in main.py
from train import train_phase1
from inference import run_intelligent_inference

# Phase 1
model = train_phase1(config)

# Phase 2 & 3 (Auto-tunes thresholds with Optuna if needed)
run_intelligent_inference(model, test_loader, config)
```

---

## 🔍 Hyperparameter Optimization

The system includes an automated tuning module using **Optuna**. If the manual thresholds for Grad-CAM masks do not yield sufficient quality masks on the validation set, Optuna searches for the optimal:

* `CONSENSUS_THRESHOLD`
* `ACTIVATION_PERCENTILE`
* `CONFIDENCE_THRESHOLD`

---

## 📜 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{Azizimovahed2025Synergistic,
  title={A Synergistic CNN-Transformer Architecture with XAI-Guided Correction for Robust Multi-Class Medical Image Classification},
  author={Azizimovahed, Yousef and Jalili, Amirreza and Sajedi, Hedieh},
  journal={Department of Computer Science, University of Tehran},
  year={2025}
}
```

---

## 🤝 Acknowledgments

* **Dataset sources:** Kaggle Alzheimer's MRI, MedMNIST v2.
* **Libraries used:** `timm`, `pytorch-grad-cam`, `optuna`.

<!-- end list -->
