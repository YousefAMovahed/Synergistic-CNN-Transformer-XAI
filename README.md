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

The proposed architecture demonstrates superior performance across multiple datasets compared to state-of-the-art (SOTA) models.

| Dataset | Best Competing Model (SOTA) | Baseline (Hybrid Only) | **Proposed (Hybrid + XAI)** | Improvement over SOTA |
| :--- | :---: | :---: | :---: | :---: |
| **Alzheimer's Dataset 1** | 97.50% (Acc) / 90.15% (F1) <br> *(EfficientNet-B0)* | 99.11% (Acc) / 91.30% (F1) | **99.55% (Acc) / 92.48% (F1)** | **+2.33% (F1)** |
| **Alzheimer's Dataset 2** | 99.20% (Acc) / 94.80% (F1) <br> *(ResNet-50)* | 99.88% (Acc) / 95.51% (F1) | **99.94% (Acc) / 95.92% (F1)** | **+1.12% (F1)** |
| **RetinaMNIST** | 50.80% (Acc) / 33.20% (F1) <br> *(ResNet-50)* | 52.80% (Acc) / 34.49% (F1) | **54.30% (Acc) / 35.86% (F1)** | **+2.66% (F1)** |
| **PneumoniaMNIST** | 94.60% (Acc) / 86.80% (F1) <br> *(EfficientNet-B0)* | 95.70% (Acc) / 88.16% (F1) | **96.10% (Acc) / 88.29% (F1)** | **+1.49% (F1)** |
| **DermaMNIST** | 73.10% (Acc) / 42.50% (F1) <br> *(Swin-T)* | 75.30% (Acc) / 44.30% (F1) | **76.50% (Acc) / 45.41% (F1)** | **+2.91% (F1)** |

> *Note: The 'Best Competing Model' column refers to the highest-performing standard backbone (e.g., ResNet, EfficientNet, Swin) tested in our experiments. Metric shown as **Accuracy / Macro F1-Score**. Our XAI-guided correction consistently achieves the highest F1-scores, particularly in handling class imbalance.*

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
