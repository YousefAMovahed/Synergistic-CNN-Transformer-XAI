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

The proposed architecture was evaluated against state-of-the-art (SOTA) benchmarks including ResNet, EfficientNet, Swin Transformer, VGG, and YOLOv8. The table below compares the **Balanced Accuracy** and **Macro F1-Score** of our method against the strongest performing competitor for each dataset.

| Dataset | Best Competitor (Model) | Best Competitor (Bal. Acc / F1) | Baseline (Ours) (Bal. Acc / F1) | **Proposed (Hybrid + XAI)** (Bal. Acc / F1) |
| :--- | :---: | :---: | :---: | :---: |
| **Alzheimer's Dataset 1** | EfficientNet-v2-s (Pretrained) | 86.83% / 74.61% | 91.30% / 79.90% | **92.48% / 85.95%** |
| **Alzheimer's Dataset 2** | ResNet-18 (Scratch) | 95.42% / 93.93% | 95.51% / 93.47% | **95.92% / 94.02%** |
| **RetinaMNIST** | ResNet-50 (Scratch) | 34.22% / 30.58% | 34.49% / 30.99% | **35.86% / 33.35%** |
| **PneumoniaMNIST** | ResNet-18 (Pretrained) | **91.58%** / **91.19%** | 88.16% / 89.24% | 88.29% / 89.40% |
| **DermaMNIST** | YOLOv8s-cls (Pretrained) | **50.78%** / 44.44% | 44.30% / 47.72% | 45.41% / **48.45%** |

> **Key Observations:**
> * **Significant Improvement:** On complex, multi-class tasks like *Alzheimer's Dataset 1*, our proposed method improves Balanced Accuracy by **+5.65%** and F1-Score by **+11.34%** compared to the best competitor (EfficientNet-v2-s).
> * **Robustness:** Even in cases where competitors like YOLOv8 achieve higher raw accuracy (e.g., DermaMNIST), our XAI-guided model achieves a superior **Macro F1-Score (+4.01%)**, indicating better handling of minority classes in imbalanced data.

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
