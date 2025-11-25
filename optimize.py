import torch
import optuna
import yaml
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import warnings

# Custom Imports
from models.hybrid_model import HybridVisionModel
# Import logic from inference.py to avoid code duplication
from inference import create_consensus_masks, run_intelligent_inference

warnings.filterwarnings("ignore")

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f: return yaml.safe_load(f)

def objective(trial, model, loader, device, classes):
    """
    Optuna objective function:
    1. Suggests hyperparameters.
    2. Generates masks.
    3. Runs inference.
    4. Returns Balanced Accuracy.
    """
    # 1. Suggest parameters to try
    cfg_trial = {
        'consensus_threshold': trial.suggest_float('consensus_threshold', 0.4, 0.95, step=0.05),
        'activation_percentile': trial.suggest_int('activation_percentile', 85, 99),
        'confidence_threshold': trial.suggest_float('confidence_threshold', 0.5, 0.95, step=0.05),
        'dilation_size': 3 # Fixed for now
    }

    # 2. Phase 2: Generate Masks with trial params
    # We pass verbose=False to keep the output clean during optimization
    consensus_masks, cam = create_consensus_masks(model, loader, device, cfg_trial, verbose=False)

    # Pruning: If params are too strict and no masks are generated, return 0 score
    if not consensus_masks:
        return 0.0

    # 3. Phase 3: Run Inference (using validation set as proxy for tuning)
    y_true, y_pred, _ = run_intelligent_inference(
        model, loader, consensus_masks, cam, device, cfg_trial, classes
    )

    # 4. Calculate Score (Balanced Accuracy)
    from sklearn.metrics import balanced_accuracy_score
    score = balanced_accuracy_score(y_true, y_pred)
    
    return score

if __name__ == "__main__":
    cfg = load_config()
    device = cfg['device']
    print(f"🔬 Starting Hyperparameter Optimization with Optuna ({cfg['inference']['optuna_trials']} trials)...")

    # Load Data (We use Validation set for finding best params)
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Ensure validation data exists
    val_path = os.path.join(cfg['data']['data_dir'], 'val')
    if not os.path.exists(val_path):
        print(f"❌ Validation folder not found at {val_path}")
        exit()
        
    val_dataset = datasets.ImageFolder(val_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False)
    
    # Load Model Architecture
    num_classes = len(val_dataset.classes)
    model = HybridVisionModel(
        cfg['model']['cnn_backbone'], cfg['model']['swin_backbone'], num_classes,
        cfg['model']['embed_dim'], cfg['model']['num_heads'], cfg['model']['ff_dim']
    ).to(device)
    
    # Load Trained Weights
    if os.path.exists("best_hybrid_model.pth"):
        model.load_state_dict(torch.load("best_hybrid_model.pth", map_location=device))
        print("✅ Loaded trained weights.")
    else:
        print("⚠️ Warning: 'best_hybrid_model.pth' not found. Optimization will run on random weights (meaningless)!")

    # Start Optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, model, val_loader, device, val_dataset.classes), 
        n_trials=cfg['inference']['optuna_trials']
    )

    print("\n" + "="*60)
    print("✨ Optimization Finished!")
    print(f"🏆 Best Balanced Accuracy: {study.best_value:.4f}")
    print("🧩 Best Hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    print("="*60)
    print("👉 Please update your 'configs/config.yaml' manually with these values.")
