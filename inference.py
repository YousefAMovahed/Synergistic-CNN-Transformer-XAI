import os
import yaml
import torch
import numpy as np
import optuna
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from sklearn.metrics import classification_report, balanced_accuracy_score

# Custom Imports
from models.hybrid_model import HybridVisionModel
from utils.gradcam import generate_single_instance_mask

# --- Helper Functions ---
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f: return yaml.safe_load(f)

def create_consensus_masks(model, loader, device, cfg_inf, verbose=True):
    """
    Phase 2: Generates consensus masks from correctly classified validation images.
    """
    model.eval()
    # Target the last convolutional layer of ConvNeXt backbone for GradCAM
    target_layer = model.cnn_backbone.stages[-1].blocks[-1].conv_dw
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    class_masks = {} # Stores list of masks per class
    
    if verbose: print("Generating Consensus Masks...")
    
    with torch.no_grad():
        for images, labels in loader: # Add tqdm here if needed
            images = images.to(device)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1)
            
            for i in range(len(labels)):
                # Only use CORRECT predictions to build the consensus
                if preds[i] == labels[i]:
                    label_idx = labels[i].item()
                    mask = generate_single_instance_mask(
                        cam, images[i], label_idx, 
                        activation_percentile=cfg_inf['activation_percentile'],
                        dilation_size=cfg_inf['dilation_size']
                    )
                    if label_idx not in class_masks: class_masks[label_idx] = []
                    class_masks[label_idx].append(mask)

    # Aggregate masks
    final_consensus = {}
    for cls, masks in class_masks.items():
        if len(masks) < 5: continue # Skip if not enough samples
        
        # Average the binary masks
        avg_mask = np.mean(np.stack(masks), axis=0)
        # Threshold to get final consensus
        final_mask = (avg_mask >= cfg_inf['consensus_threshold']).astype(np.uint8)
        
        if np.sum(final_mask) > 50: # Ensure mask is not empty
            final_consensus[cls] = final_mask
            
    if verbose: print(f"✅ Created masks for {len(final_consensus)} classes.")
    return final_consensus, cam

def run_intelligent_inference(model, loader, consensus_masks, cam, device, cfg_inf, class_names):
    """
    Phase 3: Inference with 'Dual Confirmation' correction logic.
    """
    model.eval()
    all_preds, all_labels = [], []
    corrections = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Intelligent Inference"):
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            probs = F.softmax(logits, dim=1)
            
            # Get Top-2 predictions
            top2_probs, top2_idxs = torch.topk(probs, 2, dim=1)
            
            final_batch_preds = top2_idxs[:, 0].clone()
            
            for i in range(len(labels)):
                pred_1 = top2_idxs[i, 0].item()
                conf_1 = top2_probs[i, 0].item()
                
                # Check for uncertainty
                if conf_1 < cfg_inf['confidence_threshold'] and pred_1 in consensus_masks:
                    # Generate mask for test image based on current prediction
                    test_mask = generate_single_instance_mask(
                        cam, images[i], pred_1, 
                        activation_percentile=cfg_inf['activation_percentile']
                    )
                    
                    # Compare with all consensus masks
                    best_iou = -1
                    proposed_class = pred_1
                    
                    for cls_idx, cons_mask in consensus_masks.items():
                        # Calculate Intersection (Simplified IoU for speed)
                        intersection = np.sum(test_mask & cons_mask)
                        if intersection > best_iou:
                            best_iou = intersection
                            proposed_class = cls_idx
                    
                    # Dual Confirmation Logic:
                    # 1. Visual evidence (Mask) points to 'proposed_class'
                    # 2. Probabilistic evidence (Second best) matches 'proposed_class'
                    pred_2 = top2_idxs[i, 1].item()
                    
                    if proposed_class == pred_2:
                        final_batch_preds[i] = pred_2
                        corrections += 1

            all_preds.extend(final_batch_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds, corrections

# --- Main Execution ---
def main():
    cfg = load_config()
    device = cfg['device']
    
    # Setup Data (Test Set)
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Assuming standard folder structure
    val_dataset = datasets.ImageFolder(os.path.join(cfg['data']['data_dir'], 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(cfg['data']['data_dir'], 'test'), transform=transform)
    
    val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'], shuffle=False)
    
    # Load Model
    num_classes = len(test_dataset.classes)
    model = HybridVisionModel(
        cfg['model']['cnn_backbone'], cfg['model']['swin_backbone'], num_classes,
        cfg['model']['embed_dim'], cfg['model']['num_heads'], cfg['model']['ff_dim']
    ).to(device)
    
    # Load Weights
    if os.path.exists("best_hybrid_model.pth"):
        model.load_state_dict(torch.load("best_hybrid_model.pth", map_location=device))
        print("✅ Loaded trained weights.")
    else:
        print("⚠️ No weights found! Please run train.py first.")
        return

    # Phase 2: Create Consensus Masks (Using Default Config)
    print("\n--- Phase 2: Generating Masks ---")
    consensus_masks, cam = create_consensus_masks(model, val_loader, device, cfg['inference'])
    
    # Phase 3: Run Inference
    print("\n--- Phase 3: Running Intelligent Inference ---")
    y_true, y_pred, corrections_made = run_intelligent_inference(
        model, test_loader, consensus_masks, cam, device, cfg['inference'], test_dataset.classes
    )
    
    print(f"\n📊 Results:")
    print(f"Interventions Made: {corrections_made}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

if __name__ == "__main__":
    main()
