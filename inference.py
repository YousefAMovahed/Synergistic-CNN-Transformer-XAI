import os
import csv
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from sklearn.metrics import classification_report, balanced_accuracy_score

# Custom Imports
from models.hybrid_model import HybridVisionModel
from utils.gradcam import generate_single_instance_mask
from utils.metrics import mcnemar_exact

# --- Helper Functions ---
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f: return yaml.safe_load(f)

def create_consensus_masks(model, loader, device, cfg_inf, verbose=True):
    """
    Phase 2 (Step 1): Generates consensus masks from correctly classified validation images.
    Also returns tau_err, the mean top-1 confidence of misclassified validation images.
    """
    model.eval()
    # Target the last depthwise convolution of the final ConvNeXt stage for Grad-CAM
    target_layer = model.cnn_backbone.stages[-1].blocks[-1].conv_dw
    cam = GradCAM(model=model, target_layers=[target_layer])

    class_masks = {}  # Stores list of masks per class
    wrong_conf = []   # Top-1 confidence of misclassified validation images

    if verbose: print("Generating Consensus Masks...")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits, _ = model(images)
            probs = F.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)

            for i in range(len(labels)):
                if preds[i].item() != labels[i].item():
                    wrong_conf.append(conf[i].item())
                    continue
                # Only CORRECT predictions build the consensus
                label_idx = labels[i].item()
                mask = generate_single_instance_mask(
                    cam, images[i], label_idx,
                    activation_percentile=cfg_inf['activation_percentile'],
                    dilation_size=cfg_inf.get('dilation_size', 3)
                )
                class_masks.setdefault(label_idx, []).append(mask)

    # Aggregate masks: C_c = 1[F_c >= tau_cons], valid only if N_c >= 5 and area >= 100 pixels.
    # Classes are visited in index order, which fixes the tie-breaking of the argmax below.
    final_consensus = {}
    for cls, masks in sorted(class_masks.items()):
        if len(masks) < 5: continue
        freq_map = np.mean(np.stack(masks), axis=0)
        final_mask = (freq_map >= cfg_inf['consensus_threshold']).astype(np.uint8)
        if np.sum(final_mask) >= cfg_inf.get('min_mask_pixels', 100):
            final_consensus[cls] = final_mask

    tau_err = float(np.mean(wrong_conf)) if wrong_conf else 1.0

    if verbose:
        print(f"Created valid masks for {len(final_consensus)} classes: {sorted(final_consensus)}")
        for cls, m in sorted(final_consensus.items()):
            print(f"   class {cls}: mask area = {int(m.sum())} pixels ({100 * m.mean():.2f}% of the image)")
        print(f"tau_err (mean confidence of misclassified validation images) = {tau_err:.4f}")
    return final_consensus, cam, tau_err

def operating_threshold(cfg_inf, tau_err):
    """tau = fixed_tau if given (AD5 run: 0.6585); otherwise min(tau_conf, tau_err), or tau_conf alone
    if use_error_threshold is False."""
    if cfg_inf.get('fixed_tau') is not None:
        return float(cfg_inf['fixed_tau'])
    tau_conf = cfg_inf['confidence_threshold']
    return min(tau_conf, tau_err) if cfg_inf.get('use_error_threshold', True) else tau_conf

def run_intelligent_inference(model, loader, consensus_masks, cam, device, cfg_inf, class_names,
                              tau_err=1.0, audit_path=None):
    """
    Phase 2 (Step 3): SecondLook review with the dual-confirmation rule.

    A prediction is reviewed if p1 < tau and its class y1 has a valid consensus mask (the rule
    of the archived experiments; set require_valid_pred_mask: false for the variant without the
    second condition). The Grad-CAM mask of y1 is compared with every valid consensus mask by the
    unnormalised pixel intersection S(x, c) = sum(M_x * C_c); y_mask = argmax_c S(x, c). The label
    is switched to y2 only if y_mask == y2.
    """
    model.eval()
    tau = operating_threshold(cfg_inf, tau_err)
    require_valid = cfg_inf.get('require_valid_pred_mask', True)
    all_preds, all_labels, audit_rows = [], [], []
    sample_idx = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="SecondLook review"):
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            probs = F.softmax(logits, dim=1)
            top2_probs, top2_idxs = torch.topk(probs, 2, dim=1)
            final_batch_preds = top2_idxs[:, 0].clone()

            for i in range(len(labels)):
                y_true = labels[i].item()
                pred_1, conf_1 = top2_idxs[i, 0].item(), top2_probs[i, 0].item()
                pred_2 = top2_idxs[i, 1].item()
                triggered = conf_1 < tau and len(consensus_masks) > 0
                if require_valid and pred_1 not in consensus_masks:
                    triggered = False

                proposed_class, scores = None, {}
                if triggered:
                    # Instance mask for the current (top-1) prediction
                    test_mask = generate_single_instance_mask(
                        cam, images[i], pred_1,
                        activation_percentile=cfg_inf['activation_percentile'],
                        dilation_size=cfg_inf.get('dilation_size', 3)
                    )
                    # Unnormalised pixel intersection with every valid consensus mask
                    scores = {cls: int(np.sum(test_mask & cons)) for cls, cons in consensus_masks.items()}
                    proposed_class = max(scores, key=scores.get)
                    if proposed_class == pred_2:
                        final_batch_preds[i] = pred_2

                final = final_batch_preds[i].item()
                if final == pred_1:
                    outcome = "unchanged" if triggered else "not_reviewed"
                elif final == y_true:
                    outcome = "corrected"
                elif pred_1 == y_true:
                    outcome = "harmed"
                else:
                    outcome = "wrong_to_wrong"
                audit_rows.append({
                    "index": sample_idx, "true": y_true, "pred_1": pred_1, "p1": round(conf_1, 6),
                    "pred_2": pred_2, "triggered": int(triggered), "y_mask": proposed_class,
                    "scores": scores, "final": final, "outcome": outcome,
                })
                sample_idx += 1

            all_preds.extend(final_batch_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    counts = {k: sum(r["outcome"] == k for r in audit_rows)
              for k in ("corrected", "harmed", "wrong_to_wrong", "unchanged")}
    counts["triggered"] = sum(r["triggered"] for r in audit_rows)
    counts["tau"] = tau
    counts["mcnemar_p"] = mcnemar_exact(counts["corrected"], counts["harmed"])

    if audit_path:
        os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
        with open(audit_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)

    return all_labels, all_preds, counts

# --- Main Execution ---
def main():
    cfg = load_config()
    device = cfg['device'] if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Standard ImageFolder structure
    val_dataset = datasets.ImageFolder(os.path.join(cfg['data']['data_dir'], 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(cfg['data']['data_dir'], 'test'), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'], shuffle=False)

    num_classes = len(test_dataset.classes)
    model = HybridVisionModel(
        cfg['model']['cnn_backbone'], cfg['model']['swin_backbone'], num_classes,
        cfg['model']['embed_dim'], cfg['model']['num_heads'], cfg['model']['ff_dim']
    ).to(device)

    if os.path.exists("best_hybrid_model.pth"):
        model.load_state_dict(torch.load("best_hybrid_model.pth", map_location=device))
        print("Loaded trained weights.")
    else:
        print("No weights found! Please run train.py first.")
        return

    # Phase 2, Step 1: consensus masks (validation split only)
    print("\n--- Consensus masks ---")
    consensus_masks, cam, tau_err = create_consensus_masks(model, val_loader, device, cfg['inference'])

    # Phase 2, Step 3: SecondLook review on the test split
    print("\n--- SecondLook review ---")
    y_true, y_pred, counts = run_intelligent_inference(
        model, test_loader, consensus_masks, cam, device, cfg['inference'], test_dataset.classes,
        tau_err=tau_err, audit_path=cfg['inference'].get('audit_path', 'results/secondlook_audit.csv')
    )

    print("\nResults:")
    print(f"Operating threshold tau = {counts['tau']:.4f}")
    print(f"Reviewed: {counts['triggered']} | corrected (b): {counts['corrected']} | "
          f"harmed (c): {counts['harmed']} | wrong-to-wrong: {counts['wrong_to_wrong']} | "
          f"unchanged: {counts['unchanged']}")
    print(f"Exact McNemar p (b vs c): {counts['mcnemar_p']:.4g}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes, digits=4))

if __name__ == "__main__":
    main()
