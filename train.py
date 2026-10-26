import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np

# Custom Imports
from models.hybrid_model import HybridVisionModel
from utils.losses import FocalLoss, SupervisedContrastiveLoss

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, loader, optimizer, criterion_cls, criterion_scl, alpha, device):
    model.train()
    total_loss = 0
    
    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, embeddings = model(images)
        
        # Calculate losses
        loss_cls = criterion_cls(logits, labels)
        loss_scl = criterion_scl(embeddings, labels)
        
        # Composite Loss: L_total = alpha * L_focal + (1 - alpha) * L_supcon
        loss = alpha * loss_cls + (1 - alpha) * loss_scl
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    """Balanced accuracy on the validation split (the Phase 1 checkpoint criterion)."""
    from sklearn.metrics import balanced_accuracy_score
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits, _ = model(images.to(device))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return balanced_accuracy_score(all_labels, all_preds)

def set_seed(seed):
    """Fix random seeds for reproducibility (seeds 42, 123 and 2026 were used in the paper)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # 1. Load Config
    cfg = load_config()
    set_seed(cfg['seed'])
    device = cfg['device'] if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Phase 1 Training on {device}...")

    # 2. Data Preparation
    # Standard ImageNet normalization [cite: 4]
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(root=os.path.join(cfg['data']['data_dir'], 'train'), transform=transform)
    
    # Split Train/Val (80/20, not stratified). The default generator, seeded by set_seed above, is
    # used as in the archived notebook, so seed 42 reproduces its split (AD5: 1,785 / 447 images).
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])

    num_classes = len(full_dataset.classes)
    print(f"ℹ️ Classes: {full_dataset.classes}")

    # 3. Model Initialization
    model = HybridVisionModel(
        cnn_model_name=cfg['model']['cnn_backbone'],
        swin_model_name=cfg['model']['swin_backbone'],
        num_classes=num_classes,
        embed_dim=cfg['model']['embed_dim'],
        num_heads=cfg['model']['num_heads'],
        ff_dim=cfg['model']['ff_dim']
    ).to(device)

    # 4. Optimizer & Loss. Short budget: the pretrained backbones are frozen and only the
    # projections, the fusion block and the heads are trained.
    if cfg['train'].get('freeze_backbones', True):
        for p in model.cnn_backbone.parameters(): p.requires_grad = False
        for p in model.transformer_backbone.parameters(): p.requires_grad = False
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['train']['lr'])
    criterion_cls = FocalLoss(alpha=0.25, label_smoothing=cfg['train']['label_smoothing']).to(device)
    criterion_scl = SupervisedContrastiveLoss(temperature=cfg['train']['scl_temperature']).to(device)

    # 5. Training Loop
    best_acc = 0.0
    for epoch in range(cfg['train']['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['train']['epochs']}")
        
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_scl, 
            cfg['train']['loss_alpha'], device
        )
        
        val_ba = validate(model, val_loader, device)
        print(f"📉 Train Loss: {avg_loss:.4f} | 📈 Val balanced accuracy: {val_ba:.4f}")

        # Keep the epoch with the highest validation balanced accuracy (Phase 1 checkpoint rule)
        if val_ba > best_acc:
            best_acc = val_ba
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            print("💾 Model Saved!")

if __name__ == "__main__":
    main()
