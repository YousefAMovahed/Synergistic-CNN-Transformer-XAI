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
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    # 1. Load Config
    cfg = load_config()
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
    
    # Split Train/Val (80/20)
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

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=cfg['train']['lr'])
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
        
        val_acc = validate(model, val_loader, device)
        
        print(f"📉 Train Loss: {avg_loss:.4f} | 📈 Val Acc: {val_acc:.4f}")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            print("💾 Model Saved!")

if __name__ == "__main__":
    main()
