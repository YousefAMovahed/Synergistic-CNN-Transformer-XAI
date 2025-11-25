import torch
from torchvision import transforms

def get_transforms(cfg):
    """
    Returns the standardization pipeline described in the paper.
    No augmentation is used to ensure reproducibility and fair comparison.
    """
    img_size = cfg['data']['img_size']
    
    # Standard ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return pipeline

def get_denormalization_stats():
    """
    Returns mean and std for visualization purposes (un-normalizing).
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return mean, std
