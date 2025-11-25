import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from .preprocessing import get_transforms

def get_data_loaders(cfg):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    data_dir = cfg['data']['data_dir']
    batch_size = cfg['data']['batch_size']
    num_workers = cfg['data']['num_workers']
    
    # Get transforms
    transform = get_transforms(cfg)
    
    # 1. Load Training Data & Split
    train_path = os.path.join(data_dir, 'train')
    full_train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    
    # 80-20 Stratified Split (Approximated by random_split)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # Note: For strict stratification, we would use sklearn's StratifiedKFold,
    # but random_split is sufficient and standard for this implementation.
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg['seed'])
    )
    
    # 2. Load Test Data
    test_path = os.path.join(data_dir, 'test')
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    
    # 3. Create Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    # Return loaders and class info
    classes = full_train_dataset.classes
    return train_loader, val_loader, test_loader, classes
