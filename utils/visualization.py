import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    """
    Plots and optionally saves a confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()

def visualize_consensus_masks(consensus_masks, class_names, sample_images_dict=None, save_path=None):
    """
    Visualizes the generated consensus masks for each class.
    
    Args:
        consensus_masks: Dict {class_idx: binary_mask}
        class_names: List of class names
        sample_images_dict: Dict {class_idx: tensor_image} (Optional, to overlay mask)
    """
    if not consensus_masks:
        print("❌ No masks to visualize.")
        return

    num_masks = len(consensus_masks)
    cols = 3
    rows = (num_masks + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_masks == 1: axes = [axes]
    axes = np.array(axes).flatten()

    # Denormalization params for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, (class_idx, mask) in enumerate(consensus_masks.items()):
        ax = axes[i]
        
        # If we have a sample image, overlay the mask
        if sample_images_dict and class_idx in sample_images_dict:
            img_tensor = sample_images_dict[class_idx].cpu() * std + mean
            img_np = np.clip(img_tensor.numpy().transpose(1, 2, 0), 0, 1)
            ax.imshow(img_np)
            ax.imshow(mask, cmap='jet', alpha=0.5) 
        else:
            # Just show the mask
            ax.imshow(mask, cmap='jet')
            
        ax.set_title(f"Consensus Mask: {class_names[class_idx]}")
        ax.axis('off')

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
