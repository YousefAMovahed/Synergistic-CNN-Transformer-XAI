import numpy as np
import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.ndimage import binary_dilation

def generate_single_instance_mask(cam_generator, input_tensor, target_class_idx, activation_percentile=95, dilation_size=3):
    """
    Generates a binary mask for a single image using Grad-CAM.
    
    Args:
        cam_generator: Initialized GradCAM object.
        input_tensor: Preprocessed image tensor (C, H, W).
        target_class_idx: The class index to generate the map for.
        activation_percentile: Threshold percentile to binarize the heatmap.
        dilation_size: Kernel size for morphological dilation to smooth the mask.
        
    Returns:
        dilated_mask: Binary mask (numpy array) of significant regions.
    """
    targets = [ClassifierOutputTarget(target_class_idx)]
    
    with torch.enable_grad():
        # Generate raw grayscale heatmap
        grayscale_cam = cam_generator(input_tensor=input_tensor.unsqueeze(0), targets=targets)[0, :]
        
    # Thresholding
    threshold_value = np.percentile(grayscale_cam, activation_percentile)
    hotspot_mask = (grayscale_cam >= threshold_value)
    
    # Dilation (Smoothing)
    dilation_structure = np.ones((dilation_size, dilation_size))
    dilated_mask = binary_dilation(hotspot_mask, structure=dilation_structure)
    
    return dilated_mask.astype(np.uint8)
