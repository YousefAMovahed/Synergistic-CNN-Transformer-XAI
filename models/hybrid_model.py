import torch
import torch.nn as nn
import timm
from .fusion import CrossAttentionFusionModule

class HybridVisionModel(nn.Module):
    """
    Hybrid architecture combining ConvNeXt (Local) and Swin Transformer (Global).
    """
    def __init__(self, cnn_model_name, swin_model_name, num_classes, embed_dim, num_heads, ff_dim):
        super().__init__()
        
        # 1. Load Pretrained Backbones
        # Remove classification heads (num_classes=0) to get raw features
        self.cnn_backbone = timm.create_model(cnn_model_name, pretrained=True, num_classes=0, global_pool='')
        self.transformer_backbone = timm.create_model(swin_model_name, pretrained=True, num_classes=0, global_pool='')

        # 2. Projections to match embedding dimensions
        self.cnn_proj = nn.Linear(self.cnn_backbone.num_features, embed_dim)
        self.transformer_proj = nn.Linear(self.transformer_backbone.num_features, embed_dim)

        # 3. Fusion Module
        self.fusion_module = CrossAttentionFusionModule(embed_dim, num_heads, ff_dim)
        self.post_fusion_norm = nn.LayerNorm(embed_dim)

        # 4. Classification & Projection Heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # Projection head for Contrastive Loss
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 128)
        )
        
        # Final Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Extract features
        cnn_map = self.cnn_backbone.forward_features(x)         # (B, C, H, W)
        swin_map = self.transformer_backbone.forward_features(x) # (B, C, H, W)

        # Flatten and Project
        # CNN: (B, C, H, W) -> (B, H*W, C)
        cnn_seq = cnn_map.flatten(2).permute(0, 2, 1)
        cnn_proj = self.cnn_proj(cnn_seq)

        # Swin: (B, H, W, C) -> (B, H*W, C) (Depending on timm version output shape)
        # Note: swin_small usually outputs (B, H, W, C) or (B, C, H, W). 
        # We handle standard (B, C, H, W) or permuted inside timm.
        # Assuming timm returns (B, C, H, W) for standard consistency:
        if len(swin_map.shape) == 4 and swin_map.shape[1] == self.transformer_backbone.num_features:
             swin_seq = swin_map.flatten(2).permute(0, 2, 1)
        else:
             # If channel is last
             swin_seq = swin_map.flatten(1, 2)
             
        transformer_proj = self.transformer_proj(swin_seq)

        # Fuse
        fused = self.fusion_module(cnn_proj, transformer_proj)
        norm_fused = self.post_fusion_norm(fused)

        # Pool and Classify
        pooled = self.pool(norm_fused.permute(0, 2, 1)) # (B, Dim, 1)
        embedding = self.flatten(pooled)
        
        proj_output = self.projection_head(embedding) # For SupCon Loss
        logits = self.classifier(embedding)           # For Focal Loss

        return logits, proj_output
