import torch
import torch.nn as nn

class CrossAttentionFusionModule(nn.Module):
    """
    Merges local (CNN) and global (Transformer) features using Cross-Attention.
    Query comes from Swin (Global), Key/Value come from ConvNeXt (Local).
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        # Multi-head attention layer
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, cnn_features, transformer_features):
        # cnn_features: (Batch, Seq_Len, Dim) -> Acts as Key/Value
        # transformer_features: (Batch, Seq_Len, Dim) -> Acts as Query
        
        attn_output, _ = self.mha(query=transformer_features, key=cnn_features, value=cnn_features)
        x = self.norm1(transformer_features + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        output = self.norm2(x + self.dropout(ffn_output))
        
        return output
