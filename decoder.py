import torch
import torch.nn as nn
from multi_scale_attention import MultiScaleAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_multi_scale=True):
        super().__init__()
        
        if use_multi_scale:
            self.self_attention = MultiScaleAttention(d_model, n_heads, dropout=dropout)
            self.cross_attention = MultiScaleAttention(d_model, n_heads, dropout=dropout)
        else:
            from attention import MultiHeadAttention
            self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
            self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
            
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # Masked self-attention + residual
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention + residual
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward + residual
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1, use_multi_scale=True):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, use_multi_scale)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return x