import torch
import torch.nn as nn
import math

class AdaptivePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, max_relative_position=128):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Learnable relative position embeddings
        self.relative_pos_embedding = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
        
        # Content-based position adaptation
        self.content_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, 1)
        
        # Fallback sinusoidal encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Get relative positions
        positions = torch.arange(seq_len, device=x.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = torch.clamp(
            relative_positions, -self.max_relative_position, self.max_relative_position
        ) + self.max_relative_position
        
        # Get relative position embeddings
        rel_pos_emb = self.relative_pos_embedding(relative_positions)
        rel_pos_emb = rel_pos_emb.mean(dim=1)  # Average over sequence dimension
        
        # Content-based adaptation
        content_features = self.content_proj(x)
        pos_features = self.pos_proj(rel_pos_emb.unsqueeze(0).expand(batch_size, -1, -1))
        
        # Adaptive gating
        gate_input = torch.cat([content_features, pos_features], dim=-1)
        gate_weights = torch.sigmoid(self.gate(gate_input))
        
        # Combine adaptive and sinusoidal encodings
        sin_pos = self.pe[:, :seq_len]
        adaptive_pos = gate_weights * pos_features + (1 - gate_weights) * sin_pos
        
        return x + adaptive_pos