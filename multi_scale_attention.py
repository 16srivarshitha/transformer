import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, n_heads, scale_factors=[1, 2], dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale_factors = scale_factors
        
        assert n_heads % len(scale_factors) == 0, "n_heads must be divisible by the number of scales"
        self.n_scales = len(scale_factors)
        
        self.heads_per_scale = n_heads // self.n_scales
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    # --- CORRECTED downsample FUNCTION ---
    # Made more robust to handle all mask types
    def downsample(self, x, factor):
        if factor == 1:
            return x
        
        if len(x.shape) == 4:
            # Case 1: Padding mask (B, 1, 1, L) from the encoder. Only downsample the last dimension.
            if x.shape[1] == 1 and x.shape[2] == 1:
                return x[:, :, :, ::factor]
            # Case 2: Square causal mask (B, H, L, L). Downsample both sequence dimensions.
            elif x.shape[2] == x.shape[3]:
                return x[:, :, ::factor, ::factor]
            # Case 3: Q, K, or V tensor (B, H, L, D_k). Downsample the sequence length dimension.
            else:
                return x[:, :, ::factor, :]
        elif len(x.shape) == 3: # For tensors like (B, L, D)
            return x[:, ::factor, :]
        else:
            raise ValueError(f"Unsupported tensor shape for downsampling: {x.shape}")

    def upsample(self, x, target_len):
        if x.shape[2] == target_len:
            return x

        b, h, l, d = x.shape
        
        x = x.reshape(b * h, l, d).transpose(1, 2)
        x_interp = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        x_interp = x_interp.transpose(1, 2)
        
        return x_interp.reshape(b, h, target_len, d)
    
    # --- CORRECTED forward METHOD ---
    def forward(self, query, key, value, mask=None):
        # Use independent lengths for query, key, and value for cross-attention
        batch_size = query.size(0)
        query_len, key_len, value_len = query.size(1), key.size(1), value.size(1)
        
        # Reshape using correct sequence lengths
        Q = self.w_q(query).view(batch_size, query_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, value_len, self.n_heads, self.d_k).transpose(1, 2)
        
        outputs = []
        
        for i, scale in enumerate(self.scale_factors):
            start_head = i * self.heads_per_scale
            end_head = (i + 1) * self.heads_per_scale
            
            Q_scale = Q[:, start_head:end_head]
            K_scale = K[:, start_head:end_head]
            V_scale = V[:, start_head:end_head]
            
            Q_down = self.downsample(Q_scale, scale)
            K_down = self.downsample(K_scale, scale)
            V_down = self.downsample(V_scale, scale)
            
            scores = torch.matmul(Q_down, K_down.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                # Downsample the mask. The corrected downsample function handles all cases.
                mask_down = self.downsample(mask, scale)
                # Let PyTorch's broadcasting handle the mask application.
                scores = scores.masked_fill(mask_down == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            output_scale = torch.matmul(attention_weights, V_down)
            
            # Upsample back to the original QUERY length
            output_scale = self.upsample(output_scale, query_len)
            
            outputs.append(output_scale)
        
        output = torch.cat(outputs, dim=1)
        # Reshape using the QUERY length, as that is the output sequence length
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        
        return self.w_o(output)