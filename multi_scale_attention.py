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
        
        # Ensure n_heads is divisible by the number of scales
        assert n_heads % len(scale_factors) == 0
        self.n_scales = len(scale_factors)
        
        self.heads_per_scale = n_heads // self.n_scales
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def downsample(self, x, factor):
        if factor == 1:
            return x
        
        # This logic correctly handles all cases
        if len(x.shape) == 4:
            # Check if it's a square tensor like an attention mask (L x L)
            if x.shape[2] == x.shape[3]:
                return x[:, :, ::factor, ::factor]
            # Otherwise, it's Q, K, or V (L x D_k), so only slice the L dimension
            else:
                return x[:, :, ::factor, :]
        elif len(x.shape) == 3: # (B, L, D)
            return x[:, ::factor, :]
        else:
            raise ValueError(f"Unsupported tensor shape for downsampling: {x.shape}")

   
    def upsample(self, x, target_len):
        # x has shape (batch_size, heads_per_scale, seq_len_down, d_k)
        if x.shape[2] == target_len:
            return x

        b, h, l, d = x.shape
        
        x = x.reshape(b * h, l, d).transpose(1, 2)
        x_interp = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        x_interp = x_interp.transpose(1, 2)
        
        return x_interp.reshape(b, h, target_len, d)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
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
                mask_down = self.downsample(mask, scale)
                # If mask is 4D (B, H, L, L), we need to downsample both L dimensions
                if len(mask_down.shape) == 4 and mask_down.shape[2] != mask_down.shape[3]:
                    # Downsample the last dimension too
                    mask_down = mask_down[:, :, :, ::scale]
                scores = scores.masked_fill(mask_down == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            output_scale = torch.matmul(attention_weights, V_down)
            
            output_scale = self.upsample(output_scale, seq_len)
            
            outputs.append(output_scale)
        
        output = torch.cat(outputs, dim=1)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(output)