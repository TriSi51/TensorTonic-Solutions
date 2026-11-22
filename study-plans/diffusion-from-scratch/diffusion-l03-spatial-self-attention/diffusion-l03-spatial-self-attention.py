import math
import torch
import torch.nn.functional as F

def spatial_self_attention(x: torch.Tensor, params: dict, num_groups: int, num_heads: int) -> torch.Tensor:
    """
    Returns x plus the attention output, shape (B, C, H, W) with the dtype of x.
    """
    H, W = x.shape[2], x.shape[3]
    x_norm = F.group_norm(x, num_groups, params["norm_weight"], params["norm_bias"])
    x_flat = x_norm.flatten(2).transpose(1,2)  # B, N, C
    B,N,C = x_flat.shape
    dim = C // num_heads
    Q = F.linear(x_flat, params["q_weight"], params["q_bias"])
    K = F.linear(x_flat, params["k_weight"], params["k_bias"])
    V = F.linear(x_flat, params["v_weight"], params["v_bias"])

    Q = Q.reshape(B, N, num_heads, dim).transpose(1,2)
    K = K.reshape(B, N, num_heads, dim).transpose(1,2)
    V = V.reshape(B, N, num_heads, dim).transpose(1,2)
    score = (Q @ K.transpose(-2,-1)) / torch.sqrt(torch.tensor(dim, dtype = Q.dtype))
    score = torch.softmax(score, dim = -1)
    score = score @ V   # B, numhead, N, dim

    score = score.transpose(1,2).reshape(B, N, C)
    
    residual = x
    out = F.linear(score, params["out_weight"], params["out_bias"])
    out = out.transpose(-2, -1).reshape(B, C, H, W)
    return residual + out
    
    
    