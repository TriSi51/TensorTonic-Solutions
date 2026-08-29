import torch
from typing import Optional

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns: attention output tensor of shape (batch, seq_q, d_v)
    """
    dk = query.shape[-1]
    key_t = key.transpose(-2,-1)

    scores = torch.matmul(query, key_t)

    scores = scores / (dk ** 0.5)
    if mask is not None:
        scores.masked_fill_(mask, float("-inf")) # B, Sq  Sk 

    scores = torch.softmax(scores, dim = -1)
    attention_scores = torch.matmul(scores, value)
    return attention_scores
    
    
