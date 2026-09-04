import torch
from typing import Tuple
import math

def cached_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (outputs, key_cache, value_cache) for the full sequence, built incrementally
    """
    batch, seq, dk = query.shape
    dv = value.shape[-1]
    key_cache = torch.empty_like(key)
    value_cache= torch.empty_like(value)
    outputs = torch.empty(
        (batch,seq, dv),
        dtype = value.dtype,
    )

    for t in range(seq):
        key_cache[:, t, :] = key[:, t,:]
        value_cache[:, t, :] = value[:, t,:]

        K = key_cache[:, :t+1, :]
        V = value_cache[:, :t+1, :]

        q= query[:, t:t+1, :]

        scores = torch.matmul(
            q,
            K.transpose(-2, -1)
            
        )  # B, 1, dk  @ B, dk, t+1 = B, 1, t+1

        scores = scores / (dk ** 0.5)

        weights= torch.softmax(scores, dim = -1)
        out = torch.matmul(weights, V) # (B, 1, dv)

        outputs[:, t:t+1, :]= out

    return outputs, key_cache, value_cache