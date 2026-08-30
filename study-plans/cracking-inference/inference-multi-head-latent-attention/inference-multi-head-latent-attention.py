import torch
from typing import Tuple

def multi_head_latent_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_down: torch.Tensor,
    w_up_k: torch.Tensor,
    w_up_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (output tensor of shape (batch, seq, d_model), latent tensor of shape (batch, seq, d_latent))
    """
    b, s, d_model = hidden_states.shape
    dk = d_model // num_heads
    c = hidden_states @ w_down
    query = hidden_states @ w_q
    key = c @ w_up_k
    value = c @ w_up_v

    query = query.reshape(b,s , num_heads, dk).transpose(1,2)
    key = key.reshape(b, s, num_heads, dk).transpose(1,2)
    value = value.reshape(b,s, num_heads, dk).transpose(1,2)

    scores =( query @ key.transpose(-2,-1) ) / (dk ** 0.5)
    if causal:
        masks = torch.triu(torch.ones(s,s,dtype = torch.bool), diagonal= 1)
        scores.masked_fill_(masks, float("-inf"))

    attention_weights = torch.softmax(scores, dim = -1)
    attention_scores = attention_weights @ value
    attention_scores = attention_scores.transpose(1,2).reshape(b,s ,d_model)
    attention_scores = attention_scores @ w_o
    return (attention_scores,c )
    
