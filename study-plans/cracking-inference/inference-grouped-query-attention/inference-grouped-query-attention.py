import torch

def grouped_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (batch, seq, d_model)
    """
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads"
        )
    b,s, d_model = hidden_states.shape
    group_size = num_query_heads // num_kv_heads
    dk = d_model // num_query_heads
    query = hidden_states @ w_q # B, S, DMODEL
    key = hidden_states @ w_k # B, S, numkvhead * dk
    value = hidden_states @ w_v # B, S, numkvhead * dk

    query = query.reshape(b,s, num_query_heads,dk).transpose(1,2)
    key = key.reshape(b,s, num_kv_heads, dk).transpose(1,2) # B, num-kv-head, S,dk
    value = value.reshape(b,s, num_kv_heads, dk).transpose(1,2) # B, num-kv-head, S, dk

    key = key.repeat_interleave(group_size, dim = 1)
    value = value.repeat_interleave(group_size, dim = 1)
    scores = (query @ key.transpose(-2,-1)) / (dk ** 0.5) 

    if causal:
        masks = torch.triu(torch.ones(s,s,dtype = torch.bool), diagonal= 1)
        scores.masked_fill_(masks, float("-inf"))
    scores = torch.softmax(scores, dim = -1)
    scores = scores @ value
    scores = scores.transpose(1,2).reshape(b,s, d_model)
    scores = scores @ w_o
    return scores
    
    
