import torch

def multi_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (batch, seq, d_model)
    """
    batch, seq, d_model = hidden_states.shape
    dk = d_model // num_query_heads

    query = hidden_states @ w_q # b,s,d_model
    key = hidden_states @ w_k  # b, s, dk
    value = hidden_states @ w_v # b, s, dk

    query = query.reshape(batch, seq, num_query_heads, dk).transpose(1,2) # b, num_query_heads, s, dk
    key = key.unsqueeze(1)
    value  = value.unsqueeze(1)
    scores = query @ key.transpose(-1,-2) # b, numhead,  s,s
    scores = scores / (dk ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(seq,seq, dtype = torch.bool), diagonal = 1)
        scores.masked_fill_(mask,float("-inf"))
    
    scores = torch.softmax(scores, dim = -1)
    scores = scores @ value # b, numhead, s,dk
    scores = scores.transpose(1,2).reshape(batch,seq, d_model)
    scores = scores @ w_o
    return scores
    
