import torch

def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (batch, seq, d_model)
    """
    d_model = w_q.shape[-1]
    b, s, _ = hidden_states.shape
    dk = d_model// num_heads

    query = hidden_states @ w_q
    key = hidden_states @ w_k
    value = hidden_states @ w_v

    query = query.reshape(b, s, num_heads, dk).transpose(1,2)
    key = key.reshape(b, s, num_heads, dk).transpose(1,2)
    value = value.reshape(b,s,num_heads, dk).transpose(1,2)
    key_t = key.transpose(-1,-2)
    scores = torch.matmul(query, key_t) 
    scores = scores / (dk ** 0.5) # b, numhead, s, s

    if causal:
        masks = torch.triu(torch.ones(s,s, dtype= torch.bool), diagonal = 1)
        scores.masked_fill_(masks, float("-inf"))

    scores = torch.softmax(scores, dim = -1)
    attention_scores = scores @ value # (b, numhead, s,dk)

    attention_scores = attention_scores.transpose(1,2).reshape(b, s,d_model)
    final_scores= attention_scores @ w_o
    return final_scores