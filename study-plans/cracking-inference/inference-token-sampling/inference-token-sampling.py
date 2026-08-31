import torch

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    uniform_draws: torch.Tensor,
) -> torch.Tensor:
    """
    Returns: sampled token id tensor of shape (batch,), dtype torch.int64
    """
    if temperature == 0:
        return torch.argmax(logits, dim = -1).to(torch.int64)
    batch, vocab_size = logits.shape
    
    scaled_logits = logits / temperature
    if top_k > 0:
        k = min(top_k, vocab_size)
    
        topk_values, topk_indices = torch.topk(
            scaled_logits,
            k = k,
            dim = -1
        )
    
        filtered_logits = torch.full_like(
            scaled_logits,
            float("-inf")
        )
    
        filtered_logits.scatter_(
            dim = -1,
            index = topk_indices,
            src= topk_values
        )
    else:
        filtered_logits = scaled_logits

    probs = torch.softmax(filtered_logits, dim = -1)

    sorted_probs, sorted_indices = torch.sort(
        probs,
        dim = -1,
        descending = True
    )

    cumulative_probs = torch.cumsum(sorted_probs, dim = -1)

    # remove = cumulative_probs > top_p

    # remove[..., 1:] = remove[..., :-1].clone()
    # remove[..., 0] = False
    cumulative_before =  cumulative_probs - sorted_probs
    keep_sorted = cumulative_before < top_p
    
    
    sorted_probs = sorted_probs.masked_fill(~keep_sorted,0.0)

    probs = torch.zeros_like(sorted_probs)

    probs.scatter_(
        dim = -1,
        index = sorted_indices,
        src = sorted_probs
    )

    probs = probs / probs.sum(dim = -1, keepdim = True)

    cdf = torch.cumsum(probs, dim = -1)

    sampled = torch.sum(
        cdf <= uniform_draws.unsqueeze(-1),
        dim = -1
    )

    return sampled.to(torch.int64)
    