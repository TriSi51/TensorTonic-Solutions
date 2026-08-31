import torch
from typing import Tuple

def route_tokens_to_experts(
    router_logits: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (expert_indices, routing_weights), each of shape (num_tokens, top_k)
    """
    num_tokens, num_experts = router_logits.shape

    if top_k <= 0 or top_k > num_experts:
        raise ValueError("top_k must be between 1 and num_experts")

    sorted_indices = torch.argsort(
        router_logits,
        dim = -1,
        descending = True,
        stable = True,
    )

    expert_indices = sorted_indices[:, :top_k]

    selected_logits = torch.gather(
        router_logits,
        dim = -1,
        index = expert_indices
    )

    routing_weights = torch.softmax(selected_logits, dim = -1)
    return expert_indices, routing_weights
