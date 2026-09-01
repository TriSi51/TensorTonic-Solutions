import torch

def sparse_moe_forward(
    token_states: torch.Tensor,
    router_logits: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (num_tokens, d_model)
    """
    T, d_model = token_states.shape
    num_experts = router_logits.shape[1]

    if top_k <= 0 or top_k > num_experts:
        raise ValueError("top_k must be between 1 and num_experts")

    sorted_indices = torch.argsort(
        router_logits,
        dim = -1,
        descending= True,
        stable = True,
    )

    expert_indices = sorted_indices[:, :top_k] # (T,k)

    selected_logits = torch.gather(
        router_logits,
        dim = -1,
        index = expert_indices        
    ) # (T, k)

    gate_weights = torch.softmax(
        selected_logits,
        dim = -1
    )  # (T, k)

    # Flatten token-expert routes
    token_ids = (
        torch.arange(T)
        .unsqueeze(1)
        .expand(T,top_k)
        .reshape(-1)
    ) # (T*k, )

    expert_ids = expert_indices.reshape(-1)  # (T*k,)
    weights = gate_weights.reshape(-1) # (T*k,)

    # Group routes by expert
    # sort all routes by expert Id
    order = torch.argsort(expert_ids, stable = True)

    expert_ids = expert_ids[order]
    token_ids = token_ids[order]
    weights = weights[order]

    # Run each expert only on its routed token

    output = torch.zeros_like(token_states)
    for expert in range(num_experts):
        mask = expert_ids == expert
        
        if not mask.any():
            continue

        expert_token_ids = token_ids[mask]
        x= token_states[expert_token_ids]

        hidden = torch.relu(x @ w_in[expert])
        expert_output = hidden @ w_out[expert]

        expert_weights = weights[mask].to(expert_output.dtype)

        expert_weights = expert_weights.unsqueeze(1) # (N,1)
        weighted_output = expert_output * expert_weights
        output.index_add_(
            dim = 0,
            index = expert_token_ids,
            source = weighted_output
        )

    return output
    