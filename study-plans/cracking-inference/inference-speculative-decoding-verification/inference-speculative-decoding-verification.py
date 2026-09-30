import torch

def verify_speculative_tokens(
    draft_token_ids: torch.Tensor,
    draft_distributions: torch.Tensor,
    target_distributions: torch.Tensor,
    uniform_draws: torch.Tensor,
) -> tuple:
    """
    Returns:
        accepted_prefix: shape (A,), integer tensor
        accepted_count: scalar integer tensor
        next_token: scalar integer tensor

    Expected shapes:
        draft_token_ids:       (K,)
        draft_distributions:   (K, V)
        target_distributions:  (K + 1, V)
        uniform_draws:         (K + 1,)

    The first K draws are used for accept/reject decisions.
    The final draw uniform_draws[-1] is used to sample either:
      - the residual distribution after the first rejection, or
      - the target row K if all draft tokens are accepted.
    """
    K = draft_token_ids.shape[0]

    accepted_count = 0
    rejection_idx = None

    for i in range(K):
        token_id = draft_token_ids[i]

        q = draft_distributions[i, token_id]
        p = target_distributions[i, token_id]

        if q.item() == 0:
            acceptance_prob = torch.zeros_like(p)
        else:
            acceptance_prob = torch.clamp(p/q, max= 1.0)

        if uniform_draws[i] < acceptance_prob:
            accepted_count += 1
        else:
            rejection_idx = i
            break

    if rejection_idx is not None:
        i = rejection_idx

        p = target_distributions[i]
        q = draft_distributions[i]

        residual = torch.clamp(p -q, min = 0.0)
        residual = residual / residual.sum()

        sample_distribution = residual
    else:
        sample_distribution = target_distributions[K]

    final_draw = uniform_draws[-1]
    cdf = torch.cumsum(sample_distribution, dim =0)

    # Need first index  where CDF > draw
    # right = True means exact equality moves to the next index

    next_token = torch.searchsorted(
        cdf,
        final_draw,
        right = True,
    ).to(torch.long)

    accepted_prefix = draft_token_ids[:accepted_count].to(torch.long)

    accepted_count_tensor = torch.tensor(
        accepted_count,
        dtype = torch.long,
        device = draft_token_ids.device
    )

    return (
        accepted_prefix,
        accepted_count_tensor,
        next_token
    )