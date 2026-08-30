import torch
from typing import Tuple

def apply_rotary_position_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (rotated query tensor, rotated key tensor), same shapes as inputs
    """

    if query.shape != key.shape:
        raise ValueError("query and key must have identical shapes")

    if query.ndim != 4:
        raise ValueError("query and key must have shape (batch, heads, seq, dk)")

    if base <= 0:
        raise ValueError("base must be positive")

    B, H, S, D = query.shape

    if D % 2 != 0:
        raise ValueError("dk must be even")
    
    i = torch.arange(
        0, D // 2,
        dtype = torch.float64,
    )

    theta = base ** (-2.0 * i /D)

    # Position: (S,) reshape to (1,1,S,1)
    pos = positions.to(
            dtype = torch.float64,
    )[None, None, :, None]

    angle = pos * theta

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    def rotate(x: torch.Tensor) -> torch.Tensor:

        x_even = x[..., 0::2]

        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd  * cos

        out = torch.empty_like(x)

        out[...,0::2] = rotated_even
        out[...,1::2] = rotated_odd

        return out
    return rotate(query), rotate(key)
    
