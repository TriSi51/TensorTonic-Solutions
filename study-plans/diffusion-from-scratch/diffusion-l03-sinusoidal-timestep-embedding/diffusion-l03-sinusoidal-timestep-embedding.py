import torch
import math
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float) -> torch.Tensor:
    """
    Returns the float32 sinusoidal embedding of shape (B, dim).
    """
    half = dim //2
    i = torch.arange(half, dtype = torch.float32)
    log_freq = -1 * math.log(max_period) * i / half 
    freqs = torch.exp(log_freq)
    angles = timesteps[:, None]  * freqs[None, :]  # (B, half)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    embedding = torch.cat([cos, sin], dim = -1)

    if dim  % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim = -1)
    return embedding