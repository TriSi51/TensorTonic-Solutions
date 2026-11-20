import torch

def forward_diffusion_sample(x0: torch.Tensor, timesteps: torch.Tensor, alpha_bars: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Returns the noisy samples x_t with the same shape and dtype as x0.
    """
    alpha_bar_t = alpha_bars[timesteps]

    shape = (x0.shape[0],) + (1,) * (x0.ndim -1)

    alpha_bar_t= alpha_bar_t.reshape(shape)
    return x0 * torch.sqrt(alpha_bar_t) + noise * torch.sqrt((1.0 - alpha_bar_t))
