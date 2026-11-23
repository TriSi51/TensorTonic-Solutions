import torch

def noise_prediction_loss(x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor, weight: torch.Tensor, time_weight: torch.Tensor, bias: torch.Tensor) -> dict:
    """
    Returns a dict with the batch loss as a Python float under "loss" and the per-sample losses of shape (B,) under "per_sample_loss".
    """
    B = x0.shape[0]
    x0_flat = x0.reshape(B, -1)  # B, D
    noise_flat = noise.reshape(B, -1)  # B, D
    D = x0_flat.shape[1]

    alpha_bars_t = alpha_bars[timesteps]  # B,
    alpha_bars_t = alpha_bars_t[:, None]  # B, 1

    x_t = torch.sqrt(alpha_bars_t) * x0_flat + torch.sqrt(1.0 - alpha_bars_t) * noise_flat

    noise_pred = x_t @  weight+ (timesteps.to(x0.dtype) / alpha_bars.shape[0])[:, None] * time_weight + bias
    loss =  (noise_pred - noise_flat) ** 2
    loss = loss.mean(dim = 1)
    loss_b =loss.mean(dim = 0)
    return {
        "loss": float(loss_b),
        "per_sample_loss": loss
    }

