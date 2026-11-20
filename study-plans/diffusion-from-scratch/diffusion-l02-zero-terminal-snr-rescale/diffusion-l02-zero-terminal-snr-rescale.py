import numpy as np

def rescale_zero_terminal_snr(betas: np.ndarray) -> dict:
    """
    Returns a dict with float64 arrays "betas", "alpha_bars" and "snr" of the rescaled schedule, each of shape (T,).
    """
    alpha_t = 1.0 - betas
    alpha_bars_t = np.cumprod(alpha_t)
    a_t = np.sqrt(alpha_bars_t)
    new_a_t = (a_t - a_t[-1]) * (a_t[0] / (a_t[0] - a_t[-1]))
    new_alpha_t = new_a_t ** 2

    previous_alpha_bars = np.concatenate(([1.0], new_alpha_t[:-1]))
    new_betas = 1.0 - new_alpha_t / previous_alpha_bars

    snr = new_alpha_t / (1.0 - new_alpha_t)
    return {
        "betas": new_betas,
        "alpha_bars": new_alpha_t,
        "snr": snr
    }
