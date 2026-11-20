import numpy as np

def cosine_noise_schedule(num_steps: int, s: float, max_beta: float) -> dict:
    """
    Returns a dict with float64 arrays "betas" and "alpha_bars", each of shape (num_steps,).
    """
    steps = np.arange(num_steps +1, dtype = np.float64)
    s= np.float64(s)
    max_beta = np.float64(max_beta)
    num_steps = np.float64(num_steps)
    
    f_step = np.cos(  (steps/ num_steps + s) / (1+ s) * np.pi /2) **2  # (step) 
    print(f_step)
    betas = np.minimum(1.0 - f_step[1:] / f_step[:-1], max_beta)
    alpha_bars = np.cumprod(1.0 - betas)
    return {
        "betas": betas,
        "alpha_bars": alpha_bars

    }

