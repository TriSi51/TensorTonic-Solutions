import numpy as np

def vae_encoder(x: np.ndarray, W_mu: np.ndarray, b_mu: np.ndarray,
                W_logvar: np.ndarray, b_logvar: np.ndarray) -> dict:
    """
    Returns mu and log_var as float64 arrays in a dictionary.
    """
    mu= x  @ W_mu + b_mu
    logvar = x @ W_logvar + b_logvar

    return {
        "mu": mu,
        "log_var":logvar
    }
