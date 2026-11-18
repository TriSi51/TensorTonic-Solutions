import numpy as np

def diagonal_gaussian_kl(mean_p: np.ndarray, log_var_p: np.ndarray, mean_q: np.ndarray, log_var_q: np.ndarray) -> np.ndarray:
    """
    Returns KL(p || q) summed over the last axis, as a float64 array with the leading shape of the inputs.
    """

    log_diff = log_var_q - log_var_p
    ratio_var = np.exp(-log_diff)
    mean_term = (mean_p - mean_q) **2 / np.exp(log_var_q)

    return 0.5 * np.sum(log_diff + ratio_var+mean_term -1, axis = -1)