import numpy as np

def gaussian_mixture_score(x: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """
    Returns the score grad_x log p(x) of the isotropic Gaussian mixture at every row of x, as a float64 array of shape (N, D).
    """

    D = x.shape[1]
    diff = x[:, None, :] - means[None, :, :]
    norm = np.sum(diff ** 2, axis = -1)  # N, K
    log_term_2 = - norm / (2 * variances[None, :])

    log_term_1 = -D/2 * np.log(2 * np.pi * variances[None, :])

    log_term = np.log(weights)[None, :] + log_term_1 + log_term_2

    log_term_new = log_term - np.max(log_term, axis = -1, keepdims = True)
    deno = np.sum(np.exp(log_term_new), axis = -1, keepdims = True)
    res = np.exp(log_term_new) / deno
    
    C= res / variances[None, :]

    return C @ means - np.sum(C, axis = -1,keepdims = True) * x

    