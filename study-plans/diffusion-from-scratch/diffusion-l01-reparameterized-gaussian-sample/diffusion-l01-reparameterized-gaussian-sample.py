import numpy as np

def reparameterized_sample(mean: np.ndarray, log_variance: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """
    Returns the reparameterized samples as a float64 array with the broadcast shape of the inputs.
    """
    variance = np.exp(0.5 * log_variance)
    var_noise = variance * noise
    # np.expand_dims(x, dim = ?)  or x[None, :] x[:, None]

    return (mean + var_noise)
    