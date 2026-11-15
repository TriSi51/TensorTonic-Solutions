import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    # Write code here
    x = np.asarray(x, dtype = np.float64 )

    x= np.where(x == 1, p, 1.0-p)
    mean = p
    variance = p*(1-p)
    return {
        "pmf": x,
        "mean": float(mean),
        "variance": float(variance),
    }

    