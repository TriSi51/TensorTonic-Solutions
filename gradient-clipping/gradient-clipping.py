import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    # Write code here
    g = np.asarray(g)
    g_norm = np.linalg.norm(g)
    if float(g_norm) <= max_norm:
        return g
    else:
        return g * max_norm / g_norm