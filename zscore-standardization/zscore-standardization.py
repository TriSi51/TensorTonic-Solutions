import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    # Write code here
    X= np.asarray(X, dtype = np.float64)
    mean = np.mean(X, axis = axis, keepdims= True )
    std = np.std(X, axis = axis,ddof = 0,  keepdims = True)
    
    safe_std = np.where (std > eps, std, 1)

    z_score = (X- mean ) / safe_std

    return np.where(std > eps, z_score, 0)
    