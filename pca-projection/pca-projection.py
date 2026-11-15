import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    # Write code here
    X = np.asarray(X, dtype = np.float64)
    n, _ = X.shape
    
    X_mean = np.mean(X, axis = 0, keepdims = True)
    X_c = X - X_mean
    Cov_X = (1/(n-1)) * (X_c.T @ X_c)

        
    vals, vecs = np.linalg.eigh(Cov_X)
    order = np.argsort(vals)[::-1]

    vecs = vecs[:, order]
    vecs = vecs[:, :k]

    X_proj = X_c @ vecs
    return X_proj.tolist()
    