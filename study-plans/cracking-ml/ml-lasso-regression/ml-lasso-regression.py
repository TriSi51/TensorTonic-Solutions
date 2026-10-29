import numpy as np

def lasso_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype = np.float64)
    y = np.asarray(y, dtype = np.float64)

    n_samples, n_features = X.shape

    w = np.zeros(n_features, dtype = np.float64)
    b = np.float64(0.0)

    for epoch in range(epochs):
        pred = X @ w + b
        error = pred - y 
        
        grad_w = (2/ n_samples) * (X.T @ error) + (alpha * np.sign(w))
        grad_b = (2/ n_samples) * np.sum(error)

        w = w - lr * grad_w
        b = b - lr * grad_b
    return np.round(w, 4).tolist(), np.round(float(b), 4)