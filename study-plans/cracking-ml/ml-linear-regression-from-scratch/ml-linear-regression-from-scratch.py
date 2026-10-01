import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype = np.float64)
    y = np.asarray(y, dtype = np.float64)

    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype = np.float64)
    b = 0.0

    for _ in range(epochs):
        predictions = X @ w + b
        errors = predictions - y

        grad_w = (2.0 / n_samples) * (X.T @ errors)
        grad_b = 2.0 * errors.mean()

        w -= lr * grad_w
        b -= lr* grad_b
    return np.round(w,4).tolist(), round(float(b), 4)
