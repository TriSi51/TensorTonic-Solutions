import numpy as np

def ridge_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X= np.asarray(X, dtype = np.float64)
    y = np.asarray(y, dtype = np.float64)

    n_samples, n_features = X.shape

    w = np.zeros(n_features, dtype = np.float64)
    b = np.float64(0.0)

    for epoch in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y

        grad_w = (2.0 / n_samples) * (X.T @ error) + 2 * alpha * w 
        grad_b = (2.0 / n_samples) * np.sum(error)

        w -= lr * grad_w
        b -= lr * grad_b

    return np.round(w, 4).tolist(), round(float(b), 4)