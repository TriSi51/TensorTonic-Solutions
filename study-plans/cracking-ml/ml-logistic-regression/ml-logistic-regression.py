import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype = np.float64)
    y = np.asarray(y, dtype = np.float64)

    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype = np.float64)

    bias = 0.0

    for _ in range(n_iters):
        logits = X@weights + bias
        clipped_logits = np.clip(logits, -500, 500)

        probabilities = 1.0 / (1.0 + np.exp(-clipped_logits))

        errors = probabilities - y

        dw = (X.T @ errors) / n_samples
        db = np.sum(errors) / n_samples

        weights -= lr * dw 
        bias -= lr * db
    return weights.round(4).tolist(), round(bias, 4)
    
