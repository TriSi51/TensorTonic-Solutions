import numpy as np

def perceptron(X: list, y: list, lr: float = 0.1, epochs: int = 100) -> tuple:
    """
    Returns the trained weight list and bias.
    """
    X= np.asarray(X, dtype = np.float64)
    y = np.asarray(y, dtype = np.float64)
    
    n_samples, n_features = X.shape
    
    w = np.zeros(n_features, dtype = np.float64)
    bias= 0.0

    for _ in range(epochs):
        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            z = x_i @ w  + bias
            y_pred  = 1.0 if z >= 0 else 0.0

            error = y_i - y_pred
            w += lr * error * x_i
            bias += lr * error
    return w.round(4).tolist(), round(bias , 4)
            