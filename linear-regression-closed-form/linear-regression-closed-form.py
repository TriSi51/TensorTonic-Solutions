import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    # Write code here
    X = np.asarray(X, dtype = np.float64)
    y = np.asarray(y, dtype = np.float64)

    inverse = np.linalg.inv(X.T @ X)
    w = (inverse @ X.T) @ y
    w_list = w.tolist()
    return [float(value) for value in w]