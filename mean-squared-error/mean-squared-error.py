import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    # Write code here
    y_pred = np.asarray(y_pred, dtype = np.float64)
    y_true = np.asarray(y_true, dtype = np.float64)

    result = np.mean((y_pred - y_true) ** 2, axis = -1)
    return float(result)