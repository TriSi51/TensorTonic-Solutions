import math
import numpy as np 
def sigmoid_focal_loss(logits: list, targets: list, alpha: float, gamma: float) -> float:
    """
    Returns the mean focal loss as a float, rounded to 4 decimals.
    """
    z = np.asarray(logits,dtype = np.float64)
    y = np.asarray(targets, dtype = np.float64)

    s = 2 * y - 1
    t = s*z
    # - log(pt)
    ce = np.logaddexp(0, -t)

    # (1 - pt)^sigma
    focal_weight  = np.exp(-gamma * np.logaddexp(0, t))

    alpha = y * alpha + (1-y)* (1-alpha)

    loss = alpha * focal_weight * ce

    return round(float(np.mean(loss)), 4)

    