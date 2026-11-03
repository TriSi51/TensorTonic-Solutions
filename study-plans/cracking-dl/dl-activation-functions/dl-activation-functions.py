import numpy as np

def activation_functions(x: float, activation: str) -> list:
    """
    Returns the activation value and analytical derivative.
    """
    if activation == "relu":
        y = np.maximum(0.0, x)
        dy = 1.0 if x > 0 else 0.0

    elif activation == "leaky_relu":
        alpha = 0.01
        y = np.where(x >= 0, x, alpha * x)
        dy = 1.0 if x > 0 else alpha

    elif activation == "sigmoid":
        if x >= 0:
            s = 1 / (1 + np.exp(-x))
        else:
            e = np.exp(x)
            s = e / (1 + e)

        y = s
        dy = s * (1 - s)

    elif activation == "tanh":
        y = np.tanh(x)
        dy = 1 - y**2

    elif activation == "gelu":
        c = np.sqrt(2 / np.pi)
        u = c * (x + 0.044715 * x**3)
        t = np.tanh(u)

        y = 0.5 * x * (1 + t)
        dy = (
            0.5 * (1 + t)
            + 0.5 * x * (1 - t**2)
            * c * (1 + 3 * 0.044715 * x**2)
        )

    elif activation == "swish":
        if x >= 0:
            s = 1 / (1 + np.exp(-x))
        else:
            e = np.exp(x)
            s = e / (1 + e)

        y = x * s
        dy = s + x * s * (1 - s)

    else:
        raise ValueError("Unknown activation")

    return [round(float(y), 4), round(float(dy), 4)]