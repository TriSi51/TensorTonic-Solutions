import numpy as np

def finite_difference_derivative(coefficients: list, x: float, h: float) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    
    co = np.asarray(coefficients, dtype = np.float64)
    x = np.float64(x)
    h = np.float64(h)

    def function(v):
        value = np.float64(0.0)

        for c in co[::-1]:
            value = value * v + c
            # print(value)

        return value

    fx = function(x)
    fxh = function(x+h)
    return float(fx) , float(fxh) , float((fxh -fx) /h)
    