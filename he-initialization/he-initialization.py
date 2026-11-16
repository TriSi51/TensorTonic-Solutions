import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    # Write code here
    L  = math.sqrt(6.0 / float(fan_in))


    width = len(W)
    length = len(W[0])

    for i in range(width):
        for j in range(length):
            W[i][j] = W[i][j] * 2* L  -L

    return W