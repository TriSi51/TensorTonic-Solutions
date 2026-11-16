import math

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    # Write code here
    width = len(W)
    length = len(W[0])

    L = math.sqrt( 6.0 / (float(fan_in + fan_out)))
    
    for i in range(width):
        for j in range(length):
            W[i][j] = W[i][j] * 2 * L - L

    return W