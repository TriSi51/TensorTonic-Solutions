def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # Write code here
    for step in range(steps):
        x0 = x0 -lr*(  2 * a *x0 + b)

    return x0