def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    # Write code here
    n_samples = len(X)
    d_in = len(W)
    d_out = len(W[0])

    return  [[sum(X[i][k] * W[k][j] for k in range(d_in)) + b[j] 
                for j in range(d_out)]
                for i in range(n_samples)]
             