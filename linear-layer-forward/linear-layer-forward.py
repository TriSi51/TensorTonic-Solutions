def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    # Write code here
    Y = []
    for i in range(len(X)):
        output_2=[]
        for d_out in range(len(W[0])):
            total = 0.0
            for d_in in range(len(W)):
                total += X[i][d_in] * W[d_in][d_out]
            total += b[d_out]

            output_2.append(total)

        Y.append(output_2)

    return Y
                