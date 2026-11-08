import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
        # Write code here
    A = np.asarray(A, dtype = np.float64)
    
    col, row = A.shape

    outputs = np.zeros((row,col), dtype = np.float64)

    for c in range(col):
        for r in range(row):
            outputs[r,c] = A[c,r]
    
    return outputs