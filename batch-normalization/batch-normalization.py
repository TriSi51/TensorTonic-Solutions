import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.asarray(x, dtype = np.float64)
    gamma =  np.asarray(gamma,dtype = np.float64)
    beta= np.asarray(beta,dtype = np.float64)
    eps = np.float64(eps)

    if x.ndim ==2:
        axes = (0,)
    else:
        axes = (0,2,3)
    x_mean = np.mean(x, axis =axes, keepdims = True)

    x_new = (x-x_mean)**2

    var = np.mean(x_new, axis = axes, keepdims = True)

    x_norm = (x- x_mean) / np.sqrt(var + eps)
    y_i  = np.array(0.0)
    if x.ndim == 4:
        y_i = gamma.reshape(1,x.shape[1],1,1) * x_norm + beta.reshape(1,x.shape[1],1,1)
    else:
        y_i = gamma * x_norm + beta
    
    return y_i
    
    