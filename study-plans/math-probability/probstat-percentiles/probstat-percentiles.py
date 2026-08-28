import numpy as np

def percentiles(x, q):
    """
    Returns: numpy array of percentile values.
    """
    
    # sorted? 
    return np.percentile(x,q)
    
    