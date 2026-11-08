import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a = np.asarray(a, dtype = np.float64)
    b = np.asarray(b, dtype = np.float64)
    
    
    product = np.dot(a,b)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a==0 or norm_b ==0:
        return 0.0
    result = product / (norm_a * norm_b)

    return float(result)