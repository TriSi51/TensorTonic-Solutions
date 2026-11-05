import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    """
    Returns a float32 tensor with the requested shape.
    """
    result = 0
    if method == "zeros":
        result = torch.zeros(shape, dtype = torch.float32)
    elif method == "ones":
        result = torch.ones(shape, dtype = torch.float32)
    elif method =="full":
        result = torch.full(shape, fill_value = value, dtype = torch.float32)
    return result
        
