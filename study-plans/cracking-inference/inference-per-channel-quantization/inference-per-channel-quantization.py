import torch

def per_channel_int8_quantize(
    x: torch.Tensor,
    channel_axis: int,
) -> tuple:
    """
    Returns (quantized, scale, dequantized): int8 codes, broadcastable float scales, float reconstruction.
    """
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")

    if not -x.ndim <= channel_axis < x.ndim:
        raise IndexError("channel_axis is out of range")

    channel_axis %= x.ndim
    
    reduce_dims = tuple(d for  d in range(x.ndim) if d != channel_axis)

    if reduce_dims:
        max_abs = x.abs().amax(dim = reduce_dims, keepdim = True)
    else:
        # For a  1D tensor, each element is its own channel
        max_abs = x.abs()

    scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs /127)

    quantized = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)

    dequantized = scale * quantized.to(x.dtype)
    return quantized, scale, dequantized