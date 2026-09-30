import torch

def symmetric_int8_quantize(
    x: torch.Tensor,
) -> tuple:
    """
    Returns (quantized, scale, dequantized): int8 codes, scalar float scale, float reconstruction.
    """
    max_abs = x.abs().max() 
    if max_abs == 0:
        scale = torch.ones((), dtype = x.dtype, device = x.device)
        quantized = torch.zeros_like(x, dtype = torch.int8)
        dequantized = torch.zeros_like(x)
        return quantized , scale, dequantized

    scale = max_abs / 127

    quantized = torch.clamp(
        torch.round(x/ scale),
        -127,
        127
    ).to(torch.int8)

    dequantized = quantized.to(x.dtype) * scale
    return quantized, scale, dequantized
        
