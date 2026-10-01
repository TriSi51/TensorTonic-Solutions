import torch
import torch.nn.functional as F
def groupwise_int4_quantize(
    weight: torch.Tensor,
    group_size: int,
) -> tuple:
    """
    Returns (codes, scales, dequantized): int8 codes, float row-group scales, float reconstruction.
    """
    if weight.ndim != 2:
        raise ValueError("W must be 2D matrix")
    if group_size <= 0:
        raise ValueError("group size must be positive")

    R, C = weight.shape
    num_groups = (C + group_size -1) // group_size

    if C == 0:
        return (
            weight.to(torch.int8),
            weight.new_empty((R,0)),
            weight.clone(),
        )

    padded_cols = num_groups * group_size
    W_padded = F.pad(weight, (0, padded_cols -C))

    W_groups = W_padded.reshape(R, num_groups, group_size)

    max_abs = W_groups.abs().amax(dim = -1)
    scales = torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        max_abs/ 7,
    )

    quantized_groups = torch.clamp(
        torch.round(W_groups / scales.unsqueeze(-1)),
        -7,
        7
    ).to(torch.int8)

    dequantized_groups = quantized_groups.to(weight.dtype) * scales.unsqueeze(-1)

    quantized = quantized_groups.reshape(R, padded_cols)[:, :C]
    dequantized = dequantized_groups.reshape(R, padded_cols)[:, :C]

    return quantized, scales, dequantized
    
    
