import torch

def blockwise_fp8_quantize(
    x: torch.Tensor,
    row_block_size: int,
    col_block_size: int,
    fp8_max: float,
) -> tuple:
    """
    Returns (scaled, block_scales, dequantized), floating tensors with one scale per 2D block.
    """
    if x.ndim != 2:
        raise ValueError("x must be a 2D matrix")
    if row_block_size <= 0 or col_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if fp8_max <= 0:
        raise ValueError("fp8_max must be positive")

    x_float = x.float()
    rows, cols = x.shape

    n_row_blocks = (rows + row_block_size -1) // row_block_size
    n_col_blocks = (cols + col_block_size -1) // col_block_size

    scaled = torch.empty_like(x_float)
    dequantized = torch.empty_like(x_float)

    block_scales = torch.empty(
        (n_row_blocks, n_col_blocks),
        dtype = x_float.dtype,
        device = x.device
    )

    for bi in range(n_row_blocks):
        r0 = bi * row_block_size
        r1 = min(r0 + row_block_size, rows)

        for bj in range(n_col_blocks):
            c0 = bj * col_block_size
            c1 = min(c0 + col_block_size, cols)

            block = x_float[r0:r1, c0:c1]
            absmax = block.abs().max()

            scale =absmax / fp8_max
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            
            proxy = torch.clamp(block / scale, -fp8_max, fp8_max)

            scaled[r0:r1, c0:c1] = proxy
            dequantized[r0:r1, c0:c1] = proxy * scale
            block_scales[bi,bj]= scale
    return scaled, block_scales, dequantized
                                                            