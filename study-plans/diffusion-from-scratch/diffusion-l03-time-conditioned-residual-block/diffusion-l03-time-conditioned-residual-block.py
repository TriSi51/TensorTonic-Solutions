import torch
import torch.nn.functional as F

def time_residual_block(x: torch.Tensor, time_embedding: torch.Tensor, params: dict, num_groups: int) -> torch.Tensor:
    """
    Returns the block output of shape (B, C_out, H, W) with the dtype of x.
    """
    time_shift = F.linear(F.silu(time_embedding), params["time_weight"], params["time_bias"])
    group_norm_1 = F.group_norm(x, num_groups, params["norm1_weight"], params["norm1_bias"])
    h =  F.conv2d(F.silu(group_norm_1), params["conv1_weight"], params["conv1_bias"], padding = 1)
    h = h + time_shift[:,:, None, None]
    group_norm_2 = F.group_norm(h,num_groups, params["norm2_weight"], params["norm2_bias"])
    h = F.conv2d(F.silu(group_norm_2), params["conv2_weight"], params["conv2_bias"], padding = 1)
    x_skip = x
    if "skip_weight" in params:
        x_skip = F.conv2d(x, params["skip_weight"], params["skip_bias"])

    return x_skip + h
    
