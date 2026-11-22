import math
import torch
import torch.nn.functional as F

def tiny_unet_forward(x: torch.Tensor, timesteps: torch.Tensor, params: dict, num_groups: int) -> torch.Tensor:
    """
    Returns the network output of shape (B, C_out, H, W) with the dtype of x.
    """
    E = params["time_linear1_weight"].shape[1]
    half = E // 2

    i = torch.arange(
        half, 
        device = x.device,
        dtype = x.dtype
    )
    
    freq = torch.exp(
        -math.log(10000.0) * i / half
    )

    angles = timesteps[:, None] * freq[None, :] # (B,E/2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    time_embedding = torch.cat([cos, sin], dim = -1)  # (B, E)

    h = F.linear(
        time_embedding,
        params["time_linear1_weight"],
        params["time_linear1_bias"]
    )

    h = F.silu(h)
    e = F.linear(
        h,
        params["time_linear2_weight"],
        params["time_linear2_bias"]
    )  # (B, T)

    def res_block(h,e ,prefix):
        a = F.group_norm(
            h, num_groups,
            params[f"{prefix}_norm1_weight"],
            params[f"{prefix}_norm1_bias"],
            eps = 1e-5
        )

        a = F.silu(a)

        a = F.conv2d(
            a,
            params[f"{prefix}_conv1_weight"],
            params[f"{prefix}_conv1_bias"],
            padding = 1
        )

        time_shift = F.linear(
            F.silu(e),
            params[f"{prefix}_time_weight"],
            params[f"{prefix}_time_bias"]
        )  # (B,C)

        time_shift = time_shift[:, :, None, None]

        a= a +time_shift #(B, C, H, W)

        b = F.group_norm(
            a,
            num_groups,
            params[f"{prefix}_norm2_weight"],
            params[f"{prefix}_norm2_bias"],
            eps = 1e-5
        )

        b = F.silu(b)

        b = F.conv2d(
            b,
            params[f"{prefix}_conv2_weight"],
            params[f"{prefix}_conv2_bias"],
            padding = 1
        )  # B,C, H, W
        skip = h
        if f"{prefix}_skip_weight" in params:
            skip = F.conv2d(h,
                            params[f"{prefix}_skip_weight"],
                            params[f"{prefix}_skip_bias"]
                           ) 

        return skip + b

    h  = F.conv2d(
        x,
        params["conv_in_weight"],
        params["conv_in_bias"],
        padding = 1
    )

    h1 = res_block(h, e, "down_res")

    d = F.conv2d(
        h1,
        params["downsample_weight"],
        params["downsample_bias"],
        stride = 2,
        padding = 1
    )

    u = res_block(d, e, "mid_res")

    u = F.interpolate(
        u,
        scale_factor = 2,
        mode = "nearest"
    )

    u = F.conv2d(
        u,
        params["upsample_weight"],
        params["upsample_bias"],
        padding = 1
    )

    joined = torch.cat(
        [u, h1],
        dim = 1
    )
    
    h2 = res_block(joined, e, "up_res")

    h = F.group_norm(
        h2,
        num_groups,
        params["out_norm_weight"],
        params["out_norm_bias"],
        eps = 1e-5
    )

    h = F.silu(h)
    out = F.conv2d(
        h,
        params["conv_out_weight"],
        params["conv_out_bias"],
        padding = 1
    )

    return out
        
    
 
    