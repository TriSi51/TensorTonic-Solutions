import numpy as np

def channel_statistics(batch: list) -> dict:
    """
    Returns mean and std lists of C floats in a dictionary, rounded to four decimals.
    """
    batch_np = np.asarray(batch, dtype = np.float64)

    N,H, W,C = batch_np.shape

    mean_c = np.mean(batch_np, axis = (0,1,2))

    shift_sq = (batch_np - mean_c) **2
    std = np.sqrt(np.mean(shift_sq, axis = (0,1,2)))
    mean_c = mean_c.tolist()
    std = std.tolist()
    return {
        "mean": [round(value, 4) for value in mean_c],
        "std": [ round(value,4) for value in std]
    }
    