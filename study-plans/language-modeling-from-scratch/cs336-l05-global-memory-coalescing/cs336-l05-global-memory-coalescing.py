import numpy as np

def coalescing_analysis(byte_addresses, access_width_bytes=4, cache_line_bytes=128):
    """
    Returns: dictionary containing line IDs, transaction count, and useful-byte fraction
    """

    addresses = np.asarray(byte_addresses, dtype = np.int64)
    start_lines = np.floor_divide(addresses, cache_line_bytes)
    end_lines = np.floor_divide(addresses + access_width_bytes -1, cache_line_bytes)
    all_lines = np.concatenate([
        np.arange(start, end +1) 
        for start, end in zip(start_lines,end_lines)
    ])

    line_ids = np.unique(all_lines).astype(np.int64)

    C = len(line_ids)

    all_bytes = np.concatenate([
        np.arange(a, a+access_width_bytes)
        for a in addresses
    ])

    U = len(np.unique(all_bytes))
    useful_byte_fraction = U / (C * cache_line_bytes)
    return {
        "line_ids": line_ids,
        "transaction_count": int(C),
        "useful_byte_fraction": useful_byte_fraction,
    }
    