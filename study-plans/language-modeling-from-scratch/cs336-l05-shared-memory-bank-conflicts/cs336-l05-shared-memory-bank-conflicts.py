import numpy as np

def bank_conflict_analysis(byte_addresses, num_banks=32, bank_width_bytes=4):
    """
    Returns: dictionary containing bank IDs and conflict degrees
    """
    # np.floor_divide(byte_addresses, bank_width_bytes)
    bank_ids = np.mod(np.floor_divide(byte_addresses, bank_width_bytes), num_banks)
    conflict_degree = np.empty(len(bank_ids), dtype  = np.int64)
    for b in bank_ids:
        mask = bank_ids == b
        degree = len(np.unique(byte_addresses[mask]))
        conflict_degree[mask]= degree

    return {
        "bank_ids": bank_ids,
        "conflict_degree": conflict_degree
    }