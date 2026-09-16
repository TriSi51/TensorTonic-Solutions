import torch
from typing import List, Tuple

def allocate_kv_blocks(
    seq_lengths: List[int],
    block_size: int,
    free_block_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (block_table, blocks_used, remaining_free_blocks)
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if any(length < 0 for length in seq_lengths):
        raise ValueError("sequence lengths must be non-negative")

    # 1. Calculate number of blocks needed per sequence
    # ceil(L / B) = (L + B -1) // B
    blocks_needed = [
        (length + block_size - 1) // block_size
        for length in seq_lengths
    ]

    total_needed = sum(blocks_needed)

    # 2. All-or-nothing check
    if total_needed > len(free_block_ids):
        raise RuntimeError("Not enough free blocks")

    # 3. Create block table

    num_sequences = len(seq_lengths)
    max_blocks = max(blocks_needed, default = 0)
    block_table = torch.full(
        (num_sequences, max_blocks),
        -1,
        dtype = torch.long,
    )

    # 4. Allocate blocks in free-pool order
    offset = 0
    for i, num_blocks in enumerate(blocks_needed):
        if num_blocks > 0:
            allocated = free_block_ids[offset: offset+num_blocks]

            block_table[i, :num_blocks] = torch.tensor(
                allocated, 
                dtype = torch.long
            )

            offset += num_blocks

    blocks_used = torch.tensor(
        blocks_needed,
        dtype = torch.long,
    )

    remaining_free_blocks = torch.tensor(
        free_block_ids[offset:],
        dtype = torch.long
    )

    return block_table, blocks_used, remaining_free_blocks
