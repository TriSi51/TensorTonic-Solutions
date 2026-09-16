import torch
from typing import List, Tuple

def match_prefix_cache(
    request_token_ids: List[int],
    cached_token_blocks: List[List[List[int]]],
    cached_physical_block_ids: List[List[int]],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (matched_token_count scalar tensor, reusable_physical_block_ids tensor)
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if len(cached_token_blocks) != len(cached_physical_block_ids):
        raise ValueError(
            "cached_token_blocks and cached_physical_block_ids "
            "must contain the same number of candidates"
        )

    num_complete_request_blocks = len(request_token_ids) // block_size

    best_match_blocks = 0
    best_physical_ids: List[int] = []

    for candidate_blocks, candidate_physical_ids in zip(
        cached_token_blocks,
        cached_physical_block_ids,
    ):
        max_comparable_blocks = min(
            num_complete_request_blocks,
            len(candidate_blocks),
            len(candidate_physical_ids)
        )

        matched_blocks = 0

        for block_index in range(max_comparable_blocks):
            request_start = block_index * block_size
            request_end = request_start + block_size

            request_block = request_token_ids[request_start:request_end]

            cached_block = candidate_blocks[block_index]

            if len(cached_block) != block_size:
                break
            if request_block != cached_block:
                break

            matched_blocks += 1


        if matched_blocks > best_match_blocks:
            best_match_blocks = matched_blocks
            best_physical_ids = candidate_physical_ids[:matched_blocks]

    matched_token_count = torch.tensor(
        best_match_blocks * block_size,
        dtype = torch.int64
    )

    reusable_physical_block_ids = torch.tensor(
        best_physical_ids,
        dtype = torch.int64
    )

    return matched_token_count, reusable_physical_block_ids