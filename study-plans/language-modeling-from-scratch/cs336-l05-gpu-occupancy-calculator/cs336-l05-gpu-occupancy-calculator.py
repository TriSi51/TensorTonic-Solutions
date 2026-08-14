def gpu_occupancy(threads_per_block, registers_per_thread, shared_mem_per_block,
                  max_threads_per_sm, max_warps_per_sm, max_blocks_per_sm,
                  max_registers_per_sm, max_shared_mem_per_sm, warp_size=32):
    """
    Returns: dictionary containing resident blocks, resident warps, and occupancy
    """


    warps_per_block = (
        threads_per_block + warp_size -1
    ) // warp_size

    # Hardware allocates whole warps
    effective_threads = warps_per_block * warp_size

    # Thread constraint
    blocks_by_threads = (
        max_threads_per_sm // threads_per_block
    )

    # Warp constraint
    blocks_by_warps = (
        max_warps_per_sm // warps_per_block
    )

    # Register constraint
    if registers_per_thread == 0:
        blocks_by_registers = max_blocks_per_sm
    else:
        registers_per_block = (
            registers_per_thread * effective_threads
        )

        blocks_by_registers = (
            max_registers_per_sm // registers_per_block
        )


    if shared_mem_per_block == 0:
        blocks_by_shared_mem = max_blocks_per_sm
    else:
        blocks_by_shared_mem = (
            max_shared_mem_per_sm // shared_mem_per_block
         )
    blocks_per_sm = min(
        blocks_by_threads,
        blocks_by_warps,
        blocks_by_registers,
        max_blocks_per_sm,
        blocks_by_shared_mem
    )

    resident_warps = blocks_per_sm * warps_per_block
    occupancy = resident_warps / max_warps_per_sm

    return {
        "blocks_per_sm": blocks_per_sm,
        "resident_warps": resident_warps,
        "occupancy": occupancy 
    }
    
