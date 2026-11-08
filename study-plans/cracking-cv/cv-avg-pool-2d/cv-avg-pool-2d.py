def avg_pool_2d(image: list, kernel_size: int, stride: int) -> list:
    """
    Returns the average-pooled 2D list of floats, rounded to four decimals.
    """

    H,W = len(image), len(image[0])

    new_h = (H-kernel_size) // stride +1 
    new_w = (W-kernel_size) // stride + 1
    
    
    results = []
    for i in range(new_h):
        width  = []
        for j in range(new_w):
            total = sum(
                image[i * stride + dx][j * stride + dy]
                for dx in range(kernel_size)
                for dy in range(kernel_size)
            )

            mean = round(total / (kernel_size*kernel_size), 4)
            width.append(mean)

        results.append(width)
    return results

            
            
    