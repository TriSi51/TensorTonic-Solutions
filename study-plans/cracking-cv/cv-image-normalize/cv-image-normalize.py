def normalize_image(image: list, mean: list, std: list) -> list:
    """
    Returns the normalized H-by-W-by-C list of floats, rounded to four decimals.
    """
    import numpy as np 
    image = np.asarray(image, dtype = np.float64)
    mean = np.asarray(mean, dtype= np.float64)
    std = np.asarray(std, dtype = np.float64)
    result = ((image - mean ) / std).tolist()

    return [
        [[round(value, 4) for value in pixel] for pixel in row ]
        for row in result    
    ]


