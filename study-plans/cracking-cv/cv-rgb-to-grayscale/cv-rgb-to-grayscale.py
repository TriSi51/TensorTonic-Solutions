def rgb_to_grayscale(image: list) -> list:
    """
    Returns the H-by-W grayscale list of floats, rounded to four decimals.
    """


    # h =[]
    
    # for row in image:
    #     w = []
    #     for pixel in row:
    #         r,g,b =  pixel
    #         convert =  0.299 * r + 0.587 * g + 0.114 * b
    #         round(convert,4)
    #         w.append(convert)
            

    #     h.append(w)

    # return h


    return [
        [round((0.299 * r + 0.587 * g + 0.114 * b),4) for r,g,b in row]
        for row in image
    ]