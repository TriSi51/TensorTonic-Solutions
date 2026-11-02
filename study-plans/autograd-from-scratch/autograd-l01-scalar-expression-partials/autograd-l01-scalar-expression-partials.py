def scalar_expression_partials(a: float, b: float, c: float, h: float) -> tuple[float, float, float, float]:
    """
    Returns the expression value and numerical partials for a, b, and c.
    """
    d_original = a* b + c

    inputs = [a,b,c]
    partials = []

    for i, input in enumerate(inputs):
        pertubed = inputs.copy()

        pertubed[i] += h

        d_pertubed = pertubed[0] * pertubed[1] + pertubed[2]

        partial = (d_pertubed - d_original) / h 
        partials.append(partial)
    return float(d_original), *(partial for partial in partials)
        
        