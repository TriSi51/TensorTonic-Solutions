def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """

    total = {}
    count = {}

    for cat, t in zip(categories, targets):
        total[cat] = total.get(cat,0) + t
        count[cat] = count.get(cat, 0 ) + 1

    return [total[cat] / count[cat] for cat in categories]
        
        