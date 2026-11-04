def tokenize(text: str) -> list:
    """
    Returns a list of token strings.
    """
    output = []
    current = []
    # Check normal char, then special case...

    for ch in text: 
        is_word_char = (
            "a" <= ch <= "z" 
            or "A" <= ch <= "Z"
            or "0" <= ch <= "9"
            or ch == "_"
        )

        if is_word_char:
            current.append(ch)
        else:
            if current:
                output.append("".join(current))
                current = []

            if not ch.isspace():
                output.append(ch)

    if current:
        output.append("".join(current))
    return output
        