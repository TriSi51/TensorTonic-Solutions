import re

def text_normalize(text: str, operations: list) -> str:
    """
    Returns the normalized string.
    """
    output = text
    for operation in operations:
        if operation == "lowercase":
            output = output.lower()
            
    
        elif operation == "remove_punctuation":
            removed_punc = []
            for ch in output:
                if ch.isalnum() or ch.isspace():
                    removed_punc.append(ch)
            output = "".join(removed_punc)
            
        elif operation == "remove_digits":
            removed_digit = []

            for ch in output:
                if not ch.isdecimal():
                    removed_digit.append(ch)
            output = "".join(removed_digit)
            
        elif operation == "collapse_whitespace":
            output = re.sub(r"\s+", " ", output)
            
        elif operation == "strip":
            output = output.strip()
            
    return output
            
            