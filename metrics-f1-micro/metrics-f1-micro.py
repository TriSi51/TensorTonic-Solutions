def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Return the micro-averaged F1 score rounded to four decimals.
    """
    # Write code here
    # True positive 
    TP = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
    FP = len(y_true) - TP
    FN = FP
    f1_micro = 2.0* TP / (2.0* TP + FP +FN )
    return round(f1_micro, 4)