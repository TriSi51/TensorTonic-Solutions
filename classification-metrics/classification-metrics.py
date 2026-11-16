import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype = np.float64)
    y_pred = np.asarray(y_pred, dtype = np.float64)
    
    all_cls = np.unique(np.concatenate([y_pred, y_true]))

    results = [] 
    for c in all_cls:
        true_positive_c=  ((y_pred == c) & (y_true == c)).sum()
        full_positive_pred_c = (y_pred == c).sum()
    
        full_positive_gt_c = (y_true == c).sum()

        support = (y_true == c).sum()
        precision_c = true_positive_c / full_positive_pred_c
        recall_c = true_positive_c / full_positive_gt_c
        f1_c = np.float64(0.0)
        if (recall_c + precision_c != 0):
            f1_c = 2 * precision_c * recall_c / (precision_c + recall_c)
        results.append([c,true_positive_c, full_positive_pred_c, full_positive_gt_c, precision_c, recall_c, f1_c])


    accuracy = (y_pred == y_true).sum() / y_pred.shape[0]
    
    precision = np.float64(0.0)
    recall = np.float64(0.0)
    F1 = np.float64(0.0)
    if average == "micro":
        true_positive = np.float64(0.0)
        full_positive_pred = np.float64(0.0)
        full_positive_gt = np.float64(0.0)

        for result in results:
            true_positive += result[1]
            full_positive_pred += result[2]
            full_positive_gt += result[3]

        precision = true_positive / full_positive_pred
        recall = true_positive / full_positive_gt
        if (precision + recall != 0):
            F1 = 2 * precision * recall / (precision + recall)

        
    elif average == "macro":

        for result in results:
            precision += result[4]
            recall += result[5]
            F1 += result[6]

        precision /= len(results)
        recall /= len(results)
        F1 /= len(results)

    elif average == "weighted":
        support = np.float64(0.0)

        for result in results:
            support += result[3]

        for result in results:
            precision += result[4] * result[3] / support
            recall += result[5] * result[3] / support
            F1 += result[6] * result[3] / support
    else:
        for result in results:
            if result[0] == pos_label:
                precision = result[4]
                recall = result[5]
                F1 = result[6]
                break

    return {
        "accuracy": round(float(accuracy),6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall),6),
        "f1": round(float(F1),6)
    }

    
        