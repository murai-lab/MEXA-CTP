from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

def CTPEval(all_predictions, all_decisions, all_labels):
    f1 = f1_score(all_labels, all_decisions)
    roc_auc = roc_auc_score(all_labels, all_predictions)
    pr_auc = average_precision_score(all_labels, all_predictions)
    return f1, roc_auc, pr_auc
