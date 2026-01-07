import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_metrics(y_true, y_pred_probs):
    y_pred = np.argmax(y_pred_probs, axis=1)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    auc = roc_auc_score(y_true, y_pred_probs[:, 1])

    return specificity, sensitivity, f1, auc
