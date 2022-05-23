import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def calc_recall(tp, fn):
    """
    True positive rate (TPR, also called sensitivity) 
    """
    return tp/(tp+fn)


def calc_fpr(tn, fp):
    """
    False positive rate (also known as fall-out or false alarm ratio) 
    """
    return fp/(tn+fp)


def equalized_odds(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = calc_recall(tp, fn)
    fpr = calc_fpr(tn, fp)
    return recall, fpr
