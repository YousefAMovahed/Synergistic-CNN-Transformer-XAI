import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

class MetricTracker:
    """
    Tracks and computes metrics specifically focused on:
    1. Balanced Accuracy (Primary metric in the paper)
    2. Overall Accuracy
    3. Macro F1-Score
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, preds, labels):
        """
        Args:
            preds: Tensor of predicted class indices
            labels: Tensor of true class indices
        """
        self.y_pred.extend(preds.cpu().numpy())
        self.y_true.extend(labels.cpu().numpy())

    def compute(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        bal_acc = balanced_accuracy_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred, average='macro')
        
        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": f1
        }

    def get_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)
