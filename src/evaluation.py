
import numpy as np
import sklearn.metrics as metr
from typing import *


def evaluate_class_predictions(prediction: np.ndarray,
                               ground_truth: np.ndarray,
                               multiclass: bool = False,
                               verbosity: bool = False) -> Tuple[float, ...]:
    """
    Function for evaluating the performance of a model that returns classes as predictions (not class probabilities).
    Works for binary case and multi-class case.
    :param prediction: Class predictions of the model.
    :param ground_truth: Ground truth classes.
    :param multiclass: Whether metrics are computed for multiclass labels, otherwise assume binary labels.
    :param verbosity: Whether to print computed performance metrics, or not.
    :return: None
    """
    if not multiclass:
        accuracy = metr.accuracy_score(y_true=ground_truth, y_pred=prediction)
        precision = metr.precision_score(y_true=ground_truth, y_pred=prediction)
        recall = metr.recall_score(y_true=ground_truth, y_pred=prediction)
        f1 = metr.f1_score(y_true=ground_truth, y_pred=prediction)
    else:
        labels = np.unique(ground_truth)
        accuracy = metr.accuracy_score(y_true=ground_truth, y_pred=prediction)
        precision = metr.precision_score(y_true=ground_truth, y_pred=prediction,
                                         labels=labels, average='micro')
        recall = metr.recall_score(y_true=ground_truth, y_pred=prediction,
                                   labels=labels, average='micro')
        f1 = metr.f1_score(y_true=ground_truth, y_pred=prediction,
                           labels=labels, average='micro')

    if verbosity:
        print(f"## Accuracy: {accuracy}")
        print(f"## Precision: {precision}")
        print(f"## Recall: {recall}")
        print(f"## F1-score: {f1}")

    return accuracy, precision, recall, f1

