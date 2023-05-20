
import numpy as np
import sklearn.metrics as metr
from typing import *


def evaluate_class_predictions(prediction: np.ndarray,
                               ground_truth: np.ndarray,
                               labels: np.ndarray,
                               verbosity: bool = False) -> Tuple[float, ...]:
    """
    Function for evaluating the performance of a model that returns classes as predictions (not class probabilities).
    Works for binary case and multi-class case.
    :param prediction: Class predictions of the model.
    :param ground_truth: Ground truth classes.
    :param labels: Array containing all possible unique labels..
    :param verbosity: Whether to print computed performance metrics, or not.
    :return: None
    """
    accuracy = metr.accuracy_score(y_true=ground_truth, y_pred=prediction)
    precision = metr.precision_score(y_true=ground_truth, y_pred=prediction,
                                     labels=labels, average=None, zero_division=0)
    recall = metr.recall_score(y_true=ground_truth, y_pred=prediction,
                               labels=labels, average=None, zero_division=0)
    f1 = metr.f1_score(y_true=ground_truth, y_pred=prediction,
                       labels=labels, average=None, zero_division=0)

    if verbosity:
        for l in labels:
            print(f"Predicted label {l} for {(prediction == l).sum()} samples, {(ground_truth == l).sum()} are in gt.")
        np.set_printoptions(precision=4)
        print(f"## Accuracy: {np.round(accuracy, decimals=4)}")
        print(f"## Precision: {precision}, avg: {np.round(np.mean(precision), decimals=4)}")
        print(f"## Recall: {recall}, avg: {np.round(np.mean(recall), decimals=4)}")
        print(f"## F1-score: {f1}, avg: {np.round(np.mean(f1), decimals=4)}")

    return accuracy, precision, recall, f1

