
import numpy as np
import sklearn.metrics as metr
from typing import *


def evaluate_class_predictions(prediction: np.ndarray, ground_truth: np.ndarray) -> Tuple[float, ...]:
    """
    Function for evaluating the performance of a model that returns classes as predictions (not class probabilities).
    Works for binary case and multi-class case.
    :param prediction: Class predictions of the model.
    :param ground_truth: Ground truth classes.
    :return: None
    """
    accuracy = metr.accuracy_score(y_true=ground_truth, y_pred=prediction)
    precision = metr.precision_score(y_true=ground_truth, y_pred=prediction)
    recall = metr.recall_score(y_true=ground_truth, y_pred=prediction)
    f1 = metr.f1_score(y_true=ground_truth, y_pred=prediction)
    print(f"## Accuracy: {accuracy}")
    print(f"## Precision: {precision}")
    print(f"## Recall: {recall}")
    print(f"## F1-score: {f1}")

    return accuracy, precision, recall, f1

