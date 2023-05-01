
import pickle
from typing import *

import numpy as np
import pandas as pd
import sklearn.metrics as metr
from sklearn.model_selection import GridSearchCV


def custom_metric(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  labels: np.ndarray = None,
                  beta: float = 1.0,
                  class_weights: np.ndarray = None):
    if labels is None:
        labels = np.unique(y_true)
    if class_weights is None:
        class_weights = np.ones(labels.shape[0])

    f_beta_class_scores = metr.fbeta_score(y_true=y_true, y_pred=y_pred,
                                           labels=labels, beta=beta, average=None, zero_division=0)

    score = (f_beta_class_scores * class_weights)
    # print("#########", f_beta_class_scores, f_beta_class_scores.sum()/3)
    # print("#########", score, score.sum())
    score = score.sum()

    return score


def parameter_tuning_wrapper(classifier: Any,
                             param_search_space: list[dict],
                             train_data: pd.DataFrame,
                             label_column: str = 'label',
                             verbosity: bool = False,
                             save: bool = False,
                             filename: str = '../configurations/best_param_config.pickle') -> dict:

    x = train_data.drop(columns=label_column).values
    y = train_data.loc[:, label_column].values

    # scores = ['precision', 'recall']
    # scores = 'accuracy'
    score_fct = metr.make_scorer(score_func=custom_metric,
                                 greater_is_better=True,
                                 labels=np.array([0, 1, 2]),
                                 beta=1.0,
                                 class_weights=np.array([0.45, 0.1, 0.45]))

    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=param_search_space,
                               scoring=score_fct,
                               refit='precision',
                               cv=5,
                               verbose=2)
    # works with k-fold cross validation on the training set -> default is cv=5
    grid_search.fit(x, y)

    if verbosity:
        print("The best found parameter configuration is:")
        print(grid_search.best_params_)

    best_param = grid_search.best_params_

    if save:
        with open(filename, 'wb') as f:
            pickle.dump(best_param, f, protocol=pickle.HIGHEST_PROTOCOL)

    return best_param

