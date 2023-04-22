
import pickle
from typing import *
import pandas as pd
from sklearn.model_selection import GridSearchCV


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
    scores = 'accuracy'

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_search_space, scoring=scores, cv=5, verbose=2)
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