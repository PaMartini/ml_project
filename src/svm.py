import pickle
from typing import *

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from load_data import data_pipeline_redwine
from evaluation import evaluate_class_predictions


def train_svm_model(train_data: pd.DataFrame,
                    label_column: str = 'label',
                    config: Union[dict, None] = None,
                    test: bool = False,
                    test_data: Union[None, pd.DataFrame] = None,
                    verbosity: bool = False) -> SVC:
    """
    Function for training and testing an SVM model.
    :param train_data: Dataframe with train data.
    :param label_column: Name of the column with the labels.
    :param config: Dictionary with parameters of SVM model.
    :param test: Whether to evaluate the trained model on the test set or not.
    :param test_data: Dataframe with test data.
    :param verbosity: Whether to print information on the trained model or not.
    :return: Trained SVM model.
    """
    x = train_data.drop(columns=[label_column]).values
    y = train_data.loc[:, label_column].values
    if config is None:
        # Default parameters
        config = {'C': 1,
                  'kernel': 'linear',
                  'degree': 5,
                  'gamma': 'scale',
                  'coef0': 0,
                  'shrinking': False,
                  'probability': False,
                  'tol': 1e-3,
                  'max_iter': -1,
                  'class_weight': None}

    model = SVC(C=config['C'],
                kernel=config['kernel'],
                shrinking=config['shrinking'],
                probability=config['probability'],
                tol=config['tol'],
                max_iter=config['max_iter'],
                class_weight=config['class_weight'])

    model.fit(X=x, y=y, sample_weight=None)

    if verbosity:
        print("The parameters of the trained SVM model are:")
        print(model.get_params(deep=True))
        print(f"The intercept of the decision function is {model.intercept_}.")
        if model.kernel == 'linear':
            print("For the linear kernel the feature weights are:")
            print(model.coef_)
        print("The indices of the support vectors in the training set are:")
        print(model.support_)

    if test:
        x_test = test_data.drop(columns=[label_column]).values
        y_test = test_data.loc[:, label_column].values

        pred = model.predict(X=x_test)

        evaluate_class_predictions(prediction=pred, ground_truth=y_test, verbosity=True)

    return model


def parameter_tuning_svm(train_data: pd.DataFrame,
                         label_column: str = 'label',
                         verbosity: bool = False,
                         save: bool = False,
                         filename: str = '../configurations/best_svm_config.pickle') -> dict:

    x = traind.drop(columns=label_column).values
    y = train_data.loc[:, label_column].values

    # scores = ['precision', 'recall']
    scores = 'accuracy'
    config = [
        {'C': [0.5, 1, 2, 10],
         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
         'degree': [3, 4, 6, 10],  # Degree of the polynomial kernel function.
         'gamma': ['scale', 'auto'],  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
         'coef0': [0, 0.1, 1],  # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
         'shrinking': [True, False],  # Whether to use the shrinking heuristic.
         'probability': [False],
         'tol': [0.001, 0.0001],  # Tolerance for stopping criterion.
         'max_iter': [-1],
         'class_weight': [None, 'balanced']}
    ]

    grid_search = GridSearchCV(estimator=SVC(), param_grid=config, scoring=scores, cv=5, verbose=1)
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


if __name__ == '__main__':
    traind, testd = data_pipeline_redwine(val_and_test=False)
    # Delete quality columns in data frames:
    traind = traind.drop(columns=['quality'])
    testd = testd.drop(columns=['quality'])
    # Perform parameter tuning
    # best_config = parameter_tuning_svm(train_data=traind)

    # Train model with best found configuration
    with open('../configurations/best_svm_config.pickle', 'rb') as f:
        best_param = pickle.load(f)
    train_svm_model(train_data=traind,
                    label_column='label',
                    config=best_param,
                    test=True,
                    test_data=testd,
                    verbosity=True)

