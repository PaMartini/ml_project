
from typing import *
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from evaluation import evaluate_class_predictions
from load_data import data_pipeline_redwine
from auxiliary_functions import parameter_tuning_wrapper


def train_gaussian_naive_bayes(train_data,
                               label_column: str = 'label',
                               config: Union[dict, None] = None,
                               test: bool = False,
                               test_data: Union[None, pd.DataFrame] = None,
                               verbosity: bool = False) -> GaussianNB:
    """
    Function for training and testing a gaussian naive bayes model.
    :param train_data: Dataframe with train data.
    :param label_column: Name of the column with the labels.
    :param config: Dictionary with parameters of naive bayes model.
    :param test: Whether to evaluate the trained model on the test set or not.
    :param test_data: Dataframe with test data.
    :param verbosity: Whether to print information on the trained model or not.
    :return: Trained naive bayes model.
    """

    x = train_data.drop(columns=[label_column]).values
    y = train_data.loc[:, label_column].values

    if config is None:
        config = {'priors': None,
                  'var_smoothing': 1e-9}

    model = GaussianNB(priors=config['priors'],
                       var_smoothing=config['var_smoothing'])
    model.fit(X=x, y=y)

    if verbosity:
        print(f"The class counts are {model.class_count_}, the resulting priors are {model.class_prior_}.")
    if test:
        x_test = test_data.drop(columns=[label_column]).values
        y_test = test_data.loc[:, label_column].values

        labels = np.union1d(y, y_test)

        pred = model.predict(X=x_test)

        evaluate_class_predictions(prediction=pred, ground_truth=y_test, labels=labels, verbosity=True)

    return model


def run_parameter_tuning_nb(train_data: pd.DataFrame,
                            label_column: str = 'label') -> dict:

    config = [
        {'priors': [None, np.array([1/3, 1/3, 1/3])],  # Try uniform priors
         'var_smoothing': [1e-9, 0]}
    ]

    best_config = parameter_tuning_wrapper(classifier=GaussianNB(),
                                           param_search_space=config,
                                           train_data=train_data,
                                           label_column=label_column,
                                           verbosity=True,
                                           save=True,
                                           filename='../configurations/best_nb_config.pickle')

    return best_config


if __name__ == '__main__':
    traind, testd = data_pipeline_redwine()
    # Delete quality columns in data frames:
    traind = traind.drop(columns=['quality'])
    testd = testd.drop(columns=['quality'])

    run_parameter_tuning_nb(train_data=traind, label_column='label')

    with open('../configurations/best_nb_config.pickle', 'rb') as f:
        best_nb_param = pickle.load(f)

    print(best_nb_param)

    train_gaussian_naive_bayes(train_data=traind,
                               label_column='label',
                               config=best_nb_param,
                               test=True,
                               test_data=testd,
                               verbosity=True)

