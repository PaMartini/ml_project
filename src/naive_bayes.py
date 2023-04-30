
from typing import *
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from evaluation import evaluate_class_predictions
from load_data import data_pipeline_redwine


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


if __name__ == '__main__':
    multiclass = False
    if not multiclass:
        traind, testd = data_pipeline_redwine()
        # Delete quality columns in data frames:
        traind = traind.drop(columns=['quality'])
        testd = testd.drop(columns=['quality'])
        train_gaussian_naive_bayes(train_data=traind,
                                   label_column='label',
                                   config=None,
                                   test=True,
                                   test_data=testd,
                                   verbosity=True)
    else:
        traind, testd = data_pipeline_redwine()
        # Delete label columns in data frames:
        traind = traind.drop(columns=['label'])
        testd = testd.drop(columns=['label'])
        train_gaussian_naive_bayes(train_data=traind,
                                   label_column='quality',
                                   config=None,
                                   test=True,
                                   test_data=testd,
                                   verbosity=True)

