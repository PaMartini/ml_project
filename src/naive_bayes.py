
from typing import *
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from pingouin import multivariate_normality

from evaluation import evaluate_class_predictions
from load_data import *
from auxiliary_functions import parameter_tuning_wrapper


def run_multivariate_henze_zirkler_test(data: pd.DataFrame,
                                        label_columns: list[str] = None,
                                        normalize: bool = False,
                                        verbosity: bool = False) -> Tuple[float, float, bool]:
    if label_columns is None:
        label_columns = ['label', 'quality']
    # data = data.drop(columns=label_columns).values
    if normalize:
        mean_ = np.mean(data, axis=0)
        std_ = np.std(data, axis=0)
        data = (data - mean_) / std_

    hz, pval, normal = multivariate_normality(X=data, alpha=0.05)
    if verbosity:
        print(f"## Henze-Zirkler test for normality yields {normal}. "
              f"\n## The p-value is {pval}. \n## The value of the test statistic is {hz}.")
    return hz, pval, normal


def train_gaussian_naive_bayes(train_data,
                               label_column: str = 'label',
                               config: Union[dict, None] = None,
                               test_data: Union[None, pd.DataFrame] = None,
                               verbosity: bool = False) \
        -> Union[Tuple[GaussianNB, Any, Any, Any, Any], GaussianNB]:
    """
    Function for training and testing a gaussian naive bayes model.
    :param train_data: Dataframe with train data.
    :param label_column: Name of the column with the labels.
    :param config: Dictionary with parameters of naive bayes model.
    :param test_data: Dataframe with test data. If None no testing is done.
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
    if test_data is not None:
        if verbosity:
            print("### Test results naive bayes ###")
        x_test = test_data.drop(columns=[label_column]).values
        y_test = test_data.loc[:, label_column].values

        labels = np.union1d(y, y_test)

        pred = model.predict(X=x_test)

        accuracy, precision, recall, f1 = evaluate_class_predictions(prediction=pred,
                                                                     ground_truth=y_test,
                                                                     labels=labels,
                                                                     verbosity=verbosity)
        return model, accuracy, precision, recall, f1

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
    fn_white = "../data/wine_data/winequality-red.csv"
    data_white = load_wine(filename=fn_white, verbosity=False)
    run_multivariate_henze_zirkler_test(data=data_white, normalize=True, label_columns=['quality'], verbosity=True)
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

