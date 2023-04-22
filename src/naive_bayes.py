
from typing import *
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

        pred = model.predict(X=x_test)

        evaluate_class_predictions(prediction=pred, ground_truth=y_test, verbosity=True)

    return model


if __name__ == '__main__':
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

