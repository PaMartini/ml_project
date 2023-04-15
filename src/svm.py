
from typing import *

import pandas as pd
from sklearn.svm import SVC
from load_data import data_pipeline_redwine


def train_svm_model(train_data,
                    label_column: str = 'label',
                    test: bool = False,
                    test_data: Union[None, pd.DataFrame] = None) -> SVC:
    """
    Function for training and testing an SVM model.
    :param train_data: Dataframe with train data.
    :param label_column: Name of the column with the labels.
    :param test: Whether to validate the trained model or not.
    :param test_data: Dataframe with test data.
    :return: Trained SVM model.
    """
    x = train_data.drop(columns=[label_column]).values
    y = train_data.loc[:, label_column].values
    model = SVC(C=1, kernel='linear', shrinking=False, probability=False, tol=1e-3, max_iter=-1, class_weight=None)
    model.fit(X=x, y=y, sample_weight=None)

    if test:
        x_test = test_data.drop(columns=[label_column]).values
        y_test = test_data.loc[:, label_column].values

        # Todo

    return model


if __name__ == '__main__':
    traind, testd = data_pipeline_redwine()
    # Delete quality columns in data frames:
    traind = traind.drop(columns=['quality'])
    testd = testd.drop(columns=['quality'])
    # Train model
    train_svm_model(train_data=traind, label_column='label', test=True, test_data=testd)

