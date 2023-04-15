
from typing import *

import pandas as pd
from sklearn.svm import SVC
from load_data import data_pipeline_redwine
from evaluation import evaluate_class_predictions


def train_svm_model(train_data,
                    label_column: str = 'label',
                    test: bool = False,
                    test_data: Union[None, pd.DataFrame] = None,
                    verbosity: bool = False) -> SVC:
    """
    Function for training and testing an SVM model.
    :param train_data: Dataframe with train data.
    :param label_column: Name of the column with the labels.
    :param test: Whether to evaluate the trained model on the test set or not.
    :param test_data: Dataframe with test data.
    :param verbosity: Whether to print information on the trained model or not.
    :return: Trained SVM model.
    """
    x = train_data.drop(columns=[label_column]).values
    y = train_data.loc[:, label_column].values
    model = SVC(C=1, kernel='linear', shrinking=False, probability=False, tol=1e-3, max_iter=-1, class_weight=None)
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

        evaluate_class_predictions(prediction=pred, ground_truth=y_test)

    return model


if __name__ == '__main__':
    traind, testd = data_pipeline_redwine()
    # Delete quality columns in data frames:
    traind = traind.drop(columns=['quality'])
    testd = testd.drop(columns=['quality'])
    # Train model
    train_svm_model(train_data=traind, label_column='label', test=True, test_data=testd, verbosity=True)

