
import pickle
from typing import *
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from load_data import data_pipeline_redwine
from evaluation import evaluate_class_predictions
from auxiliary_functions import parameter_tuning_wrapper


def train_svm_model(train_data: pd.DataFrame,
                    label_column: str = 'label',
                    config: Union[dict, None] = None,
                    test_data: Union[None, pd.DataFrame] = None,
                    verbosity: bool = False) -> Union[Tuple[SVC, Any, Any, Any, Any], SVC]:
    """
    Function for training and testing an SVM model.
    :param train_data: Dataframe with train data.
    :param label_column: Name of the column with the labels.
    :param config: Dictionary with parameters of SVM model.
    :param test_data: Dataframe with test data. If None no testing is done.
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

    if test_data is not None:
        if verbosity:
            print("### Test results SVM ###")
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


def run_parameter_tuning_svm(train_data: pd.DataFrame,
                             label_column: str = 'label',
                             file_name: str = '../configurations/best_svm_config.pickle') -> dict:

    config = [
        {'C': [0.5, 1, 2, 5],
         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
         'degree': [3, 4, 6, 10],  # Degree of the polynomial kernel function.
         'gamma': ['scale', 'auto'],  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
         'coef0': [0, 0.1, 1],  # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
         'shrinking': [False],  # Whether to use the shrinking heuristic.
         'probability': [False],
         'tol': [0.001],  # Tolerance for stopping criterion.
         'max_iter': [-1],
         'class_weight': [None, 'balanced']}
    ]

    best_config = parameter_tuning_wrapper(classifier=SVC(),
                                           param_search_space=config,
                                           train_data=train_data,
                                           label_column=label_column,
                                           verbosity=True,
                                           save=True,
                                           filename=file_name)

    return best_config


if __name__ == '__main__':
    traind, testd = data_pipeline_redwine()
    # Delete quality columns in data frames:
    traind = traind.drop(columns=['quality'])
    testd = testd.drop(columns=['quality'])

    # run_parameter_tuning_svm(train_data=traind, label_column='label')

    # Train model with best found configuration
    with open('../results/best_svm_config.pickle', 'rb') as f:
        best_svm_param = pickle.load(f)

    train_svm_model(train_data=traind,
                    label_column='label',
                    config=best_svm_param,
                    test_data=testd,
                    verbosity=True)

    train_svm_model(train_data=traind,
                    label_column='label',
                    config=None,
                    test_data=testd,
                    verbosity=True)

