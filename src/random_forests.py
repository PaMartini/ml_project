
from typing import *
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from load_data import data_pipeline_redwine
from evaluation import evaluate_class_predictions
from auxiliary_functions import parameter_tuning_wrapper


def train_dt_classifier(train_data,
                        label_column: str = 'label',
                        config: Union[dict, None] = None,
                        test_data: Union[None, pd.DataFrame] = None,
                        verbosity: bool = False) \
        -> Union[Tuple[DecisionTreeClassifier, Any, Any, Any, Any], DecisionTreeClassifier]:
    """
    Function for training and testing a decision tree classifier.
    :param train_data: Dataframe with train data.
    :param label_column: Name of the column with the labels.
    :param config: Dictionary with parameters of the decision tree classifier.
    :param test_data: Dataframe with test data. If None no testing is done.
    :param verbosity: Whether to print information on the trained classifier or not.
    :return: Trained decision tree classifier.
    """

    x = train_data.drop(columns=[label_column]).values
    y = train_data.loc[:, label_column].values

    if config is None:
        config = {'criterion': 'gini',
                  'splitter': 'best',
                  'max_depth': None,
                  'min_samples_split': 2,
                  'min_samples_leaf': 1,
                  'min_weight_fraction_leaf': 0,
                  'max_features': None,
                  'random_state': None,
                  'max_leaf_nodes': None,
                  'min_impurity_decrease': 0,
                  'class_weight': None,
                  'ccp_alpha': 0}

    model = DecisionTreeClassifier(criterion=config['criterion'],
                                   splitter=config['splitter'],
                                   max_depth=config['max_depth'],
                                   min_samples_split=config['min_samples_split'],
                                   min_samples_leaf=config['min_samples_leaf'],
                                   min_weight_fraction_leaf=config['min_weight_fraction_leaf'],
                                   max_features=config['max_features'],
                                   random_state=config['random_state'],
                                   max_leaf_nodes=config['max_leaf_nodes'],
                                   min_impurity_decrease=config['min_impurity_decrease'],
                                   class_weight=config['class_weight'],
                                   ccp_alpha=config['ccp_alpha'])
    model.fit(X=x, y=y)

    if verbosity:
        tree.plot_tree(model)
        plt.show()
        report = tree.export_text(model)
        #print(report)
        print(f"The feature importances according to the '{config['criterion']}' splitting criterion are:")
        print(model.feature_importances_)

    if test_data is not None:
        if verbosity:
            print("### Test results decision tree ###")
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


def train_random_forest(train_data,
                        label_column: str = 'label',
                        config: Union[dict, None] = None,
                        test_data: Union[None, pd.DataFrame] = None,
                        verbosity: bool = False) \
        -> Union[Tuple[RandomForestClassifier, Any, Any, Any, Any], RandomForestClassifier]:
    """
    Function for training and testing a random forest classifier.
    :param train_data: Dataframe with train data.
    :param label_column: Name of the column with the labels.
    :param config: Dictionary with parameters of the random forest classifier.
    :param test_data: Dataframe with test data. If None no testing is done.
    :param verbosity: Whether to print information on the trained classifier or not.
    :return: Trained random forest classifier.
    """

    x = train_data.drop(columns=[label_column]).values
    y = train_data.loc[:, label_column].values

    if config is None:
        config = {'criterion': 'gini',
                  'max_depth': None,
                  'min_samples_split': 2,
                  'min_samples_leaf': 1,
                  'min_weight_fraction_leaf': 0,
                  'max_features': None,
                  'random_state': None,
                  'max_leaf_nodes': None,
                  'min_impurity_decrease': 0,
                  'class_weight': None,
                  'ccp_alpha': 0,
                  # Parameters specific to random forest classifier
                  'n_estimators': 100,
                  'bootstrap': True,
                  'oob_score': True,
                  'n_jobs': None,
                  'warm_start': False,
                  'max_samples': None}

    model = RandomForestClassifier(criterion=config['criterion'],
                                   max_depth=config['max_depth'],
                                   min_samples_split=config['min_samples_split'],
                                   min_samples_leaf=config['min_samples_leaf'],
                                   min_weight_fraction_leaf=config['min_weight_fraction_leaf'],
                                   max_features=config['max_features'],
                                   random_state=config['random_state'],
                                   max_leaf_nodes=config['max_leaf_nodes'],
                                   min_impurity_decrease=config['min_impurity_decrease'],
                                   class_weight=config['class_weight'],
                                   ccp_alpha=config['ccp_alpha'],
                                   # Parameters specific to random forest classifier
                                   n_estimators=config['n_estimators'],
                                   bootstrap=config['bootstrap'],
                                   oob_score=config['oob_score'],
                                   n_jobs=config['n_jobs'],
                                   warm_start=config['warm_start'],
                                   max_samples=config['max_samples'])

    model.fit(X=x, y=y)

    if verbosity:
        print(f"The feature importances according to the '{config['criterion']}' splitting criterion are:")
        print(model.feature_importances_)
        if model.oob_score:
            print(f"The out of bag error of the random forest classifier is {model.oob_score_}.")

    if test_data is not None:
        if verbosity:
            print("### Test results random forest ###")
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


def run_parameter_tuning_dt(train_data: pd.DataFrame,
                            label_column: str = 'label') -> dict:

    config = [
        {'criterion': ['gini', 'entropy', 'log_loss'],
         'splitter': ['best', 'random'],
         'max_depth': [None, 7, 11],
         'min_samples_split': [2, 4, 8],
         'min_samples_leaf': [1, 5, 10],
         'min_weight_fraction_leaf': [0],
         'max_features': [None],
         'random_state': [None],
         'max_leaf_nodes': [None, 15, 20, 30],
         'min_impurity_decrease': [0],
         'class_weight': [None, 'balanced'],
         'ccp_alpha': [0]}
    ]

    '''config = [
        {'criterion': ['gini', 'entropy'],
         'splitter': ['best'],
         'max_depth': [None],
         'min_samples_split': [2],
         'min_samples_leaf': [1],
         'min_weight_fraction_leaf': [0],
         'max_features': [None],
         'random_state': [None],
         'max_leaf_nodes': [None],
         'min_impurity_decrease': [0],
         'class_weight': [None, 'balanced'],
         'ccp_alpha': [0]}
    ]'''

    best_config = parameter_tuning_wrapper(classifier=DecisionTreeClassifier(),
                                           param_search_space=config,
                                           train_data=train_data,
                                           label_column=label_column,
                                           verbosity=True,
                                           save=True,
                                           filename='../configurations/best_dt_config.pickle')

    return best_config


def run_parameter_tuning_rf(train_data: pd.DataFrame,
                            label_column: str = 'label') -> dict:

    config = [
        {'criterion': ['gini', 'entropy', 'log_loss'],
         'max_depth': [None, 7, 11],
         'min_samples_split': [2, 4, 8],
         'min_samples_leaf': [1, 5, 10],
         'min_weight_fraction_leaf': [0],
         'max_features': [None],
         'random_state': [None],
         'max_leaf_nodes': [None, 15, 20, 30],
         'min_impurity_decrease': [0],
         'class_weight': [None, 'balanced'],
         'ccp_alpha': [0],
         # Parameters specific to random forest classifier
         'n_estimators': [50, 100, 150],
         'bootstrap': [True],
         'oob_score': [True],
         'n_jobs': [None],
         'warm_start': [False],
         'max_samples': [None]
         }
    ]

    best_config = parameter_tuning_wrapper(classifier=RandomForestClassifier(),
                                           param_search_space=config,
                                           train_data=train_data,
                                           label_column=label_column,
                                           verbosity=True,
                                           save=True,
                                           filename='../configurations/best_rf_config.pickle')

    return best_config


if __name__ == '__main__':
    traind, testd = data_pipeline_redwine()
    # Delete quality columns in data frames:
    traind = traind.drop(columns=['quality'])
    testd = testd.drop(columns=['quality'])

    run_parameter_tuning_dt(train_data=traind, label_column='label')
    run_parameter_tuning_rf(train_data=traind, label_column='label')

    # Train model with best found configuration
    with open('../configurations/best_dt_config.pickle', 'rb') as f:
        best_dt_param = pickle.load(f)

    with open('../configurations/best_rf_config.pickle', 'rb') as f:
        best_rf_param = pickle.load(f)

    print(best_dt_param)
    print(best_rf_param)

    train_dt_classifier(train_data=traind,
                        label_column='label',
                        config=best_dt_param,
                        test=True,
                        test_data=testd,
                        verbosity=False)

    train_random_forest(train_data=traind,
                        label_column='label',
                        config=best_rf_param,
                        test=True,
                        test_data=testd,
                        verbosity=True)
