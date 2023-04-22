
from typing import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from load_data import data_pipeline_redwine
from evaluation import evaluate_class_predictions


def train_dt_classifier(train_data,
                         label_column: str = 'label',
                         config: Union[dict, None] = None,
                         test: bool = False,
                         test_data: Union[None, pd.DataFrame] = None,
                         verbosity: bool = False) -> DecisionTreeClassifier:

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
        print(f"The feature importances according to the '{config['splitter']}' splitting rule are:")
        print(model.feature_importances_)

    if test:
        x_test = test_data.drop(columns=[label_column]).values
        y_test = test_data.loc[:, label_column].values

        pred = model.predict(X=x_test)

        evaluate_class_predictions(prediction=pred, ground_truth=y_test, verbosity=True)

    return model


def train_random_forest(train_data,
                        label_column: str = 'label',
                        config: Union[dict, None] = None,
                        test: bool = False,
                        test_data: Union[None, pd.DataFrame] = None,
                        verbosity: bool = False) -> RandomForestClassifier:

    x = train_data.drop(columns=[label_column]).values
    y = train_data.loc[:, label_column].values

    # Todo
    return


if __name__ == '__main__':
    traind, testd = data_pipeline_redwine()
    # Delete quality columns in data frames:
    traind = traind.drop(columns=['quality'])
    testd = testd.drop(columns=['quality'])

    train_dt_classifier(train_data=traind,
                        label_column='label',
                        config=None,
                        test=True,
                        test_data=testd,
                        verbosity=True)
