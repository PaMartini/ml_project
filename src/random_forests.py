
from typing import *
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from load_data import data_pipeline_redwine
from evaluation import evaluate_class_predictions


def train_dtl_classifier(train_data,
                         label_column: str = 'label',
                         config: Union[dict, None] = None,
                         test: bool = False,
                         test_data: Union[None, pd.DataFrame] = None,
                         verbosity: bool = False) -> DecisionTreeClassifier:

    x = train_data.drop(columns=[label_column]).values
    y = train_data.loc[:, label_column].values

    # Todo
    return


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
