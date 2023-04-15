
import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection
import matplotlib.pyplot as plt
from typing import *


def load_comma_sep_csv(filename: str, verbosity: bool = False) -> pd.DataFrame:
    data = pd.read_csv(filename, sep=",")
    if verbosity:
        print(data.head())
        print(f"The features are {list(data.columns)}")
    return data


def load_wine(filename: str, verbosity: bool = False) -> pd.DataFrame:
    data = pd.read_csv(filename, sep=";")
    if verbosity:
        print(data.head())
        print(f"The features are {list(data.columns)}")
        print(f"The possible values of the quality are {np.unique(data.loc[:, 'quality'])}.")
        plt.hist(data.loc[:, 'quality'], bins=6)
        plt.show()

    return data


def preprocess_wine(data: pd.DataFrame, verbosity: bool = False) -> pd.DataFrame:
    # Normalize data
    means = np.mean(data.iloc[:, 0:(data.shape[1] - 1)].values, axis=0)
    stdev = np.std(data.iloc[:, 0:(data.shape[1] - 1)].values, axis=0)
    data.iloc[:, 0:(data.shape[1] - 1)] -= means
    data.iloc[:, 0:(data.shape[1] - 1)] /= stdev

    # Annotate with binary labels: 0 = bad (rating <= 5), 1 = good (rating > 5)
    quality_ratings_bool = (np.array(data.loc[:, 'quality']) > 5).astype(float)
    data["label"] = quality_ratings_bool

    if verbosity:
        print(data.head())
        print(f"There are {data.loc[:, 'label'].sum()} out of "
              f"{data.shape[0]} samples with good quality.")
    return data


def perform_train_val_test_split(data: pd.DataFrame,
                                 split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                                 shuffle: bool = False,
                                 preserve_class_dist: bool = False,
                                 label_column: str = 'label') -> Union[pd.DataFrame, tuple[pd.DataFrame, ...]]:
    """
    Method for splitting the data set into a training, validation and test set.
    :param data: Pandas data frame containing the data.
    :param split: Percentages of training, validation, test set.
    :param shuffle: Whether to shuffle the data or not.
    :param preserve_class_dist: Whether to preserve the class distribution during sampling.
    :param label_column: Name of the column with the labels.
    :return: Data frames according to the chosen split.
    """
    assert sum(split) == 1 and np.all(np.array(split) >= 0), \
        "Train, validation and test set split must be percentages that sum up to 1."
    assert split[0] > 0, "Training set can not be empty."
    assert shuffle if preserve_class_dist else shuffle or not shuffle, \
        "Shuffle must be True if preserve_class_dist is True."
    labels = data.loc[:, label_column].values
    if split[1] == 0 and split[2] == 0:
        return data
    elif split[1] == 0:
        if preserve_class_dist:
            train_data, test_data = sk.model_selection.train_test_split(data,
                                                                        test_size=split[2],
                                                                        train_size=split[0],
                                                                        shuffle=shuffle,
                                                                        stratify=labels)
        else:
            train_data, test_data = sk.model_selection.train_test_split(data,
                                                                        test_size=split[2],
                                                                        train_size=split[0],
                                                                        shuffle=shuffle)
        return train_data, test_data
    elif split[2] == 0:
        if preserve_class_dist:
            train_data, val_data = sk.model_selection.train_test_split(data,
                                                                       test_size=split[1],
                                                                       train_size=split[0],
                                                                       shuffle=shuffle,
                                                                       stratify=labels)
        else:
            train_data, val_data = sk.model_selection.train_test_split(data,
                                                                       test_size=split[1],
                                                                       train_size=split[0],
                                                                       shuffle=shuffle)
        return train_data, val_data
    else:
        if preserve_class_dist:
            train_data, dummy = sk.model_selection.train_test_split(data,
                                                                    test_size=split[1] + split[2],
                                                                    train_size=split[0],
                                                                    shuffle=shuffle,
                                                                    stratify=labels)
            dummy_labels = dummy.loc[:, label_column].values
            val_data, test_data = sk.model_selection.train_test_split(dummy,
                                                                      test_size=split[2],
                                                                      train_size=split[1],
                                                                      shuffle=shuffle,
                                                                      stratify=dummy_labels)
            return train_data, val_data, test_data
        else:
            train_data, dummy = sk.model_selection.train_test_split(data,
                                                                    test_size=split[1] + split[2],
                                                                    train_size=split[0],
                                                                    shuffle=shuffle)
            val_data, test_data = sk.model_selection.train_test_split(dummy,
                                                                      test_size=split[2],
                                                                      train_size=split[1],
                                                                      shuffle=shuffle)
            return train_data, val_data, test_data


def data_pipeline_redwine() -> Tuple[pd.DataFrame, ...]:
    """
    Example workflow for loading and preprocessing of redwine data set.
    :return: Redwine data set, preprocessed and split into training and test set.
    """
    fn_red = "../data/wine_data/winequality-red.csv"
    data_red = load_wine(filename=fn_red, verbosity=False)
    data_red = preprocess_wine(data_red, verbosity=False)
    # traind, vald, testd = perform_train_val_test_split(data=data_red,
    #                                                   split=(0.6, 0.2, 0.2),
    #                                                   shuffle=True,
    #                                                   preserve_class_dist=True)
    traind, testd = perform_train_val_test_split(data=data_red,
                                                 split=(0.8, 0, 0.2),
                                                 shuffle=True,
                                                 preserve_class_dist=True)
    return traind, testd


if __name__ == '__main__':
    print(data_pipeline_redwine())




    # fn_white = "../data/wine_data/winequality-white.csv"
    # data_white = load_wine(filename=fn_white, verbosity=True)

    # fn_banknotes = "../data/banknotes_data/data_banknote_authentication.txt"
    # data_banknotes = load_comma_sep_csv(filename=fn_banknotes, verbosity=True)

    # fn_obesity = "../data/obesity_data/ObesityDataSet_raw_and_data_sinthetic.csv"
    # data_obesity = load_comma_sep_csv(filename=fn_obesity, verbosity=True)



