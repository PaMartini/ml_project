import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import *


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

    return data






def load_comma_sep_csv(filename: str, verbosity: bool = False) -> pd.DataFrame:
    data = pd.read_csv(filename, sep=",")
    if verbosity:
        print(data.head())
        print(f"The features are {list(data.columns)}")
    return data



if __name__=='__main__':
    print(os.getcwd())
    fn_red = "../data/wine_data/winequality-red.csv"
    data_red = load_wine(filename=fn_red, verbosity=False)
    preprocess_wine(data_red, verbosity=True)




    # fn_white = "../data/wine_data/winequality-white.csv"
    # data_white = load_wine(filename=fn_white, verbosity=True)

    # fn_banknotes = "../data/banknotes_data/data_banknote_authentication.txt"
    # data_banknotes = load_comma_sep_csv(filename=fn_banknotes, verbosity=True)

    # fn_obesity = "../data/obesity_data/ObesityDataSet_raw_and_data_sinthetic.csv"
    # data_obesity = load_comma_sep_csv(filename=fn_obesity, verbosity=True)



