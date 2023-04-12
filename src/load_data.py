import pandas as pd
import numpy as np
from typing import *


def load_wine(filename: str, verbosity: bool = False):
    data = pd.read_csv(filename, sep=";")
    if verbosity:
        print(data.head())
        print(f"The features are {list(data.columns)}")
        print(f"The possible values of the quality are {np.unique(data.loc[:, 'quality'])}.")
    return data


def preprocess_wine():
    # Todo
    return


def load_comma_sep_csv(filename: str, verbosity: bool = False):
    data = pd.read_csv(filename, sep=",")
    if verbosity:
        print(data.head())
        print(f"The features are {list(data.columns)}")
    return data



if __name__=='__main__':
    fn_red = "../data/wine_data/winequality-red.csv"
    fn_white = "../data/wine_data/winequality-white.csv"
    data_red = load_wine(filename=fn_red, verbosity=True)
    data_white = load_wine(filename=fn_white, verbosity=True)

    fn_banknotes = "../data/banknotes_data/data_banknote_authentication.txt"
    data_banknotes = load_comma_sep_csv(filename=fn_banknotes, verbosity=True)

    fn_obesity = "../data/obesity_data/ObesityDataSet_raw_and_data_sinthetic.csv"
    data_obesity = load_comma_sep_csv(filename=fn_obesity, verbosity=True)


