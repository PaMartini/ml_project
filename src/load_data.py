import pandas as pd
from typing import *


def load_wine(filename: str, verbosity: bool = False):
    data = pd.read_csv(filename, sep=";")
    if verbosity:
        print(data.head())
    return data


if __name__=='__main__':
    fn_red = "../data/wine_data/winequality-red.csv"
    fn_white = "../data/wine_data/winequality-white.csv"
    data_red = load_wine(filename=fn_red, verbosity=True)
    data_white = load_wine(filename=fn_white, verbosity=True)


