
from load_data import *
from naive_bayes import *
from random_forests import *
from svm import *


def baseline_results(colour: str = 'red'):
    assert colour in {'red', 'white'}, "Colour must be 'red' or 'white'."
    if colour == 'red':
        traind, testd = data_pipeline_redwine(val_and_test=False)
        # Delete quality columns in data frames:
        traind = traind.drop(columns=['quality'])
        testd = testd.drop(columns=['quality'])
    elif colour == 'white':
        traind, testd = data_pipeline_whitewine(val_and_test=False)
        # Delete quality columns in data frames:
        traind = traind.drop(columns=['quality'])
        testd = testd.drop(columns=['quality'])

    print(f"### Results for the wine colour {colour} with default settings are ... ###")


if __name__ == '__main__':

    print('done')
