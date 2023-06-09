
import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from typing import *

from evaluation import evaluate_class_predictions


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
        # data.describe(include='all').to_csv("my_description.csv")
        # print(data.describe(include='all'))
        total_samples = data.shape[0]
        print(f"The dataset contains {total_samples} samples.")
        print(f"The features are {list(data.columns)}")
        quality_ratings = np.unique(data.loc[:, 'quality'].values)
        print(f"The possible values of the quality are {quality_ratings}.")
        num_samples = []
        for i in quality_ratings:
            num_ = (data.loc[:, 'quality'].values == i).sum()
            num_samples.append(num_)
            print(f"There are {num_} ({(num_ / total_samples).round(decimals=4)}%) samples with quality rating {i}.")

        num_samples = np.array(num_samples)
        plt.bar(x=quality_ratings, height=num_samples, width=0.8)

        for i in range(num_samples.shape[0]):
            plt.text(quality_ratings[i], num_samples[i] + 6, num_samples[i], ha='center')

        plt.show()

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


def perform_pca_reduction(data: pd.DataFrame, pca_dim: int, label_columns: list[str, ...]) -> tuple[pd.DataFrame, PCA]:
    """
    Method for performing a PCA dimensionality reduction on the given data frame.
    :param data: Pandas data frame containing the data.
    :param pca_dim: Number of principal components to keep. If -1, no reduction will be computed.
    :param label_columns: Names of columns with label information. They are excluded from the PCA calculation.
    :return: Data frame with dimensionality reduced data and unchanged label columns.
    """
    if pca_dim == -1:
        pca_dim = data.shape[1] - len(label_columns)
    data_w_out_labels = data.drop(columns=label_columns).values
    n_features = data_w_out_labels.shape[1]
    assert 0 < pca_dim <= n_features, \
        "Number of principal components must be strictly greater than 0 and smaller or equal to the number of features."
    pca = PCA(n_components=pca_dim)
    pca.fit(data_w_out_labels)

    pca_data = apply_fct_to_data(data=data, label_columns=label_columns, function=pca.transform)

    return pca_data, pca


def standardize_data(data: pd.DataFrame, label_columns: list[str, ...]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Function for standardizing a dataframe using (X - mean) / stdev.
    :param data: Dataframe.
    :param label_columns: Names of columns with label information. They are excluded from the normalization.
    :return: Normalized dataframe, array with means, array with stdevs.
    """

    df_w_out_labels = data.drop(columns=label_columns)
    df_label_columns = data.loc[:, label_columns]

    means = np.mean(df_w_out_labels.values, axis=0)
    stdev = np.std(df_w_out_labels.values, axis=0)
    df_w_out_labels = calc_standardization(data=df_w_out_labels, means=means, stdev=stdev)

    standardized_df = pd.concat([df_w_out_labels, df_label_columns], axis=1)

    return standardized_df, means, stdev


def calc_standardization(data: Union[np.ndarray, pd.DataFrame],
                         means: np.ndarray,
                         stdev: np.ndarray) -> Union[np.ndarray, pd.DataFrame]:
    """
    Function for the standardization calculations. Works inplace.
    :param data: Dataframe or array.
    :param means: Array with means.
    :param stdev: Array with standard deviations.
    :return: Normalized Dataframe or array.
    """
    data -= means
    data /= stdev
    return data


def normalize_data(data: pd.DataFrame, label_columns: list[str, ...]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Function for normalizing a dataframe using (X - min(X)) / (max(X) - min(X)).
    :param data: Dataframe.
    :param label_columns: Names of columns with label information. They are excluded from the normalization.
    :return: Normalized dataframe, array with mins, array with maxs.
    """
    df_w_out_labels = data.drop(columns=label_columns)
    df_label_columns = data.loc[:, label_columns]

    mins = np.min(df_w_out_labels.values, axis=0)
    maxs = np.max(df_w_out_labels.values, axis=0)
    df_w_out_labels = calc_normalization(data=df_w_out_labels, mins=mins, maxs=maxs)

    normalized_df = pd.concat([df_w_out_labels, df_label_columns], axis=1)

    return normalized_df, mins, maxs


def calc_normalization(data: Union[np.ndarray, pd.DataFrame],
                       mins: np.ndarray,
                       maxs: np.ndarray) -> Union[np.ndarray, pd.DataFrame]:
    """
    Function for the standardization calculations. Works inplace.
    :param data: Dataframe or array.
    :param mins: Array with minima.
    :param maxs: Array with maxima.
    :return: Normalized Dataframe or array.
    """
    data -= mins
    data /= (maxs - mins)
    return data


def apply_fct_to_data(data: pd.DataFrame,
                      label_columns: list[str, ...],
                      function: Callable[[np.ndarray], np.ndarray]) -> pd.DataFrame:
    """
    Function that applies a function to the part of the dataframe that is not specified as a label column.
    :param data: Dataframe.
    :param label_columns: Names of columns with label information.
    :param function: Function that is applied to the dataframe.
    :return: Processed Dataframe.
    """

    data_w_out_labels = data.drop(columns=label_columns).values
    label_columns = data.loc[:, label_columns]

    processed_data = pd.DataFrame(function(data_w_out_labels))
    processed_data.set_index(keys=np.array(data.index), inplace=True)

    final_df = pd.concat([processed_data, label_columns], axis=1)

    return final_df


def calc_oversampling_strategy(y: np.ndarray) -> dict:
    """
    Function for computing a sampling strategy dictionary.
    Try to equalize number of samples per class by oversampling minority classes,
    but do not go over 2 times the original number of class samples.
    :param y: Label vector.
    :return: Dictionary with classes as keys and desired number of samples for each class as values.
    """
    unique_labels = np.unique(y)
    num_class_dict = {}
    for l in unique_labels:
        num_class_dict[l] = (y == l).sum()
    majority_class = max(num_class_dict, key=num_class_dict.get)
    strategy_dict = {}
    for l in unique_labels:
        if num_class_dict[l] * 2 <= num_class_dict[majority_class]:
            strategy_dict[l] = num_class_dict[l] * 2
        else:
            strategy_dict[l] = num_class_dict[majority_class]
    return strategy_dict


def calc_undersampling_strategy(y: np.ndarray) -> dict:
    """
    Function for computing a sampling strategy dictionary.
    Try to equalize number of samples per class by undersampling the majority class,
     but do not go under 1/2 times the original number of class samples.
    :param y: Label vector.
    :return: Dictionary with classes as keys and desired number of samples for each class as values.
    """
    unique_labels = np.unique(y)
    num_class_dict = {}
    for l in unique_labels:
        num_class_dict[l] = (y == l).sum()
    majority_class = max(num_class_dict, key=num_class_dict.get)
    smallest_minority_class = min(num_class_dict, key=num_class_dict.get)
    strategy_dict = {}

    for l in unique_labels:
        if l != majority_class:
            strategy_dict[l] = num_class_dict[l]
    # Set number of samples for majority class
    if num_class_dict[majority_class] / 2 >= num_class_dict[smallest_minority_class]:
        strategy_dict[majority_class] = int(np.ceil(num_class_dict[majority_class] / 2))
    else:
        strategy_dict[majority_class] = num_class_dict[smallest_minority_class]

    print(f"################# dict: {strategy_dict}")

    return strategy_dict


def preprocess_wine(data: pd.DataFrame,
                    shuffle: bool = False,
                    preserve_class_dist: bool = False,
                    val_and_test: bool = False,
                    over_sample: str = None,
                    under_sample: Union[None, str] = None,
                    scaling: str = 'standardize',
                    pca_dim: int = -1,
                    labelling: str = 'bmg',
                    verbosity: bool = False) -> Tuple[pd.DataFrame, ...]:
    """
    Function for performing the train, val, test split and preprocessing the wine data set.
    :param data: Dataframe with wine data.
    :param shuffle: Whether the data is shuffled before the train test (validation) split.
    :param preserve_class_dist: Whether to approximately preserve the class distribution in the data
    when performing the split, or not.
    :param val_and_test: Whether to split the data into a train, val and test or a train and test set.
    :param over_sample: Available if 'labelling == 'bmg', whether to resample the minority classes or not.
    :param under_sample: Available if 'labelling == 'bmg', whether to undersample the majority class or not.
    :param scaling: Which scaling method to apply to the data.
    :param pca_dim: Principal components of PCA dimensionality reduction. Default -1 corresponds to no reduction.
    :param labelling: Sets how to label the data.
    :param verbosity: Whether to print information on the process, or not.
    :return: The preprocessed and split data sets.
    """
    assert labelling in {'bmg', 'binary', 'quality'}, \
        "Parameter 'labelling must be one of the following 'bmg', 'binary, 'quality'."
    assert over_sample in {'random', 'smote', None}, "Parameter 'over_sample' must be 'random', 'smote' or None."
    assert scaling in {'standardize', 'min_max_norm', None}, \
        "Parameter 'scaling' must be 'standardize', 'min_max_norm' or None."

    if labelling == 'binary':
        # Annotate with binary labels: 0 = bad (rating <= 5), 1 = good (rating > 5)
        quality_ratings_bool = (data.loc[:, 'quality'].values > 5).astype(float)
        data["label"] = quality_ratings_bool
    elif labelling == 'quality':
        data["label"] = data.loc[:, 'quality'].values
    elif labelling == 'bmg':
        quality = data.loc[:, 'quality'].values
        bad_labels = (quality <= 4).astype(float) * 0
        medium_labels = np.logical_and((5 <= quality), (quality <= 6)).astype(float) * 1
        good_labels = (7 <= quality).astype(float) * 2
        bad_medium_good = bad_labels + medium_labels + good_labels
        data['label'] = bad_medium_good

    if verbosity:
        print(data.head())
        if labelling == 'binary':
            print(f"There are {data.loc[:, 'label'].sum()} out of "
                  f"{data.shape[0]} samples with good quality.")
        if labelling == 'quality':
            print(f"There are {data.loc[:, 'label'].values.shape[0]} samples in total.")
            for i in np.unique(data.loc[:, 'label'].values):
                print(f"There are {(data.loc[:, 'label'].values == i).sum()} samples with label {i}.")
        if labelling == 'bmg':
            print("Label 0=bad, 1= medium, 2=good quality.")
            print("In the whole dataset ...")
            for i in np.unique(data.loc[:, 'label'].values):
                print(f"there are {(data.loc[:, 'label'].values == i).sum()} samples with label {i}.")

    if val_and_test:
        traind, vald, testd = perform_train_val_test_split(data=data,
                                                           split=(0.6, 0.2, 0.2),
                                                           shuffle=shuffle,
                                                           preserve_class_dist=preserve_class_dist)

        if scaling is not None:
            if scaling == 'standardize':
                traind, means, stdev = standardize_data(data=traind, label_columns=['quality', 'label'])
                # Apply normalization to val and test data using means, stdev from traindata
                val_w_out_labels = vald.drop(columns=['quality', 'label'])
                val_label_columns = vald.loc[:, ['quality', 'label']]
                testd_w_out_labels = testd.drop(columns=['quality', 'label'])
                testd_label_columns = testd.loc[:, ['quality', 'label']]
                val_w_out_labels -= means
                val_w_out_labels /= stdev
                testd_w_out_labels -= means
                testd_w_out_labels /= stdev
                testd = pd.concat([testd_w_out_labels, testd_label_columns], axis=1)
                vald = pd.concat([val_w_out_labels, val_label_columns], axis=1)
            if scaling == 'min_max_norm':
                traind, mins, maxs = normalize_data(data=traind, label_columns=['quality', 'label'])
                # Apply min-max-normalization to val and test data using mins, maxs from traindata
                val_w_out_labels = vald.drop(columns=['quality', 'label'])
                val_label_columns = vald.loc[:, ['quality', 'label']]
                testd_w_out_labels = testd.drop(columns=['quality', 'label'])
                testd_label_columns = testd.loc[:, ['quality', 'label']]
                val_w_out_labels -= mins
                val_w_out_labels /= (maxs - mins)
                testd_w_out_labels -= mins
                testd_w_out_labels /= (maxs - mins)
                testd = pd.concat([testd_w_out_labels, testd_label_columns], axis=1)
                vald = pd.concat([val_w_out_labels, val_label_columns], axis=1)


        if pca_dim != -1:
            traind, train_pca = perform_pca_reduction(data=traind, pca_dim=pca_dim, label_columns=['quality', 'label'])
            # Apply PCA transformation to val and test data
            vald = apply_fct_to_data(data=vald, label_columns=['quality', 'label'], function=train_pca.transform)
            testd = apply_fct_to_data(data=testd, label_columns=['quality', 'label'], function=train_pca.transform)

            if verbosity:
                print(f"Performed PCA dimensionality reduction to {pca_dim} principal components.")
                print("The singular values are:")
                print(train_pca.singular_values_)
                print("The percentage of variance explained by each of the selected components is:")
                print(train_pca.explained_variance_ratio_)

        return traind, vald, testd

    else:
        traind, testd = perform_train_val_test_split(data=data,
                                                     split=(0.67, 0, 0.33),
                                                     shuffle=shuffle,
                                                     preserve_class_dist=preserve_class_dist)
        if scaling is not None:
            if scaling == 'standardize':
                traind, means, stdev = standardize_data(data=traind, label_columns=['quality', 'label'])
                # Apply normalization to test data using means, stdev from train data
                testd_w_out_labels = testd.drop(columns=['quality', 'label'])
                testd_label_columns = testd.loc[:, ['quality', 'label']]
                testd_w_out_labels -= means
                testd_w_out_labels /= stdev
                testd = pd.concat([testd_w_out_labels, testd_label_columns], axis=1)
            if scaling == 'min_max_norm':
                traind, mins, maxs = normalize_data(data=traind, label_columns=['quality', 'label'])
                # Apply min-max-normalization to val and test data using mins, maxs from traindata
                testd_w_out_labels = testd.drop(columns=['quality', 'label'])
                testd_label_columns = testd.loc[:, ['quality', 'label']]
                testd_w_out_labels -= mins
                testd_w_out_labels /= (maxs - mins)
                testd = pd.concat([testd_w_out_labels, testd_label_columns], axis=1)

        if pca_dim != -1:
            traind, train_pca = perform_pca_reduction(data=traind, pca_dim=pca_dim, label_columns=['quality', 'label'])
            # Apply PCA transformation to test data
            testd = apply_fct_to_data(data=testd, label_columns=['quality', 'label'], function=train_pca.transform)

            if verbosity:
                print(f"Performed PCA dimensionality reduction to {pca_dim} principal components.")
                print("The singular values are:")
                print(train_pca.singular_values_)
                print("The percentage of variance explained by each of the selected components is:")
                print(train_pca.explained_variance_ratio_)

        if labelling == 'bmg':
            num_train_samples = traind.shape[0]
            num_bad = (traind.loc[:, 'label'].values == 0).sum()
            num_med = (traind.loc[:, 'label'].values == 1).sum()
            num_good = (traind.loc[:, 'label'].values == 2).sum()
            num_bad_te = (testd.loc[:, 'label'].values == 0).sum()
            num_med_te = (testd.loc[:, 'label'].values == 1).sum()
            num_good_te = (testd.loc[:, 'label'].values == 2).sum()
            if verbosity:
                print(f"In the training set there "
                      f"are {num_bad} ({(num_bad / num_train_samples).round(decimals=4)}%) bad, "
                      f"{num_med} ({(num_med / num_train_samples).round(decimals=4)}%) medium, "
                      f"{num_good} ({(num_good / num_train_samples).round(decimals=4)}%) good samples. ")
                print(f"In the test set there "
                      f"are {num_bad_te} bad, {num_med_te} medium, {num_good_te} good samples. ")
                baseline_pred = np.ones(testd.loc[:, 'label'].values.shape[0]) * 1
                print("The baseline metrics given an all majority prediction are:")
                b_acc, b_perc, b_rec, b_f1 = evaluate_class_predictions(
                    prediction=baseline_pred,
                    ground_truth=testd.loc[:, 'label'].values,
                    labels=np.unique(traind.loc[:, 'label'].values),
                    verbosity=True)

            if over_sample is not None:
                if over_sample == 'random':
                    ros = RandomOverSampler(sampling_strategy=calc_oversampling_strategy,
                                            shrinkage=None)
                    traind, _ = ros.fit_resample(X=traind, y=traind.loc[:, 'label'].values)

                elif over_sample == 'smote':
                    smote = SMOTE(sampling_strategy=calc_oversampling_strategy,
                                  k_neighbors=5)
                    traind, _ = smote.fit_resample(X=traind, y=traind.loc[:, 'label'].values)

                if verbosity:
                    num_train_samples = traind.shape[0]
                    num_bad = (traind.loc[:, 'label'].values == 0).sum()
                    num_med = (traind.loc[:, 'label'].values == 1).sum()
                    num_good = (traind.loc[:, 'label'].values == 2).sum()
                    print(f"After oversampling the training set consists of {num_train_samples} samples with "
                          f"are {num_bad} ({(num_bad / num_train_samples).round(decimals=4)}%) bad, "
                          f"{num_med} ({(num_med / num_train_samples).round(decimals=4)}%) medium, "
                          f"{num_good} ({(num_good / num_train_samples).round(decimals=4)}%) good samples. ")

            if under_sample is not None:
                if under_sample == 'random':
                    rus = RandomUnderSampler(sampling_strategy=calc_undersampling_strategy, replacement=False)
                    traind, _ = rus.fit_resample(X=traind, y=traind.loc[:, 'label'].values)

                if verbosity:
                    num_train_samples = traind.shape[0]
                    num_bad = (traind.loc[:, 'label'].values == 0).sum()
                    num_med = (traind.loc[:, 'label'].values == 1).sum()
                    num_good = (traind.loc[:, 'label'].values == 2).sum()
                    print(f"After undersampling the training set consists of {num_train_samples} samples with "
                          f"are {num_bad} ({(num_bad / num_train_samples).round(decimals=4)}%) bad, "
                          f"{num_med} ({(num_med / num_train_samples).round(decimals=4)}%) medium, "
                          f"{num_good} ({(num_good / num_train_samples).round(decimals=4)}%) good samples. ")

        return traind, testd


def data_pipeline_redwine(shuffle: bool = True,
                          preserve_class_dist: bool = True,
                          val_and_test: bool = False,
                          over_sample: Union[str, None] = None,
                          under_sample: Union[str, None] = None,
                          scaling: Union[str, None] = None,
                          pca_dim: int = -1,
                          labelling: str = 'bmg',
                          verbosity: bool = False) -> Tuple[pd.DataFrame, ...]:
    """
    Example workflow for loading and preprocessing of redwine data set.
    :return: Redwine data set, preprocessed and split into training and test set.
    """
    fn_red = "../data/wine_data/winequality-red.csv"
    data_red = load_wine(filename=fn_red, verbosity=False)
    return preprocess_wine(data=data_red,
                           shuffle=shuffle,
                           preserve_class_dist=preserve_class_dist,
                           val_and_test=val_and_test,
                           over_sample=over_sample,
                           under_sample=under_sample,
                           scaling=scaling,
                           pca_dim=pca_dim,
                           labelling=labelling,
                           verbosity=verbosity)


def data_pipeline_whitewine(shuffle: bool = True,
                            preserve_class_dist: bool = True,
                            val_and_test: bool = False,
                            over_sample: Union[str, None] = None,
                            under_sample: Union[str, None] = None,
                            scaling: Union[str, None] = None,
                            pca_dim: int = -1,
                            labelling: str = 'bmg',
                            verbosity: bool = False) -> Tuple[pd.DataFrame, ...]:
    """
    Example workflow for loading and preprocessing of whitewine data set.
    :return: Redwine data set, preprocessed and split into training and test set.
    """
    fn_white = "../data/wine_data/winequality-white.csv"
    data_white = load_wine(filename=fn_white, verbosity=False)
    return preprocess_wine(data=data_white,
                           shuffle=shuffle,
                           preserve_class_dist=preserve_class_dist,
                           val_and_test=val_and_test,
                           over_sample=over_sample,
                           under_sample=under_sample,
                           scaling=scaling,
                           pca_dim=pca_dim,
                           labelling=labelling,
                           verbosity=verbosity)


def data_pipeline_concat_red_white(shuffle: bool = True,
                                   preserve_class_dist: bool = True,
                                   val_and_test: bool = False,
                                   over_sample: Union[str, None] = None,
                                   under_sample: Union[str, None] = None,
                                   scaling: Union[str, None] = None,
                                   pca_dim: int = -1,
                                   labelling: str = 'bmg',
                                   verbosity: bool = False) -> Tuple[pd.DataFrame, ...]:

    fn_red = "../data/wine_data/winequality-red.csv"
    fn_white = "../data/wine_data/winequality-white.csv"
    data_red = load_wine(filename=fn_red, verbosity=False)
    data_white = load_wine(filename=fn_white, verbosity=False)

    data = pd.concat([data_red, data_white], axis=0)

    return preprocess_wine(data=data,
                           shuffle=shuffle,
                           preserve_class_dist=preserve_class_dist,
                           val_and_test=val_and_test,
                           over_sample=over_sample,
                           under_sample=under_sample,
                           scaling=scaling,
                           pca_dim=pca_dim,
                           labelling=labelling,
                           verbosity=verbosity)


if __name__ == '__main__':
    # load_wine("../data/wine_data/winequality-red.csv", verbosity=True)
    # load_wine("../data/wine_data/winequality-white.csv", verbosity=True)

    trainr = data_pipeline_redwine(verbosity=True, scaling='standardize', pca_dim=-1)[0]
    print(trainr.columns)
    quality = trainr.loc[:, 'label'].values
    alc = trainr.loc[:, 'alcohol'].values
    sulph = trainr.loc[:, 'sulphates'].values
    vol_acid = trainr.loc[:, 'volatile acidity'].values

    plt.scatter(x=alc, y=sulph, c=quality)
    plt.show()

    plt.scatter(x=alc, y=vol_acid, c=quality)
    plt.show()

    plt.scatter(x=sulph, y=vol_acid, c=quality)
    plt.show()

    # out_white = data_pipeline_whitewine(verbosity=True, scaling='min_max_norm', pca_dim=-1)[0]
    # plt.scatter(x=out_white.values[:, 0], y=out_white.values[:, 1], c=out_white.values[:, 12])
    # plt.show()

    # out_concat = data_pipeline_concat_red_white()

    # fn_white = "../data/wine_data/winequality-white.csv"
    # data_white = load_wine(filename=fn_white, verbosity=True)

    # fn_banknotes = "../data/banknotes_data/data_banknote_authentication.txt"
    # data_banknotes = load_comma_sep_csv(filename=fn_banknotes, verbosity=True)

    # fn_obesity = "../data/obesity_data/ObesityDataSet_raw_and_data_sinthetic.csv"
    # data_obesity = load_comma_sep_csv(filename=fn_obesity, verbosity=True)



