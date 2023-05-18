import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import seaborn as sns
import random
import matplotlib.pyplot as plt
from load_data import *
from naive_bayes import *
from random_forests import *
from svm import *


def make_dict_entries(dict_: dict, key: str, idx: int, metrics: Tuple) -> dict:
    """
    Auxiliary function for making a dictionary entry in the metrics dict.
    :param dict_: metrics_dict
    :param key: -
    :param idx: -
    :param metrics: -
    :return: -
    """
    dict_[key]['acc'][idx] = metrics[0].copy()
    dict_[key]['prec'][idx, :] = metrics[1].copy()
    dict_[key]['rec'][idx, :] = metrics[2].copy()
    dict_[key]['f1'][idx, :] = metrics[3].copy()
    return dict_


def generate_baseline_results(colour: str = 'red',
                              num_trials: int = 100,
                              scaling: Union[None, str] = None,
                              over_sample: Union[None, str] = None,
                              save_dir: Union[None, str] = None,
                              verbosity: bool = False) -> dict:
    """
    Function for producing baseline results for the scaling and oversampling method specified in input.
    :param colour: Which data set to use, 'red' or 'white'.
    :param num_trials: Number of trials to perform.
    :param scaling: Which scaling method to use.
    :param over_sample: Which oversampling method to use.
    :param save_dir: Directory to save results to.
    :param verbosity: Whether to print results, or not.
    :return: Dictionary with calculated metrics. Has structure {'classifier': {'metric': value array}}.
    """
    assert colour in {'red', 'white', 'red_white'}, "Colour must be 'red', 'white' or 'red_white'."

    dummy_dict = {'acc': np.zeros(num_trials),
                  'prec': np.zeros((num_trials, 3)),
                  'rec': np.zeros((num_trials, 3)),
                  'f1': np.zeros((num_trials, 3))}
    metrics_dict = {'svm': deepcopy(dummy_dict), 'dt': deepcopy(dummy_dict),
                    'rf': deepcopy(dummy_dict), 'nb': deepcopy(dummy_dict)}

    for i in tqdm(range(num_trials)):
        if colour == 'red':
            traind, testd = data_pipeline_redwine(verbosity=False, scaling=scaling, over_sample=over_sample)
            # Delete quality columns in data frames:
            traind = traind.drop(columns=['quality'])
            testd = testd.drop(columns=['quality'])
        elif colour == 'white':
            traind, testd = data_pipeline_whitewine(verbosity=False, scaling=scaling, over_sample=over_sample)
            # Delete quality columns in data frames:
            traind = traind.drop(columns=['quality'])
            testd = testd.drop(columns=['quality'])
        elif colour == 'red_white':
            traind, testd = data_pipeline_concat_red_white(verbosity=False, scaling=scaling, over_sample=over_sample)
            # Delete quality columns in data frames:
            traind = traind.drop(columns=['quality'])
            testd = testd.drop(columns=['quality'])

        if scaling is not None:  # SVM intractable if no scaling is used
            metrics = train_svm_model(train_data=traind, label_column='label',
                                      config=None, test_data=testd, verbosity=False)[1:]
            metrics_dict = make_dict_entries(dict_=metrics_dict, key='svm', idx=i, metrics=metrics)

        metrics = train_dt_classifier(train_data=traind, label_column='label',
                                      config=None, test_data=testd, verbosity=False)[1:]
        metrics_dict = make_dict_entries(dict_=metrics_dict, key='dt', idx=i, metrics=metrics)

        metrics = train_random_forest(train_data=traind, label_column='label',
                                      config=None, test_data=testd, verbosity=False)[1:]
        metrics_dict = make_dict_entries(dict_=metrics_dict, key='rf', idx=i, metrics=metrics)

        metrics = train_gaussian_naive_bayes(train_data=traind, label_column='label',
                                             config=None, test_data=testd, verbosity=False)[1:]
        metrics_dict = make_dict_entries(dict_=metrics_dict, key='nb', idx=i, metrics=metrics)

    if verbosity:
        print(f"### Results averaged over {num_trials} trials for "
              f"the wine colour {colour} with default settings are ... ###")

        for key, val in metrics_dict.items():
            print(f"### Results for {key}:")
            for k, v in val.items():
                np.set_printoptions(precision=4)
                mean_v = np.mean(v, axis=0)
                class_mean_v = np.mean(mean_v)
                std_v = np.std(v, axis=0)
                print(f"## {k}: trial means: {mean_v}, "
                      f"trial std: {std_v}, "
                      f"class mean: {np.round(class_mean_v, decimals=4)}")

    if save_dir is not None:
        fp = save_dir + f"base_res_{colour}_{str(scaling)}_{str(over_sample)}.pickle"
        with open(fp, 'wb') as f:
            pickle.dump(metrics_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return metrics_dict


def run_baseline_grid_search(colour: str = 'red',
                             num_trials: int = 100,
                             save_dir: Union[None, str] = None,
                             verbosity: bool = False) -> Tuple[dict, dict]:
    """
    Function for running the grid-search over the scaling and oversampling methods. Saves results into dictionaries.
    :param colour: Which data set to use, 'red' or 'white'.
    :param num_trials: Number of trials to perform.
    :param save_dir: Directory to save results to.
    :param verbosity: Whether to print results, or not.
    :return: Dictionaries with structures {'classifier': 'optimal scaling and oversampling method combination'},
    {'classifier': {'scaling and oversampling method combination': F_1 value}}
    """
    f1_dict = {}
    for s in [None, 'standardize', 'min_max_norm']:
        for o in [None, 'random', 'smote']:
            metr_dict = generate_baseline_results(colour=colour,
                                                  num_trials=num_trials,
                                                  scaling=s,
                                                  over_sample=o,
                                                  save_dir='../results/')

            if len(f1_dict) == 0:  # initialize dictionary with classifiers as keys
                for key, val in metr_dict.items():
                    f1_dict[key] = {}

            for key, val in metr_dict.items():  # Iterate over classifiers (key) and their metric dicts (val)
                f1_dict[key][str(s) + '_' + str(o)] = val['f1'].mean()

    if save_dir is not None:
        fp = save_dir + f"f1_dict_baseline_{colour}.pickle"
        with open(fp, 'wb') as f:
            pickle.dump(f1_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    f1_max_dict = {}
    for key, val in f1_dict.items():  # Iterate over classifiers (key) and their f1 dict (val)
        f1_max_dict[key] = max(val, key=val.get)
        if verbosity:
            print(f"For the classifier '{key}' the best configuration is '{f1_max_dict[key]}' "
                  f"with corresponding F1 value {f1_dict[key][f1_max_dict[key]]}.")

    if save_dir is not None:
        fp = save_dir + f"f1_max_dict_baseline_{colour}.pickle"
        with open(fp, 'wb') as f:
            pickle.dump(f1_max_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return f1_max_dict, f1_dict


def run_parameter_tuning_svm_rf_red():
    traind, testd = data_pipeline_redwine(verbosity=False, scaling='min_max_norm', over_sample='smote')
    # Delete quality columns in data frames:
    traind = traind.drop(columns=['quality'])
    testd = testd.drop(columns=['quality'])
    run_parameter_tuning_svm(train_data=traind, label_column='label',
                             file_name='../configurations/red_best_svm_config.pickle')

    traind, testd = data_pipeline_redwine(verbosity=False, scaling=None, over_sample='smote')
    # Delete quality columns in data frames:
    traind = traind.drop(columns=['quality'])
    testd = testd.drop(columns=['quality'])
    run_parameter_tuning_rf(train_data=traind, label_column='label',
                            file_name='../configurations/red_best_rf_config.pickle')


def run_parameter_tuning_svm_rf_white():
    traind, testd = data_pipeline_whitewine(verbosity=False, scaling='standardize', over_sample='smote')
    # Delete quality columns in data frames:
    traind = traind.drop(columns=['quality'])
    testd = testd.drop(columns=['quality'])
    run_parameter_tuning_svm(train_data=traind, label_column='label',
                             file_name='../configurations/white_best_svm_config.pickle')

    traind, testd = data_pipeline_whitewine(verbosity=False, scaling=None, over_sample='random')
    # Delete quality columns in data frames:
    traind = traind.drop(columns=['quality'])
    testd = testd.drop(columns=['quality'])
    run_parameter_tuning_rf(train_data=traind, label_column='label',
                            file_name='../configurations/white_best_rf_config.pickle')


def generate_improved_svm_rf_results(colour: str = 'red',
                                     num_trials: int = 100,
                                     save_dir: Union[None, str] = None,
                                     verbosity: bool = False) -> dict:

    dummy_dict = {'acc': np.zeros(num_trials),
                  'prec': np.zeros((num_trials, 3)),
                  'rec': np.zeros((num_trials, 3)),
                  'f1': np.zeros((num_trials, 3))}
    metrics_dict = {'svm': deepcopy(dummy_dict), 'rf': deepcopy(dummy_dict)}

    if colour == 'red':
        with open('../configurations/red_best_svm_config.pickle', 'rb') as f:
            best_svm_param = pickle.load(f)
        with open('../configurations/red_best_rf_config.pickle', 'rb') as f:
            best_rf_param = pickle.load(f)

    if colour == 'white':
        with open('../configurations/white_best_svm_config.pickle', 'rb') as f:
            best_svm_param = pickle.load(f)
        with open('../configurations/white_best_rf_config.pickle', 'rb') as f:
            best_rf_param = pickle.load(f)

    if verbosity:
        print(f"###### The best parameters for SVM amd RF are: ######")
        print(best_svm_param)
        print(best_rf_param)

    for i in tqdm(range(num_trials)):
        if colour == 'red':
            traind_svm, testd_svm = data_pipeline_redwine(verbosity=False, scaling='min_max_norm', over_sample='smote')
            # Delete quality columns in data frames:
            traind_svm = traind_svm.drop(columns=['quality'])
            testd_svm = testd_svm.drop(columns=['quality'])
            traind_rf, testd_rf = data_pipeline_redwine(verbosity=False, scaling=None, over_sample='smote')
            # Delete quality columns in data frames:
            traind_rf = traind_rf.drop(columns=['quality'])
            testd_rf = testd_rf.drop(columns=['quality'])
        elif colour == 'white':
            traind_svm, testd_svm = data_pipeline_redwine(verbosity=False, scaling='standardize', over_sample='smote')
            # Delete quality columns in data frames:
            traind_svm = traind_svm.drop(columns=['quality'])
            testd_svm = testd_svm.drop(columns=['quality'])
            traind_rf, testd_rf = data_pipeline_redwine(verbosity=False, scaling=None, over_sample='random')
            # Delete quality columns in data frames:
            traind_rf = traind_rf.drop(columns=['quality'])
            testd_rf = testd_rf.drop(columns=['quality'])


        metrics = train_svm_model(train_data=traind_svm, label_column='label',
                                  config=best_svm_param, test_data=testd_svm, verbosity=False)[1:]
        metrics_dict = make_dict_entries(dict_=metrics_dict, key='svm', idx=i, metrics=metrics)

        metrics = train_random_forest(train_data=traind_rf, label_column='label',
                                      config=best_rf_param, test_data=testd_rf, verbosity=False)[1:]
        metrics_dict = make_dict_entries(dict_=metrics_dict, key='rf', idx=i, metrics=metrics)

    if verbosity:
        print(f"###### The results for the {colour} dataset are ... ######")
        for key, val in metrics_dict.items():
            print(f"## Method {key}:")
            for k, v in val.items():
                np.set_printoptions(precision=4)
                mean_v = np.mean(v, axis=0)
                class_mean_v = np.mean(mean_v)
                std_v = np.std(v, axis=0)
                print(f"## {k}: trial means: {mean_v}, "
                      f"trial std: {std_v}, "
                      f"class mean: {np.round(class_mean_v, decimals=4)}")

    if save_dir is not None:
        fp = save_dir + f"improved_results_{colour}.pickle"
        with open(fp, 'wb') as f:
            pickle.dump(metrics_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return metrics_dict


def drop_correlated_features(df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             corr_thresh: float = 0.5,
                             drop_columns: list[str, ...] = None,
                             verbosity: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:

    working_df = df.drop(columns=drop_columns).copy(deep=True)
    var = working_df.var(axis=0)
    rand_ = False
    # Decide whether to choose which of two correlated features at random
    if np.all(np.around(var.values, decimals=6) == np.around(var.values[0], decimals=6)):
        rand_ = True

    corr = working_df.corr()
    np.fill_diagonal(a=corr.values, val=0)
    drop_features = []
    features = list(working_df.columns)
    for ft1 in features:
        for ft2 in features:
            if abs(corr.loc[ft1, ft2]) >= corr_thresh:
                if not rand_:
                    if var.loc[ft1] >= var.loc[ft2]:
                        drop_features.append(ft2)
                    else:
                        drop_features.append(ft1)
                else:
                    drop_features.append(random.choice([ft1, ft2]))
                    # Or drop ft correlated to more other features

    drop_features = list(set(drop_features))
    df = df.drop(columns=drop_features).copy(deep=True)
    test_df = test_df.drop(columns=drop_features).copy(deep=True)

    if verbosity:
        print(f"Dropped the features {drop_features}")
        new_corr = df.drop(columns=drop_columns).corr()
        if not len(new_corr.columns) == 0:
            sns.heatmap(new_corr, xticklabels=new_corr.columns, yticklabels=new_corr.columns, annot=True)
        #plt.show()

    return df, test_df


def generate_improved_nb_results(colour: str = 'red',
                                 num_trials: int = 100,
                                 pca_dim: int = -1,
                                 rem_corr: Union[None, float] = None,
                                 save_dir: Union[None, str] = None,
                                 verbosity: bool = False) -> dict:
    metrics_dict = {'nb': {'acc': np.zeros(num_trials),
                           'prec': np.zeros((num_trials, 3)),
                           'rec': np.zeros((num_trials, 3)),
                           'f1': np.zeros((num_trials, 3))}}

    for i in tqdm(range(num_trials)):
        if colour == 'red':
            traind_nb, testd_nb = data_pipeline_redwine(verbosity=False, scaling='standardize',
                                                        over_sample=None, pca_dim=pca_dim)
            # Delete quality columns in data frames:
            traind_nb = traind_nb.drop(columns=['quality'])
            testd_nb = testd_nb.drop(columns=['quality'])
        elif colour == 'white':
            traind_nb, testd_nb = data_pipeline_redwine(verbosity=False, scaling='standardize',
                                                        over_sample=None, pca_dim=pca_dim)
            # Delete quality columns in data frames:
            traind_nb = traind_nb.drop(columns=['quality'])
            testd_nb = testd_nb.drop(columns=['quality'])

        if rem_corr is not None:
            traind_nb, testd_nb = drop_correlated_features(df=traind_nb, test_df=testd_nb,
                                                           corr_thresh=rem_corr,
                                                           drop_columns=['label'],
                                                           verbosity=False)

        if not traind_nb.shape[1] <= 1:
            # If all features are removed, this is a failure case. All metrics are 0, continue.
            metrics = train_gaussian_naive_bayes(train_data=traind_nb, label_column='label',
                                                 config=None, test_data=testd_nb, verbosity=False)[1:]
            metrics_dict = make_dict_entries(dict_=metrics_dict, key='nb', idx=i, metrics=metrics)

    if verbosity:
        print(f"###### The results for the {colour} dataset are ... ######")
        for key, val in metrics_dict.items():
            print(f"## Method {key}:")
            for k, v in val.items():
                np.set_printoptions(precision=4)
                mean_v = np.mean(v, axis=0)
                class_mean_v = np.mean(mean_v)
                std_v = np.std(v, axis=0)
                print(f"## {k}: trial means: {mean_v}, "
                      f"trial std: {std_v}, "
                      f"class mean: {np.round(class_mean_v, decimals=4)}")

    if save_dir is not None:
        fp = save_dir + f"improved_results_nb_{colour}.pickle"
        with open(fp, 'wb') as f:
            pickle.dump(metrics_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return metrics_dict


def test_nb_with_pca(colour: str = 'red'):
    for i in list(range(-1, 12)):
        if i == 0:
            continue
        print(f"###### PCA dim {i} ######")
        if i == -1:
            print("## PCA dim -1 corresponds to no dimensionality reduction.")
        generate_improved_nb_results(colour=colour, pca_dim=i, rem_corr=None, verbosity=True, num_trials=100)


def test_nb_with_feature_selection(colour: str = 'red'):
    print("###### No feature selection ######")
    generate_improved_nb_results(colour=colour, pca_dim=-1, rem_corr=None, verbosity=True, num_trials=100)
    for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        print(f"###### Correlation threshold {thresh} ######")
        generate_improved_nb_results(colour=colour, pca_dim=-1, rem_corr=thresh, verbosity=True, num_trials=100)


if __name__ == '__main__':
    # generate_baseline_results(colour='red', num_trials=100,
    #                           scaling='standardize', over_sample='random', verbosity=True)
    # run_baseline_grid_search(colour='white', num_trials=100, save_dir='../results/', verbosity=True)

    # run_parameter_tuning_svm_rf_red()
    # run_parameter_tuning_svm_rf_white()

    # generate_improved_svm_rf_results(colour='white', num_trials=100, save_dir='../results/', verbosity=True)
    # generate_improved_nb_results(colour='red', pca_dim=-1, rem_corr=False, verbosity=True, num_trials=100)

    # test_nb_with_pca(colour='red')
    # test_nb_with_pca(colour='white')

    # test_nb_with_feature_selection(colour='red')
    # test_nb_with_feature_selection(colour='white')

    print('done')
