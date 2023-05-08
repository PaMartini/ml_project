
from tqdm import tqdm
from copy import deepcopy
from load_data import *
from naive_bayes import *
from random_forests import *
from svm import *


def make_dict_entries(dict_: dict, key: str, idx: int, metrics: Tuple) -> dict:
    dict_[key]['acc'][idx] = metrics[0]
    dict_[key]['prec'][idx, :] = metrics[1]
    dict_[key]['rec'][idx, :] = metrics[2]
    dict_[key]['f1'][idx, :] = metrics[3]
    return dict_


def generate_baseline_results(colour: str = 'red',
                              num_trials: int = 100,
                              scaling: Union[None, str] = None,
                              over_sample: Union[None, str] = None,
                              save_dir: Union[None, str] = None,
                              verbosity: bool = False) -> dict:
    assert colour in {'red', 'white', 'red_white'}, "Colour must be 'red', 'white' or 'red_white'."
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

    dummy_dict = {'acc': np.zeros(num_trials),
                  'prec': np.zeros((num_trials, 3)),
                  'rec': np.zeros((num_trials, 3)),
                  'f1': np.zeros((num_trials, 3))}
    metrics_dict = {'svm': deepcopy(dummy_dict), 'dt': deepcopy(dummy_dict),
                    'rf': deepcopy(dummy_dict), 'nb': deepcopy(dummy_dict)}

    for i in tqdm(range(num_trials)):
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

    print(f1_dict)

    if save_dir is not None:
        fp = save_dir + f"f1_dict_baseline.pickle"
        with open(fp, 'wb') as f:
            pickle.dump(f1_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    f1_max_dict = {}
    for key, val in f1_dict.items():  # Iterate over classifiers (key) and their f1 dict (val)
        f1_max_dict[key] = max(val, key=val.get)
        if verbosity:
            print(f"For the classifier '{key}' the best configuration is '{f1_max_dict[key]}' "
                  f"with corresponding F1 value {f1_dict[key][f1_max_dict[key]]}.")

    if save_dir is not None:
        fp = save_dir + f"f1_max_dict_baseline.pickle"
        with open(fp, 'wb') as f:
            pickle.dump(f1_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return f1_max_dict, f1_dict


if __name__ == '__main__':
    # generate_baseline_results(colour='white', num_trials=100, scaling='min_max_norm', over_sample=None)
    run_baseline_grid_search(colour='red', num_trials=100, save_dir='../results/', verbosity=True)

    print('done')
