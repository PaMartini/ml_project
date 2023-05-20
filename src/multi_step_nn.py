import numpy as np
import pandas as pd
import torch

from load_data import *
from evaluation import *
from neural_net import *


def add_min_maj_label(wine_df: pd.DataFrame, label_col: str = 'label'):
    """
    Function for adding a column with entry 1 if sample belongs to minority class, 0 otherwise.
    :param wine_df: Dataframe with wine data.
    :param label_col: Column where quality labels are stored.
    :return: Altered dataframe.
    """
    label_array = wine_df.loc[:, label_col].values
    # New label is 1 if minority class, 0 otherwise
    new_label_array = np.logical_or(label_array == 0, label_array == 2).astype(float)
    wine_df['min_maj_label'] = new_label_array

    return wine_df


def get_multi_step_dfs(colour: str = 'red',
                       scaling: Union[None, str] = None,
                       over_sample: Union[None, str] = None,
                       drop_column: list[str, ...] = None) \
        -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray], Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:

    assert colour in {'red', 'white', 'red_white'}, "Parameter 'colour' must be 'red', 'white' or 'red_white'."

    if drop_column is None:
        drop_column = ['quality']

    if colour == 'red':
        traindf, testdf = data_pipeline_redwine(val_and_test=False,
                                                scaling=scaling, over_sample=over_sample)
    elif colour == 'white':
        traindf, testdf = data_pipeline_whitewine(val_and_test=False,
                                                  scaling=scaling, over_sample=over_sample)
    elif colour == 'red_white':
        traindf, testdf = data_pipeline_concat_red_white(val_and_test=False,
                                                         scaling=scaling, over_sample=over_sample)

    traindf = traindf.drop(columns=drop_column)
    testdf = testdf.drop(columns=drop_column)

    # Add minority/majority class labels
    traindf = add_min_maj_label(wine_df=traindf, label_col='label')
    testdf = add_min_maj_label(wine_df=testdf, label_col='label')

    # Create dfs with only minority class samples, remove 'min_maj_label' column
    min_traindf = traindf.loc[traindf['min_maj_label'] == 1, :].copy(deep=True)
    min_testdf = testdf.loc[testdf['min_maj_label'] == 1, :].copy(deep=True)

    # Drop min_maj_labels, change quality labels to 0 = bad, 1 = good
    min_traindf = min_traindf.drop(columns=['min_maj_label'])
    min_testdf = min_testdf.drop(columns=['min_maj_label'])

    min_traindf.loc[min_traindf.loc[:, 'label'].values == 2, 'label'] = 1
    min_testdf.loc[min_testdf.loc[:, 'label'].values == 2, 'label'] = 1

    # Drop quality labels fom dataframes, only keep min_maj_labels
    traindf = traindf.drop(columns=['label'])
    testdf = testdf.drop(columns=['label'])

    # Define labels that are present in the datasets
    min_maj_labels = np.array([0, 1])
    quality_labels = np.array([0, 2])

    return (traindf, testdf, min_maj_labels), (min_traindf, min_testdf, quality_labels)


def get_wine_dataloaders(wine_train_df: pd.DataFrame,
                         wine_test_df: pd.DataFrame,
                         batch_size: int = 1,
                         shuffle: bool = False) -> Tuple[DataLoader, DataLoader]:
    train_ds = WineDataset(wine_df=wine_train_df)
    test_ds = WineDataset(wine_df=wine_test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


def run_multistep_training(n_epochs: Tuple[int, int] = (20, 20),
                           test: bool = False,
                           plot_losses: bool = False,
                           save_losses: bool = False,
                           plot_save_metrics: bool = False):

    a, b = get_multi_step_dfs(colour='red',
                              scaling='min_max_norm',
                              over_sample='random',
                              drop_column=['quality'])

    traindf, testdf, min_maj_labels = a
    min_traindf, min_testdf, quality_labels = b

    trainl_mm, testl_mm = get_wine_dataloaders(wine_train_df=traindf, wine_test_df=testdf,
                                               batch_size=10, shuffle=True)
    trainl_min, testl_min = get_wine_dataloaders(wine_train_df=min_traindf, wine_test_df=min_testdf,
                                                 batch_size=10, shuffle=True)


    num_labels_mm = min_maj_labels.shape[0]
    num_labels_min = quality_labels.shape[0]
    # Initialize model with the correct shape.
    model_mm = WineNet(in_size=11, out_size=num_labels_mm)
    model_min = WineNet(in_size=11, out_size=num_labels_min)
    # Initialize optimizer.
    optimizer_mm = torch.optim.Adam(model_mm.parameters(), lr=0.0001, weight_decay=0.0)
    optimizer_min = torch.optim.Adam(model_mm.parameters(), lr=0.0001, weight_decay=0.0)
    # Calculate class weights
    # class_weights = None
    class_weights_mm = calc_class_weights(train_loader=trainl_mm, verbosity=True)
    class_weights_min = calc_class_weights(train_loader=trainl_min, verbosity=True)
    # Initialize loss function
    # loss_fct_mm = nn.CrossEntropyLoss(weight=class_weights_mm, reduction='mean', label_smoothing=0)
    # loss_fct_min = nn.CrossEntropyLoss(weight=class_weights_min, reduction='mean', label_smoothing=0)
    loss_fct_mm = nn.BCEWithLogitsLoss(weight=class_weights_mm)
    loss_fct_min = nn.BCEWithLogitsLoss(weight=class_weights_min)
    # loss_fct = FocalLoss(include_background=True, gamma=10.0, weight=class_weights, reduction='mean')
    # loss_fct = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0)

    trained_model_mm, train_losses_mm, val_losses_mm, metrics_dict_mm = train_loop(train_loader=trainl_mm,
                                                                                   val_loader=testl_mm,
                                                                                   labels=min_maj_labels,
                                                                                   model=model_mm,
                                                                                   optimizer=optimizer_mm,
                                                                                   loss_fct=loss_fct_mm,
                                                                                   num_epochs=n_epochs[0],
                                                                                   track_metrics=plot_save_metrics,
                                                                                   save=True,
                                                                                   verbosity=True)

    trained_model_min, train_losses_min, val_losses_min, metrics_dict_min = train_loop(train_loader=trainl_min,
                                                                                       val_loader=testl_min,
                                                                                       labels=quality_labels,
                                                                                       model=model_min,
                                                                                       optimizer=optimizer_min,
                                                                                       loss_fct=loss_fct_min,
                                                                                       num_epochs=n_epochs[1],
                                                                                       track_metrics=plot_save_metrics,
                                                                                       save=True,
                                                                                       verbosity=True)
    if save_losses:
        np.save('train_losses_mm.npy', train_losses_mm)
        np.save('val_losses_mm.npy', val_losses_mm)
    if plot_losses:
        plot_loss(train_loss=train_losses_mm, val_loss=val_losses_mm)
        plot_loss(train_loss=train_losses_min, val_loss=val_losses_min)

    if plot_save_metrics:
        with open('metrics_dict_mm.pickle', 'wb') as f:
            pickle.dump(metrics_dict_mm, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('metrics_dict_min.pickle', 'wb') as f:
            pickle.dump(metrics_dict_min, f, protocol=pickle.HIGHEST_PROTOCOL)
        plot_metrics(metrics_dict=metrics_dict_mm, single_plots=False)
        plot_metrics(metrics_dict=metrics_dict_min, single_plots=False)

    if test:
        multistep_validation(testl_min=testl_min,
                             model_mm=trained_model_mm,
                             model_min=trained_model_min)


def multistep_validation(testl_min: DataLoader,
                         model_mm: WineNet,
                         model_min: WineNet):

    val_gt_labels = np.array([])
    predictions = np.array([])

    for i, data in tqdm(enumerate(testl_min), total=len(testl_min)):
        x = data[0].to(torch.float32)
        # y = data[1].to(torch.float32)
        int_y = np.array(data[2])
        val_gt_labels = np.hstack((val_gt_labels, int_y))
        model_mm.eval()
        model_min.eval()
        pred_array = np.zeros(x.shape[0])
        with torch.no_grad():
            mm_out = model_mm(x)  # output is softmax
            mm_pred = np.array(torch.argmax(mm_out, dim=1).detach())
            # For samples predicted to be in majority class accept prediction, set pred to 1 (og label)
            pred_array[mm_pred == 0] = 1
            # For samples predicted to be in minority class, if some exist, pass through second net
            if (mm_pred == 1).sum() >= 1:
                min_samples = x[mm_pred == 1, :]
                min_out = model_min(min_samples)
                min_pred = np.array(torch.argmax(min_out, dim=1).detach())
                # Change entries to og labels, insert in pred array
                min_pred[min_pred == 1] = 2
                pred_array[mm_pred == 1] = min_pred
            # Update predictions array
            predictions = np.hstack((predictions, pred_array))

    evaluate_class_predictions(prediction=predictions,
                               ground_truth=val_gt_labels,
                               labels=np.array([0, 1, 2]),
                               verbosity=True)


if __name__ == '__main__':
    run_multistep_training(n_epochs=(50, 50), test=True, plot_losses=True, plot_save_metrics=True)

    print('done')


