import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from monai.losses import FocalLoss
from tqdm import tqdm
from datetime import datetime
import os
import pickle

from load_data import *
from evaluation import *


# Dataset ##############################################################################################################
class WineDataset(Dataset):
    """
    Dataset class for the wine quality dataset from the UCI ml repo.
    Expects a pandas dataframe with structure [samples]x[features, labels] as an input.
    For each sample returns the feature vector, the one hot encoded label vector and the integer label.
    """
    def __init__(self, wine_df: pd.DataFrame):
        self.df = wine_df
        self.unique_labels, self.num_labels = self.find_labels()

    def find_labels(self):
        labels = np.unique(self.df.iloc[:, -1].values)
        num_labels = labels.shape[0]
        return labels, num_labels

    def __getitem__(self, index):
        row = self.df.iloc[index].values
        features = row[:-1]
        int_label = row[-1]
        # One hot encoding, assuming label structure is 0,1,2,...
        label = np.zeros(self.num_labels)
        label[int(row[-1])] = 1
        return features, label, int_label

    def __len__(self):
        return len(self.df)


def get_data_loaders_wine_data(colour: str = 'red',
                               val_and_test: bool = False,
                               scaling: Union[None, str] = None,
                               over_sample: Union[None, str] = None,
                               batch_size: int = 1,
                               label_column: str = 'label',
                               drop_column: list[str, ...] = None,
                               shuffle: bool = False) \
        -> Union[Tuple[DataLoader, DataLoader, DataLoader, np.ndarray], Tuple[DataLoader, DataLoader, np.ndarray]]:

    assert colour in {'red', 'white', 'red_white'}, "Parameter 'colour' must be 'red', 'white' or 'red_white'."

    if drop_column is None:
        drop_column = ['quality']

    if val_and_test:
        if colour == 'red':
            traindf, valdf, testdf = data_pipeline_redwine(val_and_test=val_and_test,
                                                           scaling=scaling, over_sample=over_sample)
        elif colour == 'white':
            traindf, valdf, testdf = data_pipeline_whitewine(val_and_test=val_and_test,
                                                             scaling=scaling, over_sample=over_sample)
        elif colour == 'red_white':
            traindf, valdf, testdf = data_pipeline_concat_red_white(val_and_test=val_and_test,
                                                                    scaling=scaling, over_sample=over_sample)

        traindf = traindf.drop(columns=drop_column)
        valdf = testdf.drop(columns=drop_column)
        testdf = testdf.drop(columns=drop_column)

        traind = WineDataset(traindf)
        vald = WineDataset(valdf)
        testd = WineDataset(testdf)

        train_loader = DataLoader(traind, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(vald, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(testd, batch_size=batch_size, shuffle=shuffle)

        # Find labels
        labels = np.union1d(traindf.loc[:, label_column].values, valdf.loc[:, label_column].values)
        labels = np.union1d(labels, testdf.loc[:, label_column].values)

        return train_loader, val_loader, test_loader, labels

    else:
        if colour == 'red':
            traindf, testdf = data_pipeline_redwine(val_and_test=val_and_test,
                                                    scaling=scaling, over_sample=over_sample)
        elif colour == 'white':
            traindf, testdf = data_pipeline_whitewine(val_and_test=val_and_test,
                                                      scaling=scaling, over_sample=over_sample)
        elif colour == 'red_white':
            traindf, testdf = data_pipeline_concat_red_white(val_and_test=val_and_test,
                                                             scaling=scaling, over_sample=over_sample)

        traindf = traindf.drop(columns=drop_column)
        testdf = testdf.drop(columns=drop_column)

        traind = WineDataset(traindf)
        testd = WineDataset(testdf)

        train_loader = DataLoader(traind, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(testd, batch_size=batch_size, shuffle=shuffle)

        labels = np.union1d(traindf.loc[:, label_column].values, testdf.loc[:, label_column].values)

        return train_loader, test_loader, labels


# Model ################################################################################################################
class WineNet(torch.nn.Module):
    def __init__(self, in_size: int = 11, out_size: int = 1):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.Linear(in_features=self.in_size, out_features=6 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=6 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=6 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=4 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=4 * self.in_size, out_features=self.out_size),
        )

        self.net2 = nn.Sequential(
            nn.Linear(in_features=self.in_size, out_features=6 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=6 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=6 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=6 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=6 * self.in_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=6 * self.in_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=self.out_size),
        )

        self.net3 = nn.Sequential(
            nn.Linear(in_features=self.in_size, out_features=9 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=9 * self.in_size, out_features=9 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=9 * self.in_size, out_features=9 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=9 * self.in_size, out_features=9 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=9 * self.in_size, out_features=9 * self.in_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=9 * self.in_size, out_features=6 * self.in_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=4 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=4 * self.in_size, out_features=self.out_size),
        )

        self.last_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net2(x)
        x = self.last_activation(x)
        return x


# Auxiliary functions ##################################################################################################
def save_ckp(state_dict: dict, checkpoint_dir: str = "../checkpoints/") -> None:
    """
    Function for saving a checkpoint.
    :param state_dict: Dictionary with the model and optimizer state.
    :param checkpoint_dir: Directory where to store the checkpoint file.
    :return:
    """
    fp = os.path.join(checkpoint_dir, 'model_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.pt')
    torch.save(state_dict, fp)
    print(f'Saved model.')


def load_ckp(ckp_name: str,
             model: WineNet,
             optimizer: Type[torch.optim.Optimizer] = None,
             checkpoint_dir: str = "../checkpoints/",
             ) -> Tuple[Any, Any, Any]:
    """
    Function for loading a saved checkpoint of a model and optimizer.
    :param ckp_name: Name of checkpoint file.
    :param model: Model for which model state checkpoint is loaded.
    :param optimizer: Optimizer for which optimizer state checkpoint is loaded.
    :param checkpoint_dir: Directory where checkpoint file is stored.
    :return:
    """
    ckp = torch.load(os.path.join(checkpoint_dir, ckp_name))
    model.load_state_dict(ckp['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckp['optimizer'])
    return model, optimizer, ckp['epoch']


def plot_loss(train_loss: np.ndarray, val_loss: np.ndarray):
    """
    Function for plotting the train and validation loss.
    :param train_loss: Array with train loss as entries. shape: (num epochs, )
    :param val_loss: Array with val loss as entries. shape: (num epochs, )
    :return: None
    """
    plt.plot(np.arange(len(train_loss)), train_loss, label='Training loss')
    plt.plot(np.arange(len(val_loss)), val_loss, label='Validation loss')
    plt.yscale('log')
    plt.legend()
    # plt.savefig('losses_plot.png')
    plt.show()


def plot_metrics(metrics_dict: dict, single_plots: bool = False) -> None:
    """
    Function for plotting the performance metrics of calculated during the training of the model.
    :param metrics_dict: Dictionary with the metrics for each epoch stored in numpy arrays.
    :param single_plots: Whether to produce a single plot for each of the metrics.
    :return: None
    """
    # Plot metrics for individual classes.
    for i, key in enumerate(metrics_dict):
        if key == 'accuracy':
            labels = key
        else:
            labels = np.array([key + " " + str(j) for j in range(metrics_dict[key].shape[1])])
        plt.plot(np.arange(len(metrics_dict[key])), metrics_dict[key], label=labels)
        if single_plots:
            plt.legend()
            plt.show()
    if not single_plots:
        plt.legend()
        plt.show()

    # Plot metrics averaged over classes.
    for key in metrics_dict:
        if key == 'accuracy':
            plt.plot(np.arange(len(metrics_dict[key])), metrics_dict[key], label=key)
        else:
            plt.plot(np.arange(len(metrics_dict[key])), metrics_dict[key].sum(axis=1) / 3, label=key)
    plt.legend()
    plt.show()


def check_model():
    """
    Function for checking the structure of the WineNet model.
    :return: None
    """
    model = WineNet()
    summary(model, input_size=(11,))


def calc_class_weights(train_loader, verbosity: bool = False) -> torch.Tensor:
    """
    Function for calculating class weights to balance the classes inversely to the number of samples belonging to them.
    :param train_loader: Train loader
    :param verbosity: Whether to print the calculated class weights, or not.
    :return: Array with the class weights.
    """
    if verbosity:
        print("## Calculating class weights ...")
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        y = data[1].to(torch.float32)
        if i == 0:
            class_counts = y.sum(axis=0)
            batch_size = y.shape[0]
        else:
            class_counts += y.sum(axis=0)

    n_samples = batch_size * len(train_loader)
    n_classes = class_counts.shape[0]

    class_weights = n_samples * torch.ones(class_counts.shape[0]) / (n_classes * class_counts)

    if verbosity:
        print(f"The calculated class weights are {class_weights}.")

    return class_weights


# Training #############################################################################################################
def train_loop(train_loader: DataLoader,
               val_loader: DataLoader,
               labels: np.ndarray,
               model: WineNet,
               optimizer: Any,
               loss_fct: Any,
               num_epochs: int = 100,
               track_metrics: bool = False,
               save: bool = False,
               checkpoint_dir: str = "../checkpoints/",
               verbosity: bool = False) -> Tuple[WineNet, np.ndarray, np.ndarray, Union[dict, None]]:

    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    if track_metrics:
        accuracy = np.zeros(num_epochs)
        # precision = np.zeros(num_epochs)
        precision = np.zeros((num_epochs, labels.shape[0]))
        recall = np.zeros((num_epochs, labels.shape[0]))
        f1 = np.zeros((num_epochs, labels.shape[0]))

    for epoch in range(num_epochs):
        if verbosity:
            print(f"### Starting epoch {epoch + 1} of {num_epochs} ###")
        running_train_loss = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            x = data[0].to(torch.float32)
            y = data[1].to(torch.float32)
            model.train()
            model_out = model(x)
            optimizer.zero_grad()
            loss = loss_fct(model_out, y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        running_val_loss = 0
        if track_metrics:
            running_acc = 0
            running_pre = 0
            running_rec = 0
            running_f1 = 0
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            x = data[0].to(torch.float32)
            y = data[1].to(torch.float32)
            int_y = np.array(data[2])
            model.eval()
            with torch.no_grad():
                model_out = model(x)
                loss = loss_fct(model_out, y)
                running_val_loss += loss.item()

                if track_metrics:
                    preds = np.array(torch.argmax(model_out, dim=1).detach())

                    acc, pre, rec, f = evaluate_class_predictions(prediction=preds, ground_truth=int_y,
                                                                  labels=labels, verbosity=False)
                    running_acc += acc
                    running_pre += pre
                    running_rec += rec
                    running_f1 += f

        train_losses[epoch] = running_train_loss / len(train_loader)
        val_losses[epoch] = running_val_loss / len(val_loader)
        if track_metrics:
            accuracy[epoch] = running_acc / len(val_loader)
            precision[epoch, :] = running_pre / len(val_loader)
            recall[epoch, :] = running_rec / len(val_loader)
            f1[epoch, :] = running_f1 / len(val_loader)

        if verbosity:
            print(f"### Current train loss is {train_losses[epoch]} ###")
            print(f"### Current validation loss is {val_losses[epoch]} ###")

    if save:
        model_state = {'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'epoch': num_epochs}
        save_ckp(state_dict=model_state, checkpoint_dir=checkpoint_dir)

    if track_metrics:
        metrics_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    else:
        metrics_dict = None

    return model, train_losses, val_losses, metrics_dict


def run_training(n_epochs: int = 20,
                 test: bool = False,
                 plot_losses: bool = False,
                 save_losses: bool = False,
                 plot_save_metrics: bool = False):

    train_loader, val_loader, labels = get_data_loaders_wine_data(colour='red',
                                                                  val_and_test=False,
                                                                  scaling='min_max_norm',
                                                                  over_sample='random',
                                                                  batch_size=10,
                                                                  label_column='label',
                                                                  drop_column=['quality'],
                                                                  shuffle=True)
    num_labels = labels.shape[0]
    # Initialize model with the correct shape.
    model = WineNet(in_size=11, out_size=num_labels)
    # Initialize optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
    # Calculate class weights
    # class_weights = None
    class_weights = calc_class_weights(train_loader=train_loader, verbosity=True)
    # Initialize loss function
    loss_fct = nn.CrossEntropyLoss(weight=class_weights, reduction='mean', label_smoothing=0)
    # loss_fct = FocalLoss(include_background=True, gamma=2.0, weight=class_weights, reduction='mean')

    trained_model, train_losses, val_losses, metrics_dict = train_loop(train_loader=train_loader,
                                                                       val_loader=val_loader,
                                                                       labels=labels,
                                                                       model=model,
                                                                       optimizer=optimizer,
                                                                       loss_fct=loss_fct,
                                                                       num_epochs=n_epochs,
                                                                       track_metrics=plot_save_metrics,
                                                                       save=True,
                                                                       verbosity=True)
    if save_losses:
        np.save('train_losses.npy', train_losses)
        np.save('val_losses.npy', val_losses)
    if plot_losses:
        plot_loss(train_loss=train_losses, val_loss=val_losses)

    if plot_save_metrics:
        with open('../checkpoints/metrics_dict.pickle', 'wb') as f:
            pickle.dump(metrics_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        plot_metrics(metrics_dict=metrics_dict, single_plots=False)

    if test:
        val_gt_labels = np.array([])
        predictions = np.array([])
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            x = data[0].to(torch.float32)
            # y = data[1].to(torch.float32)
            int_y = np.array(data[2])
            model.eval()
            with torch.no_grad():
                model_out = trained_model(x)
                pred = np.array(torch.argmax(model_out, dim=1).detach())
                predictions = np.hstack((predictions, pred))
                val_gt_labels = np.hstack((val_gt_labels, int_y))
        evaluate_class_predictions(prediction=predictions, ground_truth=val_gt_labels, labels=labels, verbosity=True)


# Validation functionalities ###########################################################################################
def load_test_model(model_name: str, checkpoint_dir: str = "../checkpoints/") -> None:
    """
    Function for loading a saved model and testing it on a validation set.
    :param model_name: Filename of the model.
    :param checkpoint_dir: Directory of the saved model.
    :return: None
    """
    _, val_loader, labels = get_data_loaders_wine_data(val_and_test=False, batch_size=1, shuffle=True)
    model = WineNet(in_size=11, out_size=labels.shape[0])
    model, _, _ = load_ckp(ckp_name=model_name, model=model, checkpoint_dir=checkpoint_dir)

    val_gt_labels = np.array([])
    predictions = np.array([])
    for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        x = data[0].to(torch.float32)
        # y = data[1].to(torch.float32)
        int_y = np.array(data[2])
        model.eval()
        with torch.no_grad():
            model_out = model(x)
            pred = np.array(torch.argmax(model_out, dim=1).detach())
            predictions = np.hstack((predictions, pred))
            val_gt_labels = np.hstack((val_gt_labels, int_y))
    evaluate_class_predictions(prediction=predictions, ground_truth=val_gt_labels, labels=labels, verbosity=True)
    # train on all data, validate on red/white


if __name__ == '__main__':
    run_training(n_epochs=600, test=True, plot_losses=True, save_losses=False, plot_save_metrics=True)

    print('done')

