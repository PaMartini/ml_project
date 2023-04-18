import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from tqdm import tqdm
from datetime import datetime
import os
import pickle

from load_data import *
from evaluation import *


class WineDataset(Dataset):
    def __init__(self, wine_df: pd.DataFrame):
        self.df = wine_df

    def __getitem__(self, index):
        row = self.df.iloc[index].values
        features = row[:-1]
        label = row[-1]
        return features, label

    def __len__(self):
        return len(self.df)


def get_data_loaders_wine_data(val_and_test: bool = False,
                               batch_size: int = 1,
                               shuffle: bool = False) -> Tuple[DataLoader, ...]:
    if val_and_test:
        traindf, valdf, testdf = data_pipeline_redwine(val_and_test=val_and_test)
        traindf = traindf.drop(columns=['quality'])
        valdf = testdf.drop(columns=['quality'])
        testdf = testdf.drop(columns=['quality'])

        traind = WineDataset(traindf)
        vald = WineDataset(valdf)
        testd = WineDataset(testdf)

        train_loader = DataLoader(traind, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(vald, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(testd, batch_size=batch_size, shuffle=shuffle)

        return train_loader, val_loader, test_loader

    else:
        traindf, testdf = data_pipeline_redwine(val_and_test=val_and_test)
        traindf = traindf.drop(columns=['quality'])
        testdf = testdf.drop(columns=['quality'])

        traind = WineDataset(traindf)
        testd = WineDataset(testdf)

        train_loader = DataLoader(traind, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(testd, batch_size=batch_size, shuffle=shuffle)

        return train_loader, test_loader


class WineNet(torch.nn.Module):
    def __init__(self, in_size: int = 11, out_size: int = 1):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.Linear(in_features=self.in_size, out_features=2 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=2 * self.in_size, out_features=6 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=6 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=6 * self.in_size, out_features=2 * self.in_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=2 * self.in_size, out_features=self.out_size),
        )

        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = self.last_activation(x)
        return x


def save_ckp(model_state: dict, checkpoint_dir: str = "../checkpoints/") -> None:
    fp = os.path.join(checkpoint_dir, 'model_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.pt')
    torch.save(model_state, fp)
    print(f'Saved model.')


def load_ckp(ckp_name: str,
             model: Type[nn.Module],
             optimizer: Type[torch.optim.Optimizer] = None,
             checkpoint_dir: str = "../checkpoints/",
             ) -> Tuple[Any, Any, Any]:

    ckp = torch.load(os.path.join(checkpoint_dir, ckp_name))
    model.load_state_dict(ckp['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckp['optimizer'])
    return model, optimizer, ckp['epoch']


def train_loop(train_loader: DataLoader,
               val_loader: DataLoader,
               model: Type[torch.nn.Module],
               optimizer: Any,
               loss_fct: Any,
               num_epochs: int = 100,
               track_metrics: bool = False,
               save: bool = False,
               checkpoint_dir: str = "../checkpoints/",
               verbosity: bool = False) -> Tuple[Type[nn.Module], np.ndarray, np.ndarray, Union[dict, None]]:

    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    if track_metrics:
        accuracy = np.zeros(num_epochs)
        precision = np.zeros(num_epochs)
        recall = np.zeros(num_epochs)
        f1 = np.zeros(num_epochs)

    flatten = nn.Flatten(start_dim=0, end_dim=-1)

    for epoch in range(num_epochs):
        if verbosity:
            print(f"### Starting epoch {epoch + 1} of {num_epochs} ###")
        running_train_loss = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            x = data[0].to(torch.float32)
            y = data[1].to(torch.float32)
            model.train()
            model_out = model(x)
            model_out = flatten(model_out)
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
            model.eval()
            with torch.no_grad():
                model_out = model(x)
                model_out = flatten(model_out)
                loss = loss_fct(model_out, y)
                running_val_loss += loss.item()

                if track_metrics:
                    preds = np.array((model_out >= 0.5)).astype(float)
                    acc, pre, rec, f = evaluate_class_predictions(prediction=preds, ground_truth=y, verbosity=False)
                    running_acc += acc
                    running_pre += pre
                    running_rec += rec
                    running_f1 += f

        train_losses[epoch] = running_train_loss / len(train_loader)
        val_losses[epoch] = running_val_loss / len(val_loader)
        if track_metrics:
            accuracy[epoch] = running_acc / len(val_loader)
            precision[epoch] = running_pre / len(val_loader)
            recall[epoch] = running_rec / len(val_loader)
            f1[epoch] = running_f1 / len(val_loader)

        if verbosity:
            print(f"### Current train loss is {train_losses[epoch]} ###")
            print(f"### Current validation loss is {val_losses[epoch]} ###")

    if save:
        model_state = {'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'epoch': num_epochs}
        save_ckp(model_state=model_state, checkpoint_dir=checkpoint_dir)

    if track_metrics:
        metrics_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    else:
        metrics_dict = None

    return model, train_losses, val_losses, metrics_dict


def run_training(n_epochs: int = 20, plot_losses: bool = False, save_losses: bool = False, plot_save_metrics: bool = False):
    train_loader, val_loader = get_data_loaders_wine_data(val_and_test=False, batch_size=20, shuffle=True)
    model = WineNet(in_size=11, out_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fct = nn.BCELoss(reduction='mean')

    trained_model, train_losses, val_losses, metrics_dict = train_loop(train_loader=train_loader,
                                                                       val_loader=val_loader,
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
        with open('metrics_dict.pickle', 'wb') as f:
            pickle.dump(metrics_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        plot_metrics(metrics_dict=metrics_dict)


def plot_loss(train_loss: np.ndarray, val_loss: np.ndarray):
    plt.plot(np.arange(len(train_loss)), train_loss, label='Training loss')
    plt.plot(np.arange(len(val_loss)), val_loss, label='Validation loss')
    plt.yscale('log')
    plt.legend()
    # plt.savefig('losses_plot.png')
    plt.show()


def plot_metrics(metrics_dict: dict):
    for key in metrics_dict:
        plt.plot(np.arange(len(metrics_dict[key])), metrics_dict[key], label=key)
    plt.legend()
    plt.show()


def test_model(model_name: str, checkpoint_dir: str = "../checkpoints/"):
    model = WineNet(in_size=11, out_size=1)
    model, _, _ = load_ckp(ckp_name=model_name, model=model, checkpoint_dir=checkpoint_dir)
    _, val_loader = get_data_loaders_wine_data(val_and_test=False, batch_size=1, shuffle=False)

    flatten = nn.Flatten(start_dim=0, end_dim=-1)
    model_outs = np.zeros(len(val_loader))
    labels = np.zeros(len(val_loader))
    for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        x = data[0].to(torch.float32)
        y = data[1].to(torch.float32)
        labels[i] = float(flatten(y))
        model.eval()
        with torch.no_grad():
            model_out = model(x)
            model_outs[i] = float(flatten(model_out))

    preds = (model_outs >= 0.5).astype(float)
    evaluate_class_predictions(prediction=preds, ground_truth=labels)


def check_model():
    model = WineNet()
    summary(model, input_size=(11,))


if __name__ == '__main__':
    run_training(n_epochs=100, plot_losses=True, save_losses=False, plot_save_metrics=True)
    # test_model(model_name='model_20230418-090242.pt')
    print('done')

