
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from load_data import *


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


# Todo

if __name__ == '__main__':
    train_loader, _ = get_data_loaders_wine_data(val_and_test=False, batch_size=20, shuffle=True)
    for item in train_loader:
        print(item[0])
        print(item[1])

