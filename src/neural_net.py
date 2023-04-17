
import torch
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


def load_wine_data(val_and_test: bool = False) -> Tuple[DataLoader, ...]:
    if val_and_test:
        traindf, valdf, testdf = data_pipeline_redwine(val_and_test=val_and_test)
        traindf = traindf.drop(columns=['quality'])
        valdf = testdf.drop(columns=['quality'])
        testdf = testdf.drop(columns=['quality'])

        traind = WineDataset(traindf)
        vald = WineDataset(valdf)
        testd = WineDataset(testdf)

        train_loader = DataLoader(traind)
        val_loader = DataLoader(vald)
        test_loader = DataLoader(testd)

        return train_loader, val_loader, test_loader

    else:
        traindf, testdf = data_pipeline_redwine(val_and_test=val_and_test)
        traindf = traindf.drop(columns=['quality'])
        testdf = testdf.drop(columns=['quality'])

        traind = WineDataset(traindf)
        testd = WineDataset(testdf)

        train_loader = DataLoader(traind)
        test_loader = DataLoader(testd)

        return train_loader, test_loader
# Todo
