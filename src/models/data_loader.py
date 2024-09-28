""" This module provides classes for handling the processed data the data"""

from typing import Any

from pandas import DataFrame
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.models.data_splitter import BaseDataSplitter
from src.models.utils.transformations import DataAugmentation


class T1wDataset(Dataset):
    """
    Class for loading the MR-OASIS3 dataset files
    """

    def __init__(self, data_index: DataFrame, folder: str, transform=None):
        """
        Initialize the dataset of T1w MR scans
        :param data_index: a pandas dataframe with at least two columns ['MR_ID', 'Label']
        where MR_ID is the filename of the scan and 'Label' is the ground truth value
        :param folder: folder where the scans are
        :param transform: optional data transformation such as normalize(mean, std)
        """
        self.index = data_index
        self.folder = folder
        self.transform = transform

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        data = torch.FloatTensor(
            np.expand_dims(np.load(self.folder + row.MR_ID + '.npz',
                                   allow_pickle=True)["x"],
                           axis=0))
        label = torch.tensor(row.Label, dtype=torch.float)
        if self.transform:
            label = self.transform(data)
        return data, label


class MROasis3Datamodule(LightningDataModule):
    """ Lightning datamodule for the OASIS3 MR dataset"""

    def __init__(self, data_folder: str,
                 data_splitter: BaseDataSplitter,
                 batch_size: int,
                 num_workers: int = 1):
        """
        Initialize the datamodule with the given splitter
        Args:
            data_folder: folder where mr sessions are stored
            data_splitter: a BaseDataSplitter instance
            batch_size: size of the mini-batch
            num_workers: number of cpu workers to use for data loading
        """
        super().__init__()
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.splitter = data_splitter
        self.train = None
        self.val = None
        self.test = None
        self.num_workers = num_workers
        self.augment = DataAugmentation()

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        data, label = batch
        if self.trainer.training:
            data = self.augment(data)  # => we perform GPU/Batched data augmentation
        return data, label

    def prepare_data(self) -> None:
        self.train, self.val, self.test = self.splitter.get_split(0)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train = T1wDataset(self.train, self.data_folder)
        return DataLoader(train, self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        val = T1wDataset(self.val, self.data_folder)
        return DataLoader(val, self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test = T1wDataset(self.test, self.data_folder)
        return DataLoader(test, self.batch_size, num_workers=self.num_workers, shuffle=False)
