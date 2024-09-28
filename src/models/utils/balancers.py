"""
This module provides some data sampler that
can be used for data downsampling/upsampling
"""
from abc import ABC, abstractmethod

import pandas as pd
from pandas import DataFrame


class BaseSampler(ABC):
    """ Base class for data sampler"""

    def __init__(self, source_df: DataFrame):
        self.source_df = source_df

    @abstractmethod
    def get_sample(self) -> DataFrame:
        """
        Perform the resample
        Returns: a sampled dataframe
        """
    def get_classes_stat(self) -> pd.DataFrame:
        """
        Compute statistics based on the class labels
        Returns: the dataframe with statistics
        """
        return self.source_df.groupby('Label').describe()

    def __str__(self):
        return str(self.__dict__)

    def __len__(self):
        return len(self.source_df)


class IdentitySampler(BaseSampler):
    """
    This class does not perform any resampling
    """

    def get_sample(self) -> DataFrame:
        """
        Get the same sample provided at the initialization
        Returns:

        """
        return self.source_df



class RandomDownsampler(BaseSampler):
    """
    Downsample the majority class by means of a random sampler
    """

    def __init__(self, source_df: DataFrame, random_state: float = 42, ratio=1):
        super().__init__(source_df)
        self.random_state = random_state
        self.ratio = ratio

    def get_sample(self)-> DataFrame:
        classes = self.source_df.groupby("Label").count().MR_ID
        minority_class = classes.idxmin()
        minority_count = classes.min()
        temp = self.source_df[self.source_df.Label != minority_class].copy()
        temp = temp.sample(n=int(minority_count * self.ratio), random_state=self.random_state)
        temp = pd.concat([self.source_df[self.source_df.Label == minority_class], temp]).reset_index(drop=True).copy()
        return temp
