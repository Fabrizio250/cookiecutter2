"""
Classes for performing dataset splitting
"""
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


class BaseDataSplitter(ABC):
    """
    Base class for data splitter, all the splitter must subclass this
    This class is an abstract class that defines the required fields for a dataframe
    and the abstractmethod get_split
    """
    _required_fields = ["Label", "Subject"]

    def __init__(self, source_df: DataFrame):
        """
        Instantiate a new splitter
        Args:
            source_df: dataframe with label, subject and sessions
        """
        if len(source_df) == 0:
            raise ValueError("Passed an empty DataFrame")

        if not set(self._required_fields).issubset(source_df.columns):
            raise ValueError(f"Source dataframe is not valid,"
                             f" must have {self._required_fields} fields")
        self.source_df = source_df

    @abstractmethod
    def get_split(self, split_n: int) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        This method returns the three dataframes train, validation and tests respectively
        Args:
            split_n: number of the required split
        Returns:
            Tuple[DataFrame, DataFrame, DataFrame]: train, validation and test data frames
        """

    def get_merged_split(self, split_n) -> DataFrame:
        """
        Convenience method for getting the dataset as single DataFrame
        in which the belonging set (train, val, test) is encoded into
        the column 'Set'
        Args:
            split_n: number of the required split

        Returns: a single DataFrame with 'Set' column whose values
         can be one of: Train, Validation, Test

        """
        train, val, test = self.get_split(split_n)
        train["Set"] = "Train"
        val["Set"] = "Validation"
        test["Set"] = "Test"
        concatenated = pd.concat([train, val, test]).reset_index(drop=True).copy()
        return concatenated


class HoldoutSplitter(BaseDataSplitter):
    """
        HoldoutSplitter for data splitter

        Args:
            Dataframe: dataframe to split
        Returns:
            Tuple[DataFrame, DataFrame, DataFrame]: The return value.
    """

    def __init__(
            self,
            source_df: pd.DataFrame,
            train_split: float = 0.8,
            validation_split: float = 0.2,
            test_split: float = None,
            random_state: int = 42,
            shuffle: bool = True,
    ):
        """
        Initialize a simple Holdout splitter

        Params:
            source_df (Dataframe): Source data frame, usually is an index but must contains
            train_split (float): fraction of source_df to be used as train set
            validation_split (float):  fraction of the train set to be used as validation
            test_split(float): fraction of source_df to be used for testing if
            is None is inferred from 'train_split'
            random_state (int): random seed for deterministic split
            shuffle (bool): if 'source_df' must be shuffled before split
        """
        super().__init__(source_df)
        if train_split is None and test_split is None:
            raise ValueError("At least one of train split"
                             " or test split must be provided")
        if train_split <= 0 or train_split >= 1:
            raise ValueError("Train split must be a value into (0;1)")
        if train_split is not None and test_split is not None:
            if train_split + test_split > 1:
                raise ValueError("Train split and test split must sum to 1")
        if validation_split == 0:
            raise ValueError("Validation split must be greater than 0")

        self.test_frac = test_split or 1 - train_split
        self.train_frac = train_split or 1 - self.test_frac
        self.validation_frac = validation_split

        self.shuffle = shuffle
        self.random_state = random_state

    def get_split(self, split_n: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into train, validation and test sets.

        Args:
            split_n: parameter is ignored kept for compatibility

        Returns: A tuple of pandas DataFrames corresponding to train, validation and test.
        """
        new_dataframe = self.source_df.copy()
        if self.shuffle:
            new_dataframe.sample(frac=1,
                                 random_state=self.random_state,
                                 ignore_index=True).copy()

        strata = new_dataframe.groupby("Label").count().MR_ID
        strata = strata / np.sum(strata)

        train_set, test_set = self._make_stratification(new_dataframe, self.train_frac, strata)
        train_set, validation_set = self._make_stratification(train_set, 1 - self.validation_frac, strata)

        return train_set.sort_values("Subject").reset_index(drop=True), \
               validation_set.sort_values("Subject").reset_index(drop=True), \
               test_set.sort_values("Subject").reset_index(drop=True)

    def _make_stratification(self, set_to_split, split_frac, strata):
        pos_subjects = set(set_to_split[set_to_split.Label == 1].Subject)
        pos_df = set_to_split[set_to_split.Subject.isin(pos_subjects)]
        neg_df = set_to_split[~set_to_split.Subject.isin(pos_subjects)]

        train_neg, test_neg = self._make_split(pos_df, split_frac * strata[0]*2)
        train_pos, test_pos = self._make_split(neg_df, split_frac * strata[1]*2)

        return pd.concat([train_pos, train_neg], ignore_index=True), \
               pd.concat([test_pos, test_neg], ignore_index=True)

    def _make_split(self, set_to_split, split_frac):
        train_set = pd.DataFrame()
        test_set = pd.DataFrame()
        subject_list = set_to_split.Subject.unique()
        last_subject = 0
        train_elements = split_frac * len(set_to_split)
        groups = set_to_split.groupby("Subject")
        for i, subject in enumerate(subject_list):
            g_set = groups.get_group(subject)
            train_elements -= len(g_set)
            train_set = pd.concat([train_set, g_set])
            if train_elements <= 0:
                last_subject = i
                break

        for i, subject in enumerate(subject_list):
            if i > last_subject:
                g_set = groups.get_group(subject)
                test_set = pd.concat([test_set, g_set])
        return train_set, test_set


class KFoldSplitter(BaseDataSplitter):
    """
    Class for the computation of the train, validation, tests split using KFold, the validation set
    is extracted from the train set using a simple shuffle split this class ensure that
    there are no subjects into different split (in the same kfold split)
    """

    def __init__(self, source_df: pd.DataFrame,
                 n_folds: int = 10,
                 validation_split: float = 0.1,
                 random_state: Any = 42,
                 shuffle: bool = False):
        """
        Instantiate a KFold data splitter

        Args:
            source_df (pd.DataFrame): source dataframe usually is the index.csv file
             in the data folder
            n_folds (int): number of splits for the kfold CV
            validation_split (float): fraction of training split to devote to the validation
            random_state (Any): number to allow reproducibility, if None the behaviour can change
            shuffle (bool): if true the KFold split shuffles the data
        """
        super().__init__(source_df)
        self.sss = StratifiedShuffleSplit(n_splits=1,
                                          test_size=validation_split,
                                          random_state=random_state)
        self.sgkf = StratifiedGroupKFold(n_splits=n_folds,
                                         shuffle=shuffle,
                                         random_state=random_state if shuffle else None)
        self.random_state = random_state

    def get_split(self, split_n) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Get the specified split
        :param split_n: split number, must be less than the number of folds
        :return: the three different datasets, in order: train, validation, tests
        """
        assert split_n < self.sgkf.get_n_splits(), "Not enough splits"
        row_indexes, targets, groups = \
            self.source_df.index, self.source_df.Label, self.source_df.Subject
        train_ds = None
        test_ds = None
        for i, (train, test) in enumerate(self.sgkf.split(row_indexes, targets, groups)):
            if i == split_n:
                train_ds = self.source_df.iloc[train].reset_index(drop=True)
                test_ds = self.source_df.iloc[test].reset_index(drop=True)

        row_indexes, targets = train_ds.index, train_ds.Label
        train, val = next(self.sss.split(row_indexes, targets))
        val_ds = train_ds.iloc[val]
        train_ds = train_ds.iloc[train]

        conflicts = set(train_ds.Subject).intersection(set(val_ds.Subject))
        train_ds, val_ds = self._handle_conflicts(conflicts, train_ds, val_ds)

        return train_ds.sort_values("Subject").reset_index(drop=True), \
               val_ds.sort_values("Subject").reset_index(drop=True), \
               test_ds.sort_values("Subject").reset_index(drop=True)

    def _handle_conflicts(self, conflicts, train_set, validation_set):
        train_ds = train_set.copy()
        val_ds = validation_set.copy()

        while len(conflicts) > 0:
            pos_ballot_box = train_ds[~train_ds.Subject.isin(val_ds.Subject)]
            neg_ballot_box = pos_ballot_box[pos_ballot_box.Label == 0]
            pos_ballot_box = pos_ballot_box[pos_ballot_box.Label == 1]
            for conflict_sub in conflicts:
                subject = val_ds[val_ds.Subject == conflict_sub]
                for idx, row in subject.iterrows():
                    val_ds = val_ds.drop(index=idx)
                    if row.Label == 0:
                        draws = neg_ballot_box.sample(n=1, random_state=self.random_state)
                        neg_ballot_box = neg_ballot_box.drop(draws.index)
                    else:
                        draws = pos_ballot_box.sample(n=1, random_state=self.random_state)
                        pos_ballot_box = pos_ballot_box.drop(draws.index)
                    train_ds = train_ds.drop(draws.index)
                    val_ds = pd.concat([val_ds, draws])
                train_ds = pd.concat([train_ds, subject])

            conflicts = set(train_ds.Subject).intersection(set(val_ds.Subject))

        return train_ds, val_ds
