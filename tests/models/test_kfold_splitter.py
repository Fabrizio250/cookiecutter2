"""
Test for the k-fold split procedure
"""
import pandas as pd
import pytest
from src.models.data_splitter import KFoldSplitter
import os

dir = os.path.dirname(__file__)

processed_data_path = os.path.join(dir, "..", "..", "data", "processed", "data")

@pytest.mark.require_data
@pytest.fixture()
def input_data():
    path = os.path.join(processed_data_path, "index.csv")
    df = pd.read_csv(path)
    df.loc[0:150, 'Label'] = 1
    df.loc[150:300, 'Label'] = 0
    df.loc[300:380:2, 'Label'] = 1
    df.loc[301:380:2, 'Label'] = 0
    df = df.iloc[0:380].copy()
    return df


@pytest.mark.require_data
@pytest.fixture()
def splitter(input_data):
    return KFoldSplitter(input_data, n_folds=5)


@pytest.mark.require_data
@pytest.mark.usefixtures('splitter')
@pytest.mark.usefixtures('input_data')
@pytest.mark.parametrize('split_n', range(5))
def test_splits_size(splitter,input_data, split_n):
    train, val, test = splitter.get_split(split_n)
    assert len(train) == pytest.approx(4*len(input_data)/5, rel=0.2)
    assert len(test) == pytest.approx(1 * len(input_data) / 5, rel=0.2)
    assert len(val) == pytest.approx(0.1 * len(train), rel=0.2)



@pytest.mark.require_data
@pytest.mark.usefixtures('splitter')
@pytest.mark.usefixtures('input_data')
@pytest.mark.parametrize('split_n', range(5))
def test_sum_entries(splitter, input_data, split_n):
    train, valid, test = splitter.get_split(split_n)
    assert len(input_data) == len(train) + len(test) + len(valid)


@pytest.mark.require_data
@pytest.mark.usefixtures('splitter')
@pytest.mark.parametrize('split_n', range(5))
def test_leakage(splitter, split_n):
    train, val, test = splitter.get_split(split_n)
    assert not set(train.Subject).intersection(set(val.Subject))
    assert not set(train.Subject).intersection(set(test.Subject))
    assert not set(test.Subject).intersection(set(val.Subject))
