import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from src.models.data_splitter import HoldoutSplitter
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
    return HoldoutSplitter(input_data)


@pytest.mark.require_data
def test_source_df_empty():
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError):
        HoldoutSplitter(source_df=df_empty)


@pytest.mark.require_data
@pytest.mark.usefixtures('input_data')
@pytest.mark.parametrize('train_split,valid_split,test_split',
                         [(1, 0.1, None), (0.8, 0.1, 0.8),
                          (0.8, 0, 0.2), (None, 0.1, None)])
def test_invalid_split_size(input_data, train_split, valid_split, test_split):
    with pytest.raises(ValueError):
        HoldoutSplitter(input_data,
                        train_split=train_split,
                        validation_split=valid_split,
                        test_split=test_split)


@pytest.mark.require_data
@pytest.mark.usefixtures('input_data')
@pytest.mark.usefixtures('splitter')
def test_splits_length(input_data, splitter):
    train, valid, test = splitter.get_split(0)
    assert len(train) == pytest.approx(len(input_data) * splitter.train_frac, rel=0.5), "Train len error"
    assert len(valid) == pytest.approx(len(input_data) * splitter.validation_frac, rel=0.5), "Validation len error"
    assert len(test) == pytest.approx(len(input_data) * splitter.test_frac, rel=0.5), "Test len error"


@pytest.mark.require_data
def test_sum_entries(input_data, splitter):
    train, valid, test = splitter.get_split(0)
    assert len(input_data) == len(train) + len(test) + len(valid)


@pytest.mark.require_data
@pytest.mark.usefixtures('splitter')
def test_leakage_holdout(splitter):
    train, valid, test = splitter.get_split(0)
    assert not set(train.Subject).intersection(set(valid.Subject))
    assert not set(train.Subject).intersection(set(test.Subject))
    assert not set(valid.Subject).intersection(set(test.Subject))


@pytest.mark.require_data
@pytest.mark.usefixtures('splitter')
@pytest.mark.usefixtures('input_data')
def test_out_same_types(splitter, input_data):
    splitted = splitter.get_merged_split(0)
    splitted.drop(columns=['Set'], inplace=True)
    assert all(input_data.dtypes == splitted.dtypes)


@pytest.mark.require_data
@pytest.mark.usefixtures('splitter')
@pytest.mark.usefixtures('input_data')
def test_stratification(splitter, input_data):
    original_ratio = input_data.groupby("Label").count().MR_ID
    original_ratio = original_ratio/np.sum(original_ratio)

    train, _, test = splitter.get_split(0)
    train_balance = train.groupby("Label").count().MR_ID
    train_balance = train_balance/np.sum(train_balance)

    test_balance = test.groupby("Label").count().MR_ID
    test_balance = test_balance/np.sum(test_balance)

    npt.assert_allclose(original_ratio, train_balance, atol=0.2)
    npt.assert_allclose(original_ratio, test_balance, atol=0.2)


