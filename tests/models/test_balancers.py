import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import os

from src.models.utils.balancers import RandomDownsampler, IdentitySampler

dir = os.path.dirname(__file__)

processed_data_path = os.path.join(dir, "..", "..", "data", "processed", "data")

@pytest.fixture()
@pytest.mark.require_data
def sample_data():
    path = os.path.join(processed_data_path, "index.csv")
    index = pd.read_csv(path)
    index.loc[0:20, 'Label'] = 1
    index.loc[20:, 'Label'] = 0
    return index.copy()


@pytest.mark.require_data
def test_random_downsampler_deterministic(sample_data):
    sampler1 = RandomDownsampler(sample_data, random_state=42)
    sampler2 = RandomDownsampler(sample_data, random_state=42)
    assert set(sampler1.get_sample()) == set(sampler2.get_sample())  # use sets for ignore ordering


@pytest.mark.require_data
@pytest.mark.parametrize('ratio,expected', [(0.5, 10), (1, 20), (2, 40)])
def test_random_downsampler_count(sample_data, ratio, expected):
    sampler = RandomDownsampler(sample_data, random_state=1, ratio=ratio)
    samples = sampler.get_sample()

    assert len(samples[samples.Label == 0]) == expected


@pytest.mark.require_data
def test_identity_sampler(sample_data):
    sampler = IdentitySampler(sample_data)
    assert_frame_equal(sampler.get_sample(), sample_data)
