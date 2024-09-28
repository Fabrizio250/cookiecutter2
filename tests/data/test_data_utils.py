import numpy as np
import pytest

from src.data.utils import MRIProcessor, LocalMinMaxScaler


def test_shape_match():
    output_shape = (60, 128, 128)
    processor = MRIProcessor(output_size=output_shape)
    dummy_mask = np.ones((128, 256, 256))
    synth = np.random.random((128, 256, 256))
    assert processor._mask_mr(synth, dummy_mask).shape == output_shape


@pytest.mark.parametrize('max_value', [1000, 2000, 3000, 4000, 5000])
def test_scaler(max_value):
    scaler = LocalMinMaxScaler()
    synth = np.random.rand(10, 60, 128, 128)
    synth *= max_value
    result, _, _ = scaler(synth)
    assert result.min() >= 0
    assert result.max() <= 1
