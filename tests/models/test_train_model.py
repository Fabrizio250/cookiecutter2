"""
Tests for train_model.py support functions
"""
from unittest import mock

import pytest

from src.models import train_model


def mock_hexacore_processor():
    """ Fake 6 core CPU """
    return 6


def mock_no_gpu():
    """ Mock the absence of the GPU """
    return False


def mock_gpu():
    """ Mock the presence of a GPU"""
    return True


@mock.patch('torch.cuda.is_available', mock_no_gpu)
def test_auto_accelerator_no_gpu():
    assert train_model.auto_accelerator() == 'cpu'


@mock.patch('torch.cuda.is_available', mock_gpu)
def test_auto_accelerator_gpu_available():
    assert train_model.auto_accelerator() == 'gpu'


@mock.patch('torch.cuda.is_available', mock_no_gpu)
@pytest.mark.parametrize('precision,expected', [(32, 32), (16, 32)])
def test_precision_no_gpu(precision, expected):
    assert train_model.check_precision('cpu', precision) == expected


@mock.patch('torch.cuda.is_available', mock_gpu)
@pytest.mark.parametrize('accelerator, precision,expected', [('cpu', 16, 32), ('gpu', 16, 16)])
def test_precision_no_gpu(accelerator, precision, expected):
    assert train_model.check_precision(accelerator, precision) == expected


@mock.patch('torch.cuda.is_available', mock_no_gpu)
@mock.patch('multiprocessing.cpu_count', mock_hexacore_processor)
@pytest.mark.parametrize('accelerator', ['cpu', 'gpu'])
def test_auto_workers_no_gpu(accelerator):
    assert train_model.auto_workers(accelerator) == 1


@mock.patch('torch.cuda.is_available', mock_gpu)
@mock.patch('multiprocessing.cpu_count', mock_hexacore_processor)
@pytest.mark.parametrize('accelerator,expected_workers', [('cpu', 1), ('gpu', 6)])
def test_auto_workers_no_gpu(accelerator, expected_workers):
    assert train_model.auto_workers(accelerator) == expected_workers
