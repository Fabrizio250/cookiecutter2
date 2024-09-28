import pytest
import os

from src.models.data_labeler import LabelClinicalData

dir = os.path.dirname(__file__)

raw_data_path = os.path.join(dir, "..", "..", "data", "raw")

@pytest.mark.require_data
@pytest.fixture(scope='module')
def labeler():
    path = os.path.join(raw_data_path, "clinical-map.json")
    return LabelClinicalData(path)


@pytest.mark.require_data
def test_normalize_subject_one(labeler):
    diagnosis_true = [True]
    diagnosis_false = [False]
    diagnosis_empty = []
    assert all(labeler._normalize_subject(diagnosis=diagnosis_true))
    assert not all(labeler._normalize_subject(diagnosis=diagnosis_false))

    with pytest.raises(ValueError):
        labeler._normalize_subject(diagnosis=diagnosis_empty)


@pytest.mark.require_data
def test_normalize_subject_many(labeler):
    diagnosis_true = [True] * 30
    diagnosis_false = [False] * 30
    assert all(labeler._normalize_subject(diagnosis=diagnosis_true))
    assert not all(labeler._normalize_subject(diagnosis=diagnosis_false))


@pytest.mark.require_data
@pytest.mark.parametrize("test_input, expected",
                         [
                             ([True, False, False], [False, False, False]),
                             ([True, False, False, True, True], [True, True, True, True, True]),
                             ([True, True, False, False], [False, False, False, False])
                         ])
def test_normalization(labeler, test_input, expected):
    assert labeler._normalize_subject(test_input) == expected
