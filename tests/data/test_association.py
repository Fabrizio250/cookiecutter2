"""
Tests for the association methods provided by src/data/clinical.py
"""
import pandas as pd
import pytest
import os

from src.data.clinical import make_association

dir = os.path.dirname(__file__)

raw_data_path = os.path.join(dir, "..", "..", "data", "raw")

@pytest.mark.require_data
@pytest.fixture()
def clinical_data():
    path = os.path.join(raw_data_path, "clinical-data.csv")
    return pd.read_csv(path)

@pytest.mark.require_data
@pytest.fixture()
def mr_sessions():
    path = os.path.join(raw_data_path, "mri-scans.csv")
    return pd.read_csv(path)

@pytest.mark.require_data
def test_forward(clinical_data, mr_sessions):
    associated = make_association(clinical_data,
                                mr_sessions,
                                direction='forward',
                                return_complete=False)
    check_list = associated.DaysAfterEntry_mr <= associated.DaysAfterEntry_clinic
    assert all(check_list)


@pytest.mark.require_data
def test_backward(clinical_data, mr_sessions):
    associated = make_association(clinical_data,
                                      mr_sessions,
                                      direction='backward',
                                      return_complete=False)
    check_list = associated.DaysAfterEntry_mr >= associated.DaysAfterEntry_clinic
    assert all(check_list)  # add assertion here

@pytest.mark.require_data
def test_return_complete(clinical_data, mr_sessions):
    associated = make_association(clinical_data,
                                      mr_sessions,
                                      direction='nearest',
                                      return_complete=True)
    assert len(clinical_data) == len(associated), "Number of associated entries should be equal to MR sessions"

@pytest.mark.require_data
def test_wrong_structure(clinical_data, mr_sessions):
    with pytest.raises(AssertionError):
        make_association(clinical_data.drop(columns=["ADRC_ADRCCLINICALDATA_ID"]), mr_sessions)

    with pytest.raises(AssertionError):
        make_association(clinical_data, mr_sessions.drop(columns=["MR_ID"]))

