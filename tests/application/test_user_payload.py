"""Test methods for datamodels"""

import pytest
from fastapi import UploadFile
import src.app.backend.user_payload as up

BS_RAW_TYPE = 'application/gzip'
BS_MASK_TYPE = 'application/vnd.proteus.magazine'
JUNK_TYPE = 'text/json'


def test_correct_preprocessed_brainscan():
    """Test the correct preprocessed brainscan"""
    brscan = UploadFile(filename="brainscan.mgz", content_type=BS_MASK_TYPE)
    payload = up.UserPayload(brainscan=brscan)
    assert True


def test_correct_raw_brainscan():
    """Test the correct raw brainscan and brainmask"""
    brscan = UploadFile(filename="brainscan.nii.gz", content_type=BS_RAW_TYPE)
    brmask = UploadFile(filename="brainmask.mgz", content_type=BS_MASK_TYPE)
    payload = up.UserPayload(brainscan=brscan, brainmask=brmask)
    assert True


@pytest.mark.parametrize("scanname, scantype",
                         [
                             ("brainscan.mgz", JUNK_TYPE),
                             ("brainscan.aaa", BS_MASK_TYPE),
                             ("brainscan.nii.gz", BS_RAW_TYPE)
                         ])
def test_wrong_brainscan(scanname, scantype):
    """Test the wrong MIME-type of the preprocessed brainscan"""
    brscan = UploadFile(filename=scanname, content_type=scantype)
    with pytest.raises(ValueError):
        up.UserPayload(brainscan=brscan)


@pytest.mark.parametrize("scanname, maskname, scantype, masktype",
                         [
                             ("brainscan.nii.gz", "brainmask.mgz", JUNK_TYPE, BS_MASK_TYPE),
                             ("brainscan.ii", "brainmask.mgz", BS_RAW_TYPE, BS_MASK_TYPE),
                             ("brainscan.nii.gz", "brainmask.abc", BS_RAW_TYPE, BS_MASK_TYPE),
                             ("brainscan.ii", "brainmask.mgz", BS_RAW_TYPE, JUNK_TYPE)
                         ])
def test_wrong_cross_dependency(scanname, maskname, scantype, masktype):
    """Test the wrong MIME-type of the raw brainscan"""
    brscan = UploadFile(filename=scanname, content_type=scantype)
    brmask = UploadFile(filename=maskname, content_type=masktype)

    with pytest.raises(ValueError):
        up.UserPayload(brainscan=brscan, brainmask=brmask)


@pytest.mark.parametrize("n_iter", [0, 200])
def test_invalid_iter_cap(n_iter):
    """Test the low rannge of the number of iterations"""
    brscan = UploadFile(filename="brainscan.nii.gz", content_type=BS_RAW_TYPE)
    brmask = UploadFile(filename="brainmask.mgz", content_type=BS_MASK_TYPE)
    with pytest.raises(ValueError):
        up.UserPayload(brainscan=brscan, brainmask=brmask, n_iter=n_iter)
