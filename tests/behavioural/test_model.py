"""
Test model functionalities according to:
Minimal functionality
Invariance to changes
"""
import numpy as np
import pytest
import torch
import os

from src.models.mri_classifier import MRIClassifier
from src.models.utils.transformations import DataAugmentation

dir = os.path.dirname(__file__)

raw_data_path = os.path.join(dir, "..", "..", "models")
processed_data_path = os.path.join(dir, "..", "..", "data", "processed", "data")

@pytest.mark.require_model
@pytest.fixture()
def model():
    path_to_model = os.path.join(raw_data_path, "model.ckpt")
    model = MRIClassifier.load_from_checkpoint(path_to_model)
    return model


@pytest.mark.require_model
@pytest.fixture()
def sample():
    positive_scan = os.path.join(processed_data_path, "OAS30271_MR_d0004.npz")
    data = torch.FloatTensor(
        np.expand_dims(np.load(positive_scan, allow_pickle=True)["x"], axis=0))
    data = torch.unsqueeze(data, 0) # this simulate a batch size of 1
    return data


@pytest.mark.require_model
@pytest.mark.usefixtures("model")
@pytest.mark.usefixtures("sample")
def test_minimum_functionality(model, sample):
    with torch.no_grad():
        prediction = model.classifier(model(sample))
    print(prediction)
    assert prediction > 0.5


@pytest.mark.require_model
@pytest.mark.usefixtures("model")
@pytest.mark.usefixtures("sample")
def test_invariance_to_rot(model, sample):
    with torch.no_grad():
        augmenter = DataAugmentation(p_rot=1, degrees=(2,2,2))
        rotated = augmenter(sample)
        pred_source = model.classifier(model(sample))
        pred_rot = model.classifier(model(rotated))
    assert pred_source == pytest.approx(pred_rot, abs=0.2)
