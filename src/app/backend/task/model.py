"""
This module provides the MRI classifier model
"""
from typing import Tuple, Any

import captum.attr
import numpy as np
from captum.attr import IntegratedGradients
import pytorch_lightning as pl

import torch
from torch import nn


class MRIClassifier(pl.LightningModule):
    """
    MRI classifier model, more info provided in the Model Card
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.InstanceNorm3d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.InstanceNorm3d(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.InstanceNorm3d(128),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.InstanceNorm3d(256),
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        self.classifier = nn.Sigmoid()

    def forward(self, x, return_features=False) -> tuple[Any, Any] | Any:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        features = self.conv4(x)
        x = self.dense(features)
        x = self.classifier(x)
        if return_features:
            return x, features
        return x

    def predict(self, sample):
        """
        Make a prediction, return a value between 0 and 1
        """
        if isinstance(sample, np.ndarray):
            x = torch.tensor(np.expand_dims(sample, axis=0))
        else:
            x = sample
        x = torch.unsqueeze(x, 0)
        y_hat, features = self.forward(x.float(), True)
        return y_hat.item(), torch.flatten(features, start_dim=0, end_dim=-1)

    def get_attribution(self, sample,
                        attribution_method: captum.attr = IntegratedGradients,
                        n_steps=1):
        """
        Compute the gradient attribution using the selected algorithm
        """
        if isinstance(sample, np.ndarray):
            x = torch.tensor(np.expand_dims(sample, axis=0))
        else:
            x = sample
        x = torch.unsqueeze(x, 0)
        attr_algo = attribution_method(self.forward)
        attr = attr_algo.attribute(x.float(), n_steps=n_steps, internal_batch_size=1)
        return attr.reshape((128, 128, 60))
