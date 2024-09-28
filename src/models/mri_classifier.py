"""
This module provides the MRI classifier model
"""
# pylint: disable=arguments-differ
# pylint: disable=unused-argument

from typing import Union, List, Any

import captum.attr
import numpy as np
from captum.attr import IntegratedGradients
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import pytorch_lightning as pl

import torch
from torch import nn
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, Specificity, AUROC


class MRIClassifier(pl.LightningModule):
    """
    MRI classifier model, more info provided in the Model Card
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Setup model specific CLI arguments """
        parser = parent_parser.add_argument_group("MRIClassifier arguments")
        parser.add_argument("--learning_rate", type=float, default=5e-5)
        parser.add_argument("--weight_decay", type=float, default=0.96)
        parser.add_argument("--hidden_size", type=int, default=512)
        return parent_parser

    def __init__(self, learning_rate, weight_decay, batch_size, **kwargs):
        super().__init__()

        metric_collection = MetricCollection([
            Accuracy(),
            Precision(),
            Recall(),
            Specificity(),
            AUROC()
        ])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.save_hyperparameters()

        self.train_metrics = metric_collection.clone(prefix="train_")
        self.val_metrics = metric_collection.clone(prefix="val_")
        self.test_metrics = metric_collection.clone(prefix="test_")

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
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x) -> Any:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dense(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(-1)
        loss = self.criterion(y_hat, y)

        y = y.type(torch.int)
        self.train_metrics.update(self.classifier(y_hat), y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)

        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(-1)
        val_loss = self.criterion(y_hat, y)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False)
        y = y.type(torch.int)
        self.val_metrics.update(self.classifier(y_hat), y)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        dict_metric = self.val_metrics.compute()
        self.log_dict(dict_metric)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(-1)

        test_loss = self.criterion(y_hat, y)
        self.log("test_loss", test_loss)
        y = y.type(torch.int)
        self.test_metrics.update(self.classifier(y_hat), y)

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict(self, sample: Union[torch.Tensor, np.array]):
        """
        Make a prediction, return a value between 0 and 1
        """
        if isinstance(sample, np.ndarray):
            x = torch.tensor(np.expand_dims(sample, axis=0))
        else:
            x = sample
        x = torch.unsqueeze(x, 0)
        y_hat = self(x.float())
        y_hat = self.classifier(y_hat)
        return y_hat.item()

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
        attr_algo = attribution_method(self)
        attr = attr_algo.attribute(x.float(), n_steps=n_steps)
        return attr.reshape((128, 128, 60))
