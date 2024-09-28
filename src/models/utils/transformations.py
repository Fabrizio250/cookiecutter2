"""
Kornia Augmentations and transformations, these can take advantage from the use of a GPU
"""
import torch
from kornia.augmentation import RandomRotation3D
from torch import nn, Tensor


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors.
     The augmentation consists in random rotation"""

    def __init__(self, degrees=(10, 10, 10), p_mirror=0.5, p_rot=0.8) -> None:
        super().__init__()
        self.p_mirror = p_mirror
        self.p_rot = p_rot
        self.angle = degrees
        self.transforms = nn.Sequential(
            RandomRotation3D(self.angle, p=p_rot)
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass implementation,
         grad is disabled since not needed"""
        x_out = self.transforms(x)  # BxCxHxW

        return x_out
