"""
This module provides several methods for processing MR scans
"""
from math import floor, ceil
from typing import Literal, Tuple, Union, Callable

import numpy as np

from scipy.ndimage import binary_closing

import nibabel as nib
from nibabel import processing


class LocalMinMaxScaler:
    """ Apply a MinMax scaling to the given array using
     the min and max computed on the same array"""

    def __init__(self, value_range: Tuple = (0, 1)):
        self.out_min = value_range[0]
        self.out_max = value_range[1]

    def __call__(self, array: np.array) -> np.array:
        min_v = array.min()
        max_v = array.max()
        return ((array - min_v) / (max_v - min_v)) * (self.out_max - self.out_min) + self.out_min

    def __str__(self):
        return str(self.__dict__)


class MRIProcessor:
    """
    This class is used for preprocessing on MR scans,
    for the given MR and mask perform a RoOI extraction and crop to size
    """
    # This suppresses warnings
    nib.imageglobals.logger.setLevel(50)

    def __init__(self,
                 voxel_size: Tuple = (1.0, 1.0, 1.0),
                 orientation: Literal["RAS", "LPS"] = 'RAS',
                 output_size: Tuple = (128, 128, 60),
                 scaler: Callable = LocalMinMaxScaler()):
        self.voxel_size = voxel_size
        self.orientation = orientation
        self.output_size = np.array(output_size)
        self.scaler = scaler

    def __call__(self, mr_path: str, brainmask_path: str = None) -> Union[np.array, float, float]:
        """
        Process MRI from mr_path with corresponding brainmask, the resulting MR
         will contain only the masked region and will be in [0,1] interval
        :param mr_path:
        :param brainmask_path:
        :return: numpy array of processed MRI and a tuple containing original min,max values
        """

        mr_scan = self._load_mr(mr_path)
        if brainmask_path is not None:
            brainmask = self._load_mr(brainmask_path)
            brainmask = binary_closing(brainmask > 0, iterations=2)

            return self.scaler(self._mask_mr(mr_scan, brainmask))

        return self.scaler(self._crop_mr(mr_scan))

    def __str__(self):
        return str(self.__dict__)

    def _load_mr(self, path: str) -> np.ndarray:
        """
        Loads and conform the source file
        Args:
            path:

        Returns:

        """
        conform_scan = nib.load(path)
        if path.endswith(".nii.gz"):
            pixdim = conform_scan.header["pixdim"][1:4]
            orient = "".join(nib.aff2axcodes(conform_scan.affine))
        else:
            pixdim = conform_scan.header["delta"]
            orient = "RAS" if conform_scan.header["goodRASFlag"]==1 else "UNK"

        if not np.allclose(pixdim - np.array(self.voxel_size), np.array([0,0,0]), atol=0.2) or orient != self.orientation:
            conform_scan = processing.conform(conform_scan,
                                  voxel_size=self.voxel_size,
                                  orientation=self.orientation).get_fdata()
        else:
            conform_scan = np.resize(conform_scan.get_fdata(),(256,256,256))

        return conform_scan

    def _mask_mr(self, raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply the mask to raw by means of a bitwise multiplication
        Args:
            raw: raw MR scan
            mask: brainmask

        Returns: the masked MR scan

        """
        if raw.shape != mask.shape:
            raise ValueError("Shapes of MR and mask don't match")
        raw *= mask
        return self._crop_mr(raw)

    def _crop_mr(self, raw: np.ndarray):
        """ Crop to the specified size"""
        diff_shape = np.abs(self.output_size - np.array(raw.shape))
        return raw[
               ceil(diff_shape[0] / 2):-floor(diff_shape[0] / 2),
               ceil(diff_shape[1] / 2):-floor(diff_shape[1] / 2),
               ceil(diff_shape[2] / 2):-floor(diff_shape[2] / 2)
               ]
