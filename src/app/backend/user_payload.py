""" Model classes used in the API """
import os
from pydantic import BaseModel, validator, root_validator
from fastapi import UploadFile

BS_RAW_TYPE = ['application/gzip','application/x-gzip']
BS_MASK_TYPE = ['application/vnd.proteus.magazine',"application/octet-stream"]
BS_RAW_EXT = '.gz'
BS_PREP_EXT = '.mgz'


class UserPayload(BaseModel):
    """ User Payload Model """
    brainscan: UploadFile | None = None
    brainmask: UploadFile | None = None
    n_iter: int | None = 1

    @validator('brainscan', check_fields=False)
    def raw_scan(cls, value):
        """Require raw brainscan extension and MIME-type to be coherent"""

        _, file_ext = os.path.splitext(value.filename)
        if value.content_type in BS_RAW_TYPE:
            if file_ext != BS_RAW_EXT:
                raise ValueError(f"Unsupported file extension, it must be {BS_RAW_EXT}, is {file_ext}")
        if file_ext == BS_RAW_EXT:
            if value.content_type not in BS_RAW_TYPE:
                raise ValueError(f"Unsupported MIME-type, it must be {BS_RAW_TYPE}")
        return value

    @validator('brainscan', check_fields=False)
    def prep_scan(cls, value):
        """Require preprocessed brainscan extension and MIME-type to be coherent"""
        _, file_ext = os.path.splitext(value.filename)
        if value.content_type in BS_MASK_TYPE:
            if file_ext != BS_PREP_EXT:
                raise ValueError(f"Unsupported file extension, it must be {BS_PREP_EXT}, is {file_ext}")
        if file_ext == BS_PREP_EXT:
            if value.content_type not in BS_MASK_TYPE:
                raise ValueError(f"Unsupported MIME-type, it must be {BS_MASK_TYPE}")
        return value

    @validator('brainmask', check_fields=False)
    def mime_mask(cls, value):
        """Require brainmask MIME-type to be application/vnd.proteus.magazine"""
        if value is None:
            return value
        _, file_ext = os.path.splitext(value.filename)
        if file_ext != BS_PREP_EXT:
            raise ValueError(f"Unsupported file extension, it must be {BS_PREP_EXT}, is {file_ext}")
        if value.content_type not in BS_MASK_TYPE:
            raise ValueError(f"Unsupported media type, it must be {BS_MASK_TYPE}")
        return value

    @root_validator
    def scan_mask_cross_dependency(cls, values):
        """ Require a raw brainscan and a brainmask or a lone preprocessed brainscan """
        b_scan = values.get('brainscan')
        b_mask = values.get('brainmask')
        if b_mask is None and b_scan is not None and b_scan.content_type in BS_RAW_TYPE:
            raise ValueError(f"Raw brainscan need mask file, it must be {BS_RAW_TYPE}")
        return values

    @validator('n_iter')
    def iter_cap(cls, value):
        """ Require a n_iter in range 1..100 """
        if value < 1 or value > 50:
            raise ValueError("Number of iteration must be in range 1..50")
        return value
