# pylint: disable=too-few-public-methods
""" This module contains the pydantic models used into the responses"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ============================
#  COMPOSITE TYPE DEFINITIONS
# ============================

class Prediction(BaseModel):
    """ Model predictions """
    predicted_probability: float = Field(description="Predicted probability value")
    predicted_class: int = Field(description="Predicted class value, will be 0 if probability is less than 0.5")


class ProcessedInfo(BaseModel):
    """ Informations about file preprocessing"""
    source_files: List[str] = Field(description="List of files that have been processed")
    file_url: str = Field(default="", description="Link to the processed file resource")


class AttributionInfo(BaseModel):
    """ Informations about gradient attribution"""
    attribution_method: str = Field(default="Integrated Gradients", description="Attribution method name")
    iterations: int = Field(description="Number of iterations used to compute the attribution")
    file_url: str = Field(default="", description="Link to the attribution file resource")


# ============================
#      RESPONSE MODELS
# ============================

class BaseResponse(BaseModel):
    """
    Base response model
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    message: Optional[str] = ""

    class Config:
        """Config extra for documentation purposes"""
        schema_extra = {
            "example": {
                'timestamp': "2022-11-15T16:39:17.701164",
                'message': ""
            }
        }


class PredictResponse(BaseResponse):
    """
    Response model used for gather the predictions
    """
    task_id: str = None
    start_time: datetime = None
    processing_seconds: float = None
    inference_seconds: float = None
    attribution_seconds: float = None
    prediction: Prediction = None
    attribution: AttributionInfo = None
    processed: ProcessedInfo = None

    class Config:
        """Config extra for documentation purposes"""
        schema_extra = {
            'example': {
                "timestamp": "2022-11-15T17:46:46.275410",
                "message": "null",
                "task_id": "37d03b51-10de-4c26-b712-fa38a52bfbbc",
                "start_time": "2022-11-15T17:46:33.570207",
                "processing_seconds": 11.392892,
                "inference_seconds": 1.392892,
                "attribution_seconds": 115.392892,
                "prediction": {
                    "predicted_probability": 0.6101745963096619,
                    "predicted_class": 1
                },
                "attribution": {
                    "attribution_method": "Integrated Gradients",
                    "iterations": 1,
                    "attribution_file":
                        "http://127.0.0.1:8000/results/37d03b51-10de-4c26-b712-fa38a52bfbbc/files"
                        "/attribution "
                },
                "processed": {
                    "source_files": [
                        "OAS30271_MR_d0004_T1w.nii.gz",
                        "OAS30271_MR_d0004_brainmask.mgz"
                    ],
                    "processed_file":
                        "http://127.0.0.1:8000/results/37d03b51-10de-4c26-b712-fa38a52bfbbc/files/processed"
                }
            }
        }


class ProcessingResponse(BaseResponse):
    """
    Response module used to tell user that predict request
    has been taken in charge and is under processing
    """
    message = 'Processing'
    task_id: str
    requested_iter: int = None
    status_url: Optional[str] = None
    result_url: Optional[str] = None

    class Config:
        """Config extra for documentation purposes"""
        schema_extra = {
            'example': {
                "timestamp": "2022-11-15T17:41:10.040434",
                "message": "Completed",
                "accepted_time": "2022-11-15T17:36:11.040867",
                "task_id": "2f1ab2c4-faac-4939-b3e0-ddbee5162704",
                "status_url": "http://127.0.0.1:8000/predict/2f1ab2c4-faac-4939-b3e0-ddbee5162704",
                "result_url": "http://127.0.0.1:8000/results/2f1ab2c4-faac-4939-b3e0-ddbee5162704",
            }
        }
