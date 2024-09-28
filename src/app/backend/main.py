""" This module provides the API endpoints"""
import os.path
import shutil
import tempfile
from os.path import join
from typing import Optional
from datetime import datetime

from http import HTTPStatus

import numpy as np
from celery.result import AsyncResult
from fastapi import FastAPI, Request, Response, HTTPException, Form, UploadFile
from pydantic import ValidationError
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse

import user_payload
import response_models
import task.tasks as tt
import prometheus_instrumentator as pi

app = FastAPI(
    title="3DConvAD",
    description="Alzeimer\'s dementia detection using 3d convolutional neural network",
    version="1.0",
    middleware=[Middleware(CORSMiddleware,
                           allow_origins=["*"],
                           allow_credentials=True,
                           allow_methods=["GET", "POST"],
                           allow_headers=["*"],
                           expose_headers=["*"])]
)


@app.on_event('startup')
async def startup():
    pi.instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)


@app.get("/", tags=["General"], response_model=response_models.BaseResponse, status_code=HTTPStatus.OK)
def read_root():
    """ Root endpoint used only as welcome message"""
    content = {"message": "Welcome to 3DConvAD classifier! Please, read the `/docs`!"}
    return response_models.BaseResponse(**content)


@app.post("/predict", tags=["Prediction"],
          status_code=HTTPStatus.ACCEPTED,
          response_model=response_models.ProcessingResponse)
async def get_prediction(request: Request,
                         brainscan: UploadFile,
                         brainmask: Optional[UploadFile] = None,
                         n_iter: int = Form(default=1)):
    """
     Perform the prediction on the uploaded Brain scans, the brainscan can be either a raw T1w MR scan
     (as taken from the MR scanner) in the Nifti format or a processed brain volume with Freesurfer.
     In the case of a Nifti file a brainmask must be provided in the form of a Freesurfer compressed file
     """
    try:
        payload = user_payload.UserPayload(brainscan=brainscan, brainmask=brainmask, n_iter=n_iter)
    except ValidationError as exc:  # pylint: disable=raise-missing-from
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc))
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    mr_path = join(tempfile.tempdir, timestr + ";" + payload.brainscan.filename)
    brainmask_path = join(tempfile.tempdir, timestr + ";" + brainmask.filename) \
        if brainmask is not None else None
    with open(mr_path, "wb") as mri:
        shutil.copyfileobj(brainscan.file, mri)
    if brainmask is not None:
        with open(brainmask_path, "wb") as mask:
            shutil.copyfileobj(brainmask.file, mask)
    task_id = tt.predict_single_mr.delay(mr_path, brainmask_path, n_iter)
    response = {
        "task_id": str(task_id),
        "status_url": request.url_for('get_prediction_status', task_id=task_id),
        "requested_iter": n_iter}
    return response_models.ProcessingResponse(**response)


@app.get("/predict/{task_id}", tags=["Prediction"],
         status_code=HTTPStatus.OK,
         response_model=response_models.ProcessingResponse)
async def get_prediction_status(response: Response, request: Request, task_id: str):
    """
    Get the status of the prediction specified by task_id, the response will contain
    the link to the result resource only if the prediction is completed. Otherwise, will
    be empty, the two normal statuses are: Processing and completed. If the provided task_id
    does not exist the usual 404 is returned.
    """
    task = AsyncResult(task_id)
    if task.status == 'SUCCESS':
        response.status_code = HTTPStatus.OK
        return response_models.ProcessingResponse(task_id=task_id,
                                                  message='Completed',
                                                  requested_iter=task.result["attribution"]["iterations"],
                                                  status_url=request.url_for("get_prediction_status", task_id=task_id),
                                                  result_url=request.url_for("get_result", task_id=task_id))
    if task.status == 'PENDING':
        response.status_code = HTTPStatus.ACCEPTED
        response = {
            "task_id": str(task_id),
            "status_url": request.url_for('get_prediction_status', task_id=task_id),
            "requested_iter": 0}
        return response_models.ProcessingResponse(**response)
    if task.status == 'PROGRESS':
        response.status_code = HTTPStatus.ACCEPTED
        return response_models.ProcessingResponse(task_id=task_id,
                                                  requested_iter=task.result["requested_iter"],
                                                  status_url=request.url_for("get_prediction_status", task_id=task_id))
    if task.status == 'FAILURE':
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


@app.get("/results/{task_id}",
         status_code=HTTPStatus.OK,
         response_model=response_models.PredictResponse,
         response_class=JSONResponse,
         tags=['Results'])
async def get_result(request: Request, task_id: str):
    """
    Get results of the prediction specified by task_id
    """
    task = AsyncResult(task_id)
    if task.status == 'PENDING':
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND)
    if task.status == 'PROCESSING':
        raise HTTPException(status_code=HTTPStatus.PROCESSING)
    if task.status == 'FAILURE':
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    result = response_models.PredictResponse(**task.get())
    result.processed.file_url = request.url_for('get_processed_file', task_id=task_id)
    result.attribution.file_url = request.url_for('get_attribution_file', task_id=task_id)
    return result


def retrieve_file(filename: str, task):
    if task.status != "SUCCESS":
        raise FileNotFoundError

    start_time = datetime.fromisoformat(task.result["start_time"])
    path = join("/data", "feature_store", "data", str(start_time.timestamp()), filename)
    if not os.path.exists(path):
        raise FileNotFoundError

    return path


@app.get("/results/{task_id}/files/processed",
         status_code=HTTPStatus.OK,
         response_class=FileResponse,
         tags=['Results'])
async def get_processed_file(task_id: str):
    """ Download the processed file pointed by id"""
    filename = "processed.npy"
    try:
        task = AsyncResult(task_id)
        path = retrieve_file(filename, task)
        return FileResponse(path,
                            filename=filename,
                            headers={'Content-type': 'application/octet-stream'})
    except FileNotFoundError:
        raise HTTPException(status_code=404)


@app.get("/results/{task_id}/files/attribution",
         status_code=HTTPStatus.OK,
         response_class=FileResponse,
         tags=['Results'])
async def get_attribution_file(task_id: str):
    """ Download the attribution file pointed by id"""
    filename = "integrated_gradients.npy"
    try:
        task = AsyncResult(task_id)
        path = retrieve_file(filename, task)
        return FileResponse(path,
                            filename=filename,
                            headers={'Content-type': 'application/octet-stream'})
    except FileNotFoundError:
        raise HTTPException(status_code=404)


@app.get("/features/",
         status_code=HTTPStatus.OK,
         response_class=FileResponse,
         tags=['Mangement'])
async def get_features(number: int=100):
    """
    Download the features extracted by the model as a compressed numpy array, the file contains two arrays,
    the first accessible by the key `data` contains the features in the shape (number, 9216)
    sorted in ascending order by timestamp, the second labeled with key `timestamps` contains the timestamps in which
    the original scan has been processed
    """
    files = sorted(os.listdir("/data/feature_store/features"))
    if number is None:
        number = len(files)
    start_index = max(len(files) - number, 0)
    files = files[start_index:]
    result = np.zeros((len(files), 9216))
    for i in range(len(files)):
        result[i,:] = np.load(join("/data/feature_store/features", files[i]))
    timestamp = str(datetime.now().timestamp())
    np.savez_compressed(f"/data/{timestamp}.npz", data=result, timestamps=[x[:-4] for x in files])
    return FileResponse(f"/data/{timestamp}.npz",
                        filename=timestamp+".npz",
                        headers={'Content-type': 'application/octet-stream'})