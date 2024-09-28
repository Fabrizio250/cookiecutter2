# pylint: disable=abstract-method
""" This module implement the prediction task with celery """
import datetime
import json
import os
import os.path
import time
from abc import ABC
from os.path import join
from urllib.error import URLError
import psutil

import numpy as np
import psutil
from alibi_detect.cd import MMDDrift
from celery import Task
from celery.utils.log import get_task_logger
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, Info

from .worker import MODEL_PATH, celery_app
from .scan_process import MRIProcessor
from .model import MRIClassifier

logger = get_task_logger(__name__)


class PredictTask(Task, ABC):
    """Celery task for making the prediction and gradient attribution"""
    abstract = True

    def __init__(self):
        super().__init__()
        self.model = None
        self.mr_processor = None
        self.data_dir = join("/data", "feature_store", "data")
        self.feature_dir = join("/data", "feature_store", "features")

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)

    def __call__(self, *args, **kwargs):
        """ Instantiate the model and mr processor if they're not already instantiated"""
        if not self.model:
            self.model = MRIClassifier.load_from_checkpoint(MODEL_PATH)
            self.model.freeze()
        if not self.mr_processor:
            self.mr_processor = MRIProcessor()

        return self.run(*args, **kwargs)


@celery_app.task(ignore_result=False,
                 bind=True,
                 base=PredictTask,
                 path=('src.models.mri_classifier', 'MRIClassifier'),
                 name=f'{__name__}.MRI')
def predict_single_mr(self, mri_scan: str, brainmask: str = None, n_iter: int = 1) -> dict:
    """
    Predict and compute gradient attribution on a single sample
    """
    start_time = datetime.datetime.now()
    self.update_state(state='PROGRESS', meta={"accepted": start_time, "requested_iter": n_iter})
    try:
        img = self.mr_processor(mri_scan, brainmask)
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        start = end_time
        pred, features = self.model.predict(img)

        end_time = datetime.datetime.now()
        inference_time = (end_time - start).total_seconds()

        start = end_time
        attr = self.model.get_attribution(img, n_steps=n_iter)
        end_time = datetime.datetime.now()
        attribution_time = (end_time - start).total_seconds()

        out_folder = join(self.data_dir, str(start_time.timestamp()))

        os.makedirs(out_folder)
        processed_file = join(out_folder, "processed.npy")
        attribution_file = join(out_folder, "integrated_gradients.npy")
        features_file = join(self.feature_dir, str(start_time.timestamp()) + ".npy")
        np.save(processed_file, img)
        np.save(attribution_file, attr)
        np.save(features_file, features)

        predicted_class = 0 if pred < 0.5 else 1
        response = {
            'task_id': predict_single_mr.request.id,
            'start_time': start_time,
            'prediction': {
                "predicted_probability": pred,
                "predicted_class": predicted_class
            },
            'processing_seconds': processing_time,
            'inference_seconds': inference_time,
            'attribution_seconds': attribution_time,
            'attribution': {
                'attribution_method': "Integrated Gradients",
                "iterations": n_iter,
                "attribution_file": ""},
            'processed': {
                'source_files': [os.path.basename(mri_scan).split(";")[-1],
                                 os.path.basename(brainmask).split(";")[-1] if brainmask is not None else ""],
                "processed_file": ""
            }

        }
        with open(join(out_folder, "metadata.json"), "w") as outfile:
            json.dump(response, outfile, indent=4, default=str)

        return response
    except Exception as ex:  # pylint: disable=broad-except
        self.update_state(status='FAILURE', meta={'error': str(ex)})
        return {"message": str(ex)}


class DriftDetect(Task, ABC):
    """Celery task for drift detection"""
    abstract = True

    def __init__(self):
        super().__init__()
        self.data_dir = join("/data", "feature_store")
        self.feature_dir = join("/data", "feature_store", "features")
        self.detector = None
        self.registry = CollectorRegistry()
        self.drift_info = Info("drift", "Drift status growing window",
                               registry=self.registry, namespace="driftdetector")

    def __call__(self, *args, **kwargs):
        """ Instantiate the model and mr processor if they're not already instantiated"""

        return self.run(*args, **kwargs)


@celery_app.task(ignore_result=False,
                 bind=True,
                 base=DriftDetect,
                 name=f'{__name__}.DriftDetect')
def eval_drift(self):
    if psutil.virtual_memory().available > 1.07e9:

        x_ref = np.load(join(self.data_dir, "x_ref.npz"))["data"]
        detector = MMDDrift(x_ref=x_ref, p_val=0.05, backend='pytorch', x_ref_preprocessed=True)

        filelist = sorted(os.listdir(self.feature_dir))
        first_ts = float(filelist[0][:-4])
        x = np.array([np.load(join(self.feature_dir, f)) for f in filelist], dtype=float)
        result = detector.predict(x=x)["data"]
        try:
            result["distance_threshold"] = float(result["distance_threshold"])
            result.update({"since": datetime.datetime.fromtimestamp(first_ts).isoformat()})
            result = dict(zip(result.keys(), [str(v) for v in result.values()]))
            self.drift_info.info(result)

            push_to_gateway('gateway:9091', job='drift_detect', registry=self.registry)
        except URLError:
            logger.error("Gateway is not responding")
        del detector
        return "done"

    return "postponed"


class DriftDetectWindow(Task, ABC):
    """Celery task for drift detection"""
    abstract = True

    def __init__(self):
        super().__init__()
        self.data_dir = join("/data", "feature_store")
        self.feature_dir = join("/data", "feature_store", "features")
        self.registry = CollectorRegistry()
        self.drift_info = Info("drift_fixed", "Drift status fixed window (last hour)",
                               registry=self.registry, namespace="driftdetector")

    def __call__(self, *args, **kwargs):
        """ Instantiate the model and mr processor if they're not already instantiated"""

        return self.run(*args, **kwargs)


@celery_app.task(ignore_result=False,
                 bind=True,
                 base=DriftDetect,
                 name=f'{__name__}.DriftDetectWindow')
def eval_drift_fixed_window(self):

    if psutil.virtual_memory().available > 1.07e9:
        from_ts = (datetime.datetime.now() - datetime.timedelta(minutes=60)).timestamp()
        filelist = list(filter(lambda x: float(x[:-4]) > from_ts, os.listdir(self.feature_dir)))
        if len(filelist) > 1:
            x_ref = np.load(join(self.data_dir, "x_ref.npz"))["data"]
            detector = MMDDrift(x_ref=x_ref, p_val=0.05, backend='pytorch', x_ref_preprocessed=True)

            x = np.array([np.load(join(self.feature_dir, f)) for f in sorted(filelist)], dtype=float)
            result = detector.predict(x=x)["data"]
            del detector
            try:
                result["distance_threshold"] = float(result["distance_threshold"])
                result.update({"number_samples": len(filelist)})
                result = dict(zip(result.keys(), [str(v) for v in result.values()]))
                self.drift_info.info(result)
                push_to_gateway('gateway:9091', job='drift_detect', registry=self.registry)
            except URLError:
                logger.error("Gateway is not responding")
        return "done"
    return "postponed"
