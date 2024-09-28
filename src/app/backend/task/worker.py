"""This module configure the celery worker"""
import os

from celery import Celery

MODEL_PATH = "models/model.ckpt"
BROKER_URI = os.getenv("BROKER_URI", "redis://localhost:6379/0")
BACKEND_URI = os.getenv("BACKEND_URI", "redis://localhost:6379/1")

celery_app = Celery(
    'celery_task',
    broker=BROKER_URI,
    backend=BACKEND_URI,
    include=['task.tasks']
)

celery_app.conf.beat_schedule = {
    "drift-check-window": {
        "task": "task.tasks.DriftDetectWindow",
        "schedule": 600  # seconds
    },
    "drift-check": {
        "task": "task.tasks.DriftDetect",
        "schedule": 1200
    }
}
