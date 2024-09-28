"""
This script download the model artifact pointed by the field 'artifact_uri' in the config file
and rename to model.ckpt.
NOTE: the model is downloaded into the models folder under the root
"""
import os
import sys
from configparser import ConfigParser

import mlflow.artifacts

ENV_MLFLOW_USERNAME = "MLFLOW_TRACKING_USERNAME"
ENV_MLFLOW_PASSWORD = "MLFLOW_TRACKING_PASSWORD"

if __name__ == '__main__':
    config = ConfigParser()
    file_path = sys.path[0]
    config.read([file_path + "/config", file_path + "/config.local"])

    if os.getenv(ENV_MLFLOW_USERNAME) is None:
        os.environ[ENV_MLFLOW_USERNAME] = config.get("MLFLOW", "username")
    if os.getenv(ENV_MLFLOW_PASSWORD) is None:
        os.environ[ENV_MLFLOW_PASSWORD] = config.get("MLFLOW", "password")

    mlflow.set_tracking_uri(config.get("MLFLOW", "tracking_uri"))
    models_path = "models/"
    downloaded_artifact = mlflow.artifacts.download_artifacts(
        artifact_uri=config.get("ARTIFACT", "artifact_uri"),
        dst_path=models_path)
    os.replace(downloaded_artifact, os.path.join(models_path, "model.ckpt"))
