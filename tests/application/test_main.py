"""
Test methods for Web API
"""
from unittest import mock

import pytest
from starlette.testclient import TestClient
from http import HTTPStatus
from src.app.backend.main import app


def fake_task_success():
    return dict(status="SUCCESS")


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.mark.web_api
def test_get_main(client):
    """
    Test the entry point in the WEB API
    """
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK


@pytest.mark.web_api
def test_predict_no_data(client):
    """
    Request a prediction on null data, should fail
    """
    response = client.post("/predict")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
