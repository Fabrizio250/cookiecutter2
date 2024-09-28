# syntax=docker/dockerfile:1
FROM python:3.10.8-slim as builder
ENV PATH /usr/local/bin:$PATH
WORKDIR /app

RUN apt update && apt install -y python3-dev \
                          gcc \
                          libc-dev \
                          libffi-dev

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install pytorch-lightning
COPY requirements.txt /app
RUN pip3 install -r requirements.txt

COPY . /app

RUN --mount=type=secret,id=ML_FLOW_USERNAME \
    --mount=type=secret,id=ML_FLOW_PASSWORD \
    export MLFLOW_TRACKING_USERNAME=$(cat /run/secrets/ML_FLOW_USERNAME) && \
    export MLFLOW_TRACKING_PASSWORD=$(cat /run/secrets/ML_FLOW_PASSWORD) && \
    python download_model.py

ENV BROKER_URI=redis://redis:6379/0
ENV BACKEND_URI=redis://redis:6379/1

EXPOSE 8000

CMD ["bash", "starter_script.sh"]