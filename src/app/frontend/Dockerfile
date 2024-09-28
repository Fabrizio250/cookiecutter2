# syntax=docker/dockerfile:1
FROM python:3.10-alpine AS base
WORKDIR /app
COPY requirements.txt /app

ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
#ENV API_URL=http://api.3dconvad.trendatre3.duckdns.org/
ENV API_URL=http://localhost:8000/


RUN pip install -r requirements.txt

COPY . /app


EXPOSE 5000
CMD ["flask", "run"]