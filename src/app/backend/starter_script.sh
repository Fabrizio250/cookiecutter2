#!/bin/bash

# Starts the fastapi server
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level 'error' &

# Starts the celery worker for the prediction task
celery -A task.worker worker --concurrency=1 -E -B &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
