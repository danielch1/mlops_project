FROM python:3.9-slim

EXPOSE $PORT

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY config/ config/
COPY .env .

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

WORKDIR /src/models
CMD exec uvicorn predict_model:app --port $PORT --host 0.0.0.0 --workers 1
