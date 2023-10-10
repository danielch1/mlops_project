FROM python:3.9-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_monitoring.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY .env .

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir