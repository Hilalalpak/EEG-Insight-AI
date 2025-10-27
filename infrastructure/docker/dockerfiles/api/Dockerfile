FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        zlib1g-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY infrastructure/docker/dockerfiles/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/ /app/src
COPY ./src/api/ /app/api
COPY ./infrastructure/ /app/infrastructure

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8001"]