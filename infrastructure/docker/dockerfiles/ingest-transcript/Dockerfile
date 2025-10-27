FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y ffmpeg

COPY infrastructure/docker/dockerfiles/ingest-transcript/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src /app/src
COPY ./infrastructure /app/infrastructure

CMD ["python", "src/rag/cli.py", "ingest-transcript"]