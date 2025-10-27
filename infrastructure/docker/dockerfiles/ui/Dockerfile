FROM python:3.11-slim

WORKDIR /app

COPY infrastructure/docker/dockerfiles/ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ui /app

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]