# MLflow tracking server + model registry.
# The official MLflow image has no Postgres driver, so we add psycopg2 ourselves.
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.16.1 psycopg2-binary==2.9.13 "sqlalchemy<2.1"

EXPOSE 5000
