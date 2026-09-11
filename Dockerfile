FROM python:3.13.2-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SERVING_BUNDLE_ROOT=/opt/robofleet/serving_bundle
ENV PREDICTION_DATABASE_PATH=/var/lib/robofleet/prediction_events.db

WORKDIR /app

# Installing the native runtime required by LightGBM
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installing the exact model dependencies recorded during training
COPY deployment/serving_bundle/model/requirements.txt ./requirements-model.txt
COPY requirements-serving.txt .

RUN pip install \
    --no-cache-dir \
    --requirement requirements-model.txt \
    --requirement requirements-serving.txt

COPY api ./api
COPY deployment/serving_bundle /opt/robofleet/serving_bundle

# Preparing persistent storage before dropping root privileges
RUN useradd --create-home --uid 10001 apiuser \
    && mkdir --parents /var/lib/robofleet \
    && chown --recursive apiuser:apiuser /var/lib/robofleet

USER apiuser

EXPOSE 8000

HEALTHCHECK CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=5)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]