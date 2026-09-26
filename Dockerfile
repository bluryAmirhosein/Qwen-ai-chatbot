FROM python:3.12-slim

# CPU-only PyTorch by default (keeps the image several GB smaller).
# For a GPU build pass: --build-arg TORCH_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl is required for the container's own HEALTHCHECK (see docker-compose.yml).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Dependencies first, so this layer stays cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --extra-index-url "${TORCH_INDEX_URL}" -r requirements.txt

COPY --chown=app:app app ./app
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app alembic.ini .
COPY --chown=app:app entrypoint.sh .
RUN chmod +x entrypoint.sh

# Mount points: ai-models is bind-mounted from the host, data/logs are named volumes.
RUN mkdir -p /app/ai-models /app/data /app/logs \
    && chown -R app:app /app/ai-models /app/data /app/logs

USER app

EXPOSE 8000

# Run pending migrations, then start the (single-process) app.
ENTRYPOINT ["./entrypoint.sh"]