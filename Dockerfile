# ---------- Stage 1: build dependencies (compilers live only here) ----------
FROM python:3.12-slim AS builder

# CPU-only PyTorch by default (keeps the image several GB smaller).
# For a GPU build pass: --build-arg TORCH_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Extra CMake flags for llama.cpp. Leave empty to build optimized for the CPU of the
# machine running the build. If you build on one machine and run on another, use e.g.:
#   --build-arg LLAMA_CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON"
ARG LLAMA_CMAKE_ARGS=""

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_ARGS="${LLAMA_CMAKE_ARGS}"

# build-essential + cmake are required to compile llama-cpp-python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Dependencies first, so this layer stays cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --extra-index-url "${TORCH_INDEX_URL}" -r requirements.txt


# ---------- Stage 2: runtime image ----------
FROM python:3.12-slim

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# curl is required for the container's own HEALTHCHECK (see docker-compose.yml).
# libgomp1 is the OpenMP runtime that llama.cpp links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid app --create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

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