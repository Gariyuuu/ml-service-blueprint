# syntax=docker/dockerfile:1.7
#
# Two stages: a builder that resolves and compiles dependencies into a wheel and
# a virtualenv, and a runtime that copies only the finished venv. Nothing that
# builds a wheel (compilers, headers, pip caches) survives into the shipped
# image, which is where most of the size and most of the attack surface lives.

# --- builder -----------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Dependencies come from the lockfile so an image rebuilt six months from now
# resolves the same versions. Regenerate it with `make lock`.
COPY requirements.lock ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --require-hashes --no-deps -r requirements.lock

# Source is copied after dependencies so a code-only change reuses the cached
# dependency layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-deps .

# --- runtime -----------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    MLSERVICE_REGISTRY_ROOT=/app/registry \
    MLSERVICE_HOST=0.0.0.0 \
    MLSERVICE_PORT=8000 \
    MLSERVICE_ENVIRONMENT=prod \
    MLSERVICE_OBS_LOG_FORMAT=json

# curl is here only for the HEALTHCHECK below. Drop it and the healthcheck
# together if your orchestrator probes /ready over HTTP itself (Kubernetes does).
RUN apt-get update \
 && apt-get install --no-install-recommends -y curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root, no login shell, no home directory to write into.
RUN groupadd --system --gid 10001 mlservice \
 && useradd --system --uid 10001 --gid mlservice --no-create-home --shell /usr/sbin/nologin mlservice

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# The registry is baked into the image so the container is self-contained and
# immutable: one image, one model version, reproducible rollbacks by tag. Mount
# a volume over /app/registry instead if you deploy models independently of code.
COPY --chown=mlservice:mlservice registry /app/registry

USER mlservice
EXPOSE 8000

# Liveness only. /ready is the readiness signal and it is deliberately NOT used
# here: a container whose model failed to load should be reported unready to the
# load balancer, not killed and restarted into the same failure.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${MLSERVICE_PORT}/health" || exit 1

# One worker per container: scale with replicas, not with in-container workers,
# so each replica's memory footprint is one copy of the model.
CMD ["uvicorn", "mlservice.serving.app:build_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
