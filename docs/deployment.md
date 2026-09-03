# Deployment

> **Verification status.** Everything on this page below the container section —
> configuration, registry strategy, probes, observability wiring — is exercised by
> the test suite and `make golden-path`. **The container itself is not.** Docker
> was unavailable on the machine this release was built on, so the `Dockerfile`
> and `tests/container/` have never been executed. They are unverified, not known
> to be broken. Run `make train-promote && make docker-build && make docker-smoke`
> on a Docker-enabled machine before relying on them.

## Container

```bash
make train-promote      # the image bakes in registry/, so a model must exist
make docker-build
make docker-run
```

### What the image does and does not contain

Two stages. The builder resolves dependencies from `requirements.lock` into a
virtualenv; the runtime copies only that venv. Nothing that builds a wheel —
compilers, headers, pip caches — survives into the shipped image.

The runtime contains: Python 3.11 slim, the locked runtime dependencies, the
`mlservice` package, `curl` (only for the `HEALTHCHECK`), and `registry/`.

It does not contain: dev, test, or load-test dependencies; the training data;
the test suite; the docs; the OpenTelemetry extra (add it if you enable tracing).

### Non-root

```dockerfile
RUN groupadd --system --gid 10001 mlservice \
 && useradd --system --uid 10001 --gid mlservice --no-create-home --shell /usr/sbin/nologin mlservice
USER mlservice
```

No home directory, no login shell. The registry is copied with
`--chown=mlservice:mlservice` because the process must read it; it never writes
there.

### Locked dependencies

`requirements.lock` is fully pinned with hashes, installed with
`--require-hashes --no-deps`. An image rebuilt in six months resolves the same
versions. Regenerate with `make lock`; CI fails if the lockfile has drifted from
`pyproject.toml`.

### Healthcheck

```dockerfile
HEALTHCHECK CMD curl -fsS "http://127.0.0.1:${MLSERVICE_PORT}/health" || exit 1
```

It probes `/health` (liveness), **not** `/ready`. A container whose model failed
to load should be reported unready to the load balancer, not killed and restarted
into the same failure. On Kubernetes, delete the `HEALTHCHECK` and the `curl`
install entirely — kubelet probes over HTTP itself, and dropping `curl` removes a
binary from the image.

### One worker per container

```dockerfile
CMD ["uvicorn", ..., "--workers", "1"]
```

Each uvicorn worker loads its own copy of the model. Scale with replicas, not
in-container workers, so a replica's memory footprint is one model and the
scheduler can see it.

## Where the registry lives

The Dockerfile bakes `registry/` into the image. Three options, and the choice
determines how you deploy models:

| Strategy | Deploy a model by | Good when |
| --- | --- | --- |
| **Baked in** (default) | building and rolling out a new image | you want one immutable artifact per deploy and rollback-by-tag |
| **Mounted volume** | promoting a stage, then restarting pods | models ship faster than code |
| **Synced at startup** | an init container pulling from S3/GCS | the registry is shared across teams |

Baked in is the default because it makes the image the unit of deployment: the
running container cannot disagree with what you built, and `docker run <old-tag>`
is a complete rollback.

To mount instead, remove the `COPY --chown=mlservice:mlservice registry` line and
mount your registry read-only at `/app/registry`.

## Configuration

Everything is `MLSERVICE_*` and `MLSERVICE_OBS_*` environment variables. See
`.env.example` for the full surface and `configs/service.example.yaml` for
annotated production values.

The ones that actually matter in production:

```bash
MLSERVICE_ENVIRONMENT=prod
MLSERVICE_REGISTRY_ROOT=/app/registry
MLSERVICE_MODEL_NAME=tabular-classifier
MLSERVICE_MODEL_STAGE=production        # or pin MLSERVICE_MODEL_VERSION=v3
MLSERVICE_FAIL_FAST_ON_MISSING_MODEL=true
MLSERVICE_MAX_BATCH_SIZE=1000
MLSERVICE_OBS_LOG_FORMAT=json
```

### Pinning a version instead of a stage

`MLSERVICE_MODEL_VERSION=v4` overrides stage resolution. Use it for a canary
deployment (a second deployment, same image, pinned to the candidate) and for
reproducing an incident against the exact version that caused it.

### Secrets

There are none. This service reads a filesystem registry and serves HTTP; it has
no database, no API keys, no outbound credentials. `.env.example` contains no
secrets and neither should your `.env`.

If your fork adds one — a database URL, an object-store key, an OTLP header —
inject it from your secret manager at runtime. Do not add it to the image, the
lockfile, or the repo.

## Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
spec:
  replicas: 3
  selector:
    matchLabels: { app: ml-service }
  template:
    metadata:
      labels: { app: ml-service }
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        fsGroup: 10001
      containers:
        - name: service
          image: ml-service-blueprint:v3
          ports: [{ containerPort: 8000 }]
          env:
            - { name: MLSERVICE_ENVIRONMENT,  value: prod }
            - { name: MLSERVICE_MODEL_NAME,   value: tabular-classifier }
            - { name: MLSERVICE_MODEL_STAGE,  value: production }
            - { name: MLSERVICE_OBS_LOG_FORMAT, value: json }
          # Liveness: is the process alive? Never model state — a restart cannot
          # fix a missing model, so this must not trigger a restart loop.
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 15
            periodSeconds: 20
            failureThreshold: 3
          # Readiness: can this replica serve? 503 while the model is not loaded.
          readinessProbe:
            httpGet: { path: /ready, port: 8000 }
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 2
          # Startup: gives a slow model load up to 60s without a slack liveness probe.
          startupProbe:
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 3
            failureThreshold: 20
          resources:
            requests: { cpu: 250m, memory: 512Mi }
            limits:   { cpu: "1",  memory: 1Gi }
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: ["ALL"] }
```

`readOnlyRootFilesystem: true` works as shipped **unless** you enable the JSONL
drift sink, which writes to `MLSERVICE_OBS_DRIFT_SINK_PATH`. Use the `logging`
sink, or mount an `emptyDir` at that path.

### Sizing

Measure, do not copy. Memory is dominated by the deserialized model; a gradient
boosting model with 200 estimators on 30 features is small, a forest with 1000
trees on 500 features is not. CPU per request is dominated by batch size.

Start with the requests above, run your own load profile
(`loadtest/README.md`), and set limits from what you measure.

## Scaling

**Horizontally.** The service is stateless: a replica holds a read-only model and
some counters. Add replicas.

**Batch, if your callers can.** Per-row cost drops sharply with batch size —
JSON parsing, frame construction, and validation are per-request, not per-row.
This is usually a bigger win than another replica.

**Raise `MLSERVICE_MAX_BATCH_SIZE` deliberately.** It exists so one caller cannot
send a million rows and pin a worker. Raise it with a measurement, not a guess.

## Rolling out a new model

```bash
mlservice registry promote tabular-classifier v4 production --reason "..."
kubectl rollout restart deployment/ml-service
kubectl rollout status deployment/ml-service
curl -s $SERVICE/model-info | jq .model_version    # confirm
```

The stage pointer is resolved **at startup**, so a running replica keeps serving
its version until it restarts. A model swapping underneath in-flight requests
makes an incident unreadable; a rollout does not.

### Canary

Deploy a second deployment with the same image and `MLSERVICE_MODEL_VERSION=v4`,
send it a slice of traffic, and compare `mlservice_prediction_score` between the
two. The version label on every metric and every response makes the comparison
mechanical.

### Rollback

```bash
mlservice registry rollback tabular-classifier production --reason "INC-1234"
kubectl rollout restart deployment/ml-service
```

Or, if you baked the registry into the image, roll back the image tag — which is
faster and also reverts any code change that shipped alongside.

## Observability wiring

**Logs** go to stdout as JSON. Ship them with whatever you already run. Useful
fields: `request_id`, `model_version`, `route`, `status`, `duration_ms`.

**Metrics** are at `/metrics` in Prometheus format. Alerts worth having on day one:

```yaml
- alert: MLServiceNotReady
  expr: up{job="ml-service"} == 1 and mlservice_model_loaded == 0
  for: 2m

- alert: MLServiceErrorRate
  expr: |
    sum(rate(mlservice_requests_total{status=~"5.."}[5m]))
      / sum(rate(mlservice_requests_total[5m])) > 0.01
  for: 5m

- alert: MLServiceLatency
  expr: |
    histogram_quantile(0.95,
      sum by (le) (rate(mlservice_request_duration_seconds_bucket{route="/predict"}[5m]))
    ) > 0.25          # replace with your measured baseline
  for: 10m

# The one that catches a bad model rather than a bad server.
- alert: MLServicePredictionDistributionShift
  expr: |
    abs(
      histogram_quantile(0.5, sum by (le) (rate(mlservice_prediction_score_bucket[1h])))
      - histogram_quantile(0.5, sum by (le) (rate(mlservice_prediction_score_bucket[1h] offset 1d)))
    ) > 0.15          # replace with a threshold from your own history
  for: 30m

# Usually an upstream producer changed a column, not a model problem.
- alert: MLServiceSchemaViolations
  expr: sum(rate(mlservice_errors_total{kind="client_error"}[10m])) > 1
  for: 15m
```

**Tracing** is optional:

```bash
pip install '.[otel]'
MLSERVICE_OBS_TRACING_ENABLED=true
MLSERVICE_OBS_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces
MLSERVICE_OBS_TRACE_SAMPLE_RATIO=0.1
```

Enabling it without installing the extra logs a warning and continues untraced,
rather than failing the deploy.

## Performance

**This repository publishes no throughput or latency numbers.**

Not an omission. Latency for an in-process scikit-learn model is dominated by
your estimator, your feature count, your batch size, your CPU allocation, and
your co-tenants. A number measured on a maintainer's laptop would look like a
specification and would be wrong for you.

`loadtest/` ships two ready profiles (k6 and Locust) that fetch a valid instance
from `/model-info`, so they work against your model unedited. Run one against a
deployment configured the way you actually deploy, and record your numbers here:

```
| Date | Image tag | Model version | Replicas | CPU limit | Batch | p50 | p95 | p99 | RPS |
|------|-----------|---------------|----------|-----------|-------|-----|-----|-----|-----|
|      |           |               |          |           |       |     |     |     |     |
```

Then replace the placeholder thresholds in `loadtest/k6_smoke.js` and
`loadtest/locustfile.py` with your measured p95/p99 plus headroom, and re-measure
whenever the model changes — a larger estimator changes inference cost even
though not one line of service code moved.

## Pre-production checklist

- [ ] `make golden-path` passes on a fresh clone
- [ ] Gates in `configs/training.yaml` reflect a measured baseline
- [ ] `model_cards/TEMPLATE.md` says what this model may and may not decide
- [ ] `MLSERVICE_ENVIRONMENT=prod`, `MLSERVICE_OBS_LOG_FORMAT=json`
- [ ] `MLSERVICE_OBS_LOG_INCLUDE_REQUEST_BODY=false` unless you have a lawful basis
- [ ] Registry strategy chosen (baked / mounted / synced) and documented
- [ ] Readiness probe on `/ready`, liveness on `/health` — not the other way round
- [ ] Resource requests and limits set from a measurement
- [ ] Alerts wired, including the prediction-distribution one
- [ ] Load profile run against production-shaped infrastructure, numbers recorded above
- [ ] Rollback rehearsed at least once, in staging
