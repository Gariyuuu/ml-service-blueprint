# ML Service Blueprint

A reusable GitHub template for taking a machine-learning model from training to a
**validated production service**.

The value here is production engineering, not modelling. The example model is a
deliberately small tabular classifier so that nothing distracts from the parts
that are actually hard: the artifact contract, versioning, rollback, schema
enforcement at the API boundary, observability, and a container you would be
willing to page someone about.

There is no frontend, no dashboard, and no authentication product.

---

## The golden path

```
data → training → validation → artifact → version → inference service → Docker → CI → monitoring hooks
```

Clone it and run the whole thing:

```bash
make install        # virtualenv + package + dev extras
make golden-path    # runs every stage above and reports exactly what passed
```

`make golden-path` is not a smoke test with a happy ending. It executes each
stage, verifies the result, and prints PASS / FAIL / **SKIP with the reason**.
A step whose prerequisites are missing (no Docker daemon, no `k6` binary) is
reported as skipped and named as unverified — never quietly counted as a success.

```
Environment
  PASS package imports and Python version
Configuration
  PASS typed configs load and validate
Data
  PASS dataset materialises
Training
  PASS pipeline trains a model
Validation
  PASS metrics clear the configured gates
Artifact
  PASS artifact contract is complete and round-trips
  PASS tampered artifacts are rejected
Version / registry
  PASS artifact registers with a version
  PASS list / promote / rollback / history
Inference service
  PASS all endpoints respond correctly
Observability
  PASS logs, request ids, and metrics
Drift hooks
  PASS sink, detector, and fail-open behaviour
CI checks
  PASS lint, typecheck, tests
  PASS package builds
Container
  SKIP image builds, boots, and serves    docker is not installed on this machine
Load testing
  PASS load-test configuration present

15 passed, 0 failed, 1 skipped

Not verified on this machine (prerequisites absent, not failures):
  - image builds, boots, and serves: docker is not installed on this machine

GOLDEN PATH VERIFIED: Environment → Configuration → Data → Training → Validation
  → Artifact → Version / registry → Inference service → Observability → Drift hooks
  → CI checks → Load testing
```

Step by step, if you prefer:

```bash
make data              # materialise the reference dataset (no network needed)
make train             # train, evaluate against gates, register a version
make registry          # inspect versions and stage pointers
make promote VERSION=v1
make serve             # http://127.0.0.1:8000/docs
```

```bash
curl -s localhost:8000/model-info | jq '{model_version, decision_threshold, n_features}'
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d "{\"instances\": [$(curl -s localhost:8000/model-info | jq -c .example_instance)]}" | jq
```

---

## Verification status

This table is the honest state of this release. It distinguishes what is
**implemented**, what has been **executed and verified on the author's machine**,
and what is **implemented but not yet executed** because the prerequisite tooling
was unavailable.

| Area | Implemented | Locally verified | Notes |
| --- | --- | --- | --- |
| Typed configuration | yes | yes | |
| Training pipeline, determinism, gates | yes | yes | |
| Artifact contract, digest check, round-trip | yes | yes | |
| Registry: version / promote / rollback / history | yes | yes | |
| FastAPI service, schema enforcement, batch | yes | yes | via TestClient **and** a live uvicorn server |
| CLI (`train`, `predict`, `registry`, `serve`) | yes | yes | |
| Structured logs, Prometheus metrics | yes | yes | |
| OpenTelemetry tracing (optional extra) | yes | partly | the disabled/no-op path is tested; a live OTLP export was not exercised |
| Drift hooks: sink, detector, fail-open | yes | yes | |
| Test suite, ruff, strict mypy, package build | yes | yes | |
| `make golden-path` | yes | yes | 15 passed, 0 failed, 1 skipped |
| **Dockerfile and container smoke tests** | **yes** | **NO — pending** | **Docker is not installed on the machine this was built on. The image has never been built and the container tests have never run.** |

### The Docker gap, precisely

`Dockerfile`, `tests/container/test_container_smoke.py`, and the golden path's
container step are written but **have never been executed**. They are not known
to be broken; they are unverified, which is a different and weaker claim.

Both the test suite and the golden path detect the missing daemon and report
`SKIP` with the reason, rather than passing vacuously:

```
Container
  SKIP image builds, boots, and serves    docker is not installed on this machine
```

On a Docker-enabled machine, this is the command that closes the gap:

```bash
make train-promote && make docker-build && make docker-smoke
```

Update this table once it passes.

---

## The one idea

Everything the service needs to score a request — the fitted estimator, the
fitted preprocessor, the feature schema, the decision threshold, the metrics it
was accepted on, and the code that produced it — lives inside **one immutable
artifact**. The service reimplements none of it.

That closes off training/serving skew structurally rather than by discipline:

| Usual cause of skew | Why it cannot happen here |
| --- | --- |
| Preprocessing rewritten in the service | The preprocessor is *inside* the serialized pipeline. There is no serving-side feature code. |
| Columns in a different order, or the wrong dtype | Every request is reordered and coerced against the schema frozen at training time. |
| Threshold drifting from the validated one | The threshold is a field of the artifact, chosen on validation, served from metadata. |

A visible consequence: `mlservice predict data.csv` (offline) and `POST /predict`
run the same code path. They cannot disagree.

---

## What you get

**Typed configuration**, split by how often it changes — training and model
config are YAML, versioned with the experiment and snapshotted into every
artifact; service and observability config are environment-first. Unknown keys
in a YAML file are a load-time error, not a silent fallback to a default.

**A training pipeline** with a deterministic three-way split, schema derived from
the training rows only, threshold selected on validation (never on test), and
configurable **evaluation gates** that the registry enforces — a model that
misses its bar cannot be registered.

**An artifact contract**. Every version carries its pipeline, feature schema,
test and validation metrics, training timestamp, dataset SHA-256, git commit and
dirty flag, library versions, full config snapshot, and a generated model card.
The model file's digest is checked on every load.

**A registry** that is a directory: immutable versions, cheap stage pointers, and
an append-only transition log that makes `rollback` a pointer move rather than a
rebuild. `ModelRegistry` is an ABC — swap in MLflow or S3 by writing one class.

**A FastAPI service** with `/health`, `/ready`, `/predict`, `/model-info`,
`/model-card`, and `/metrics`. Batch prediction, per-request threshold override,
one error shape for every failure, and full schema validation that reports every
problem in a single 422.

**Observability**: JSON logs with request-scoped context, seven Prometheus
metrics including a prediction-score histogram, and optional OpenTelemetry
tracing that degrades to a no-op when the extra is not installed.

**Drift hooks**, not a drift product: a `DriftSink` interface with three
reference implementations, a `DriftDetector` interface with a PSI reference, and
a reporter that is sampled and **fail-open** — a broken monitoring pipeline
leaves `/predict` at 200.

**A production Dockerfile**: multi-stage, non-root (uid 10001, no shell, no home),
hash-locked dependencies, `HEALTHCHECK` on liveness (not readiness — a restart
cannot fix a missing model). Written but **not yet executed** — see
[Verification status](#verification-status).

**CI** across five jobs: lint, strict mypy, tests on 3.11 and 3.12, the golden
path, a package build, a container build with a live smoke test, and dependency
and lockfile-freshness checks.

**Load-test profiles** for k6 and Locust that fetch a valid instance from
`/model-info`, so they work against any model unedited — with **no fabricated
throughput numbers** anywhere in this repository. See below.

---

## Repository layout

```
src/mlservice/
    config/         typed config: training, model, service, observability
    training/       split, preprocess, fit, evaluate, gate, model card
    artifacts/      the artifact contract: schema, metadata, provenance, IO
    registry/       ModelRegistry ABC + local filesystem backend
    serving/        FastAPI app, request models, model holder, error handlers
    observability/  structured logging, Prometheus metrics, optional tracing
    monitoring/     drift extension points, reference sink and detector
    cli.py          mlservice train | predict | serve | registry …
tests/
    unit/ integration/ api/ container/
configs/            training.yaml, model.yaml, service.example.yaml
scripts/            make_dataset.py, verify_golden_path.py
loadtest/           k6_smoke.js, locustfile.py
docs/               architecture, deployment, model-lifecycle, template-customization
model_cards/        TEMPLATE.md — the human-authored sections
Dockerfile  Makefile  pyproject.toml  requirements.lock  .env.example
```

---

## Replacing the example model

This is the point of the template, and it should take about fifteen minutes.

1. Write your table to a CSV; point `configs/training.yaml` at it.
2. Set `data.target`, and list identifiers and leaky fields in `data.drop_columns`.
3. Set `model.estimator` and `model.hyperparameters` in `configs/model.yaml`
   (`mlservice estimators` lists what is available; adding XGBoost or LightGBM is
   two lines in `training/model_factory.py`).
4. `make train`, read the metrics.
5. Set `gates:` slightly below what you measured.
6. Rewrite `model_cards/TEMPLATE.md` — intended use, limitations, ethics.
7. `make golden-path`.

You do not touch `serving/`, `registry/`, `observability/`, the `Dockerfile`, the
`Makefile`, or CI. Schema inference, request validation, versioning, promotion,
rollback, metrics, and the container all adapt to your feature set automatically,
because they read it from the artifact rather than hard-coding it.

Full detail, including multiclass/regression, custom estimators, custom registry
backends, and auth: **[docs/template-customization.md](docs/template-customization.md)**.

---

## The reference model

`scikit-learn`'s bundled Wisconsin breast cancer diagnostic dataset: 569 rows,
30 numeric features, binary label. Two properties make it the right
*infrastructure* fixture — it needs no network, so CI, containers, and offline
clones behave identically; and a full train → evaluate → register cycle takes
under a second, so `make golden-path` is something you actually run.

It is a real dataset with real column names, not synthetic filler. A run of the
shipped config on this machine:

```
split          341 train / 114 validation / 114 test
threshold      0.180  (selected on validation, maximising F1)
test roc_auc   0.9868
test f1        0.9664
training time  0.44s
```

Your numbers will differ with library versions. **This is a demonstration
fixture, not a clinical tool** — see the model card.

---

## Performance

**This repository publishes no throughput or latency numbers, deliberately.**

Latency for an in-process scikit-learn model is dominated by your estimator, your
feature count, your batch size, your CPU allocation, and your co-tenants. A
number measured on a maintainer's laptop would read as a specification and would
be wrong for you.

`loadtest/` ships two complete profiles with **placeholder thresholds, marked as
such in the source**. Run one against infrastructure configured the way you
actually deploy, record the result in `docs/deployment.md`, and replace the
placeholders with your measured p95/p99 plus headroom. See
[loadtest/README.md](loadtest/README.md).

---

## Requirements

- Python 3.11 or 3.12
- Docker, for the container stage only — everything else runs without it, and both
  the test suite and the golden path report the container step as skipped, with
  the reason, rather than failing or passing vacuously. That stage is currently
  **unverified**; see [Verification status](#verification-status)
- `uv`, for `make lock` only

---

## Documentation

| Document | Read it when |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | you want to know why it is shaped this way |
| [docs/model-lifecycle.md](docs/model-lifecycle.md) | you are training, promoting, rolling back, or in an incident |
| [docs/deployment.md](docs/deployment.md) | you are shipping it (Kubernetes manifest, alerts, checklist) |
| [docs/template-customization.md](docs/template-customization.md) | you are replacing the model or the dataset |
| [loadtest/README.md](loadtest/README.md) | you are establishing a performance baseline |
| [.github/workflows/README.md](.github/workflows/README.md) | you are adapting CI |

---

## Common commands

```bash
make help                       # every target, described

make install                    # venv + package + dev extras
make data                       # materialise the dataset
make train                      # train, gate, register
make train-promote              # …and promote straight to production
make registry                   # models, versions, stage pointers
make promote VERSION=v2         # move the production pointer
make rollback                   # return production to its previous version
make serve                      # run the service
make serve-dev                  # …with autoreload and readable logs

make check                      # lint + typecheck + test
make test-fast                  # skip slow and container tests
make coverage
make audit                      # pip-audit against installed dependencies
make golden-path                # the whole path, reported honestly

make docker-build               # requires a registry: run make train-promote first
make docker-run
make docker-smoke               # build, boot, probe, tear down
make loadtest                   # k6 against a running service
```

---

## License

MIT.
