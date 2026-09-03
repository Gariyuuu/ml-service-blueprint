# Architecture

## The one idea

Everything the service needs to score a request — the fitted estimator, the
fitted preprocessor, the feature schema, the decision threshold, the metrics it
was accepted on, and the code that produced it — lives inside a **single
immutable artifact**. The service reimplements none of it.

That is the whole design. Every other decision here follows from it.

```
                 configs/*.yaml  (typed, versioned with the code)
                        │
                        ▼
  data/raw/*.csv ──► training pipeline ──► ModelArtifact ──► registry
                     │  deterministic      │  pipeline        │  v1, v2, v3 …
                     │  split              │  schema          │  stage pointers
                     │  fit                │  metrics         │  transition log
                     │  evaluate           │  provenance      │
                     │  gate               │  model card      │
                                                              │
                                            resolve stage ────┘
                                                    │
                                                    ▼
                                             FastAPI service
                                             ├─ /health   liveness
                                             ├─ /ready    readiness
                                             ├─ /predict  validate → score
                                             ├─ /model-info
                                             └─ /metrics
                                                    │
                                     ┌──────────────┴──────────────┐
                                     ▼                             ▼
                              observability                  drift sink
                              logs / metrics / traces     (extension point)
```

## Why training/serving skew is structurally impossible here

Skew — the model behaving differently in production than in the notebook — is
the defining failure mode of deployed ML. It has three usual causes, and each
one is closed off by construction rather than by discipline:

| Cause | How this design closes it |
| --- | --- |
| Preprocessing reimplemented in the service | The preprocessor is *inside* the serialized pipeline. There is no serving-side feature code to diverge. |
| Columns arriving in a different order or dtype | `FeatureSchema.validate_frame` reorders and coerces every request against the schema frozen at training time. |
| The threshold drifting from the one that was validated | The threshold is a field of the artifact, selected on the validation split and served from metadata. |

The consequence worth naming: **`mlservice predict` (offline CSV) and `POST /predict`
run the same code path.** If a batch job and the service ever disagree, it is a
bug in one shared function, not an integration difference — which is a far easier
class of problem.

## Module map

| Package | Responsibility | Depends on |
| --- | --- | --- |
| `config/` | Typed settings, split by lifecycle concern | nothing |
| `artifacts/` | The artifact contract: schema, metadata, provenance, save/load | `config` |
| `training/` | Split, preprocess, fit, evaluate, gate, render card | `config`, `artifacts`, `registry` |
| `registry/` | Version allocation, stage pointers, transition log | `artifacts` |
| `serving/` | FastAPI app, request models, model holder, error shape | all of the above |
| `observability/` | Structured logs, Prometheus metrics, optional tracing | `config` |
| `monitoring/` | Drift extension points, one reference sink and detector | `artifacts`, `observability` |

Dependencies point one way. `artifacts` does not know a registry exists;
`registry` does not know a service exists. That is what lets you replace the
registry backend, or serve an artifact from a batch job with no HTTP anywhere,
without touching the layers below.

## Configuration is split by how often it changes

Four config objects, and the split is not cosmetic:

| Config | Source | Changes when | Frozen into the artifact? |
| --- | --- | --- | --- |
| `TrainingConfig` | `configs/training.yaml` | you change the experiment | yes |
| `ModelConfig` | `configs/model.yaml` | you change the model | yes |
| `ServiceConfig` | `MLSERVICE_*` env | you deploy | no |
| `ObservabilityConfig` | `MLSERVICE_OBS_*` env | you change monitoring | no |

Training config is file-backed and snapshotted into every artifact's metadata,
so an old model's exact settings are always recoverable. Service config is
environment-first because it differs per deployment and must not require a
rebuild to change.

Both YAML configs reject unknown keys. A typo in `configs/training.yaml` fails
at load with the offending key named, rather than silently reverting to a
default and producing a subtly different model.

## The artifact contract

An artifact is a directory of three files:

```
registry/<model>/versions/v3/
    model.joblib     the fitted sklearn Pipeline (preprocessor → estimator)
    metadata.json    ArtifactMetadata
    model_card.md    generated model card
```

`metadata.json` carries:

- **schema** — every input column, its kind, its observed range or categories,
  and its null rate at training time (which doubles as a drift baseline)
- **version**, **created_at**, **training_duration_seconds**
- **metrics** on test, **validation_metrics**, and any **gate_failures**
- **dataset** — path, SHA-256 of the training file, row counts per split, class balance
- **provenance** — git commit, branch, dirty flag, Python version, platform, and
  the versions of scikit-learn, numpy, pandas, joblib, and mlservice
- **training_config** and **model_config_snapshot** — the full config, verbatim
- **model_file_sha256** — checked on every load

The pipeline is stored as one object rather than two so the preprocessor and
estimator cannot be reassembled out of sync. They remain individually reachable
via `artifact.preprocessor` and `artifact.estimator`.

### The digest is not decoration

`ModelArtifact.load` recomputes the model file's SHA-256 and refuses to load if
it does not match. Pickles execute code on load; a registry directory is often a
mounted volume or an object-store sync. The digest turns "someone replaced the
model file" from an invisible event into a startup failure.

## Registry design

`ModelRegistry` is an ABC. `LocalFilesystemRegistry` is the only implementation
shipped, and it is a directory:

```
registry/<model>/
    versions/v1/ v2/ v3/       immutable artifact directories
    stages.json                {"production": "v2", "staging": "v3"}
    transitions.jsonl          append-only stage-change log
```

Three properties make this worth having over "just save a pickle somewhere":

1. **Versions are immutable.** `register` allocates a new directory with an
   exclusive `mkdir` and never overwrites. Two trainers racing cannot collide.
2. **Stage pointers are separate from artifacts.** Promotion writes one small
   JSON file. Rolling back does not move a model, copy bytes, or rebuild anything.
3. **Every stage change is logged.** `rollback` is implemented on the base class
   as pure history arithmetic — read the last transition for that stage, promote
   back to its `from_version`. Any backend that logs transitions correctly gets
   a correct rollback for free.

### Adding a real backend

Subclass `ModelRegistry`, implement the seven abstract methods, and point
`ServiceConfig.registry_root` (or your own field) at it. `rollback`,
`load_stage`, `latest_version`, and `describe` are inherited.

The local backend's honest limits: its only concurrency guarantee is the
exclusive `mkdir` for version allocation. `stages.json` is written atomically
(temp file + rename) so a reader never sees a partial write, but two simultaneous
promotions to the *same* stage can still race, last-writer-wins. That is fine for
one build host and not fine for a shared multi-team registry. When you outgrow
it, that is the signal to write the MLflow or object-store subclass.

## Service design

### Two probes, two questions

`/health` is liveness: is the process alive? It deliberately does **not** check
model state. A liveness probe that fails on a missing model causes an
orchestrator to restart-loop a pod that a restart cannot fix.

`/ready` is readiness: can this replica serve? It returns 503 when no model is
loaded, which keeps a half-started pod out of the load balancer while leaving it
running and inspectable.

`fail_fast_on_missing_model` (default true) decides which failure you want at
startup: crash the deploy, or come up unready. Crashing is right for a normal
deploy — a missing model is a deploy bug. Coming up unready is right when the
registry is mounted asynchronously.

### The request path

```
POST /predict
  → RequestContextMiddleware: assign request id, start timer
  → Pydantic: is this a well-formed PredictRequest?
  → batch size check                                  → 413
  → FeatureSchema.validate_frame: reorder, coerce     → 422 with every error listed
  → pipeline.predict_proba
  → threshold → labels
  → metrics: counter, score histogram, batch size
  → DriftReporter.report (sampled, fail-open)
  → PredictResponse, tagged with the exact model version
```

Two details that matter more than they look:

**Validation collects every error before raising.** A caller with four wrong
columns gets all four in one 422, not a fix-one-rerun loop.

**The response names the model version that scored it.** When a caller reports a
bad prediction, the version is in their log line, not something you infer from
deploy timestamps.

### Error shape

Every failure — schema violation, missing model, oversized batch, unhandled
exception — returns the same body:

```json
{"error": "feature_schema_violation", "message": "...", "request_id": "...", "details": ["..."]}
```

`error` is a stable machine-readable code. Unhandled exceptions return a generic
message on purpose: internal exception text leaks file paths and feature values.
The `request_id` is the join key into the logs, where the full traceback is.

## Observability

**Logs** are JSON, one object per line, with request id and model version pulled
from contextvars rather than threaded through call signatures. uvicorn's loggers
are re-routed through the same formatter so access logs are JSON too.

**Metrics** live in their own `CollectorRegistry`, not the process-global
default, so tests can build a fresh set without duplicate-timeseries errors.
Seven series:

| Metric | Why it is here |
| --- | --- |
| `mlservice_requests_total` | request count by route and status |
| `mlservice_request_duration_seconds` | latency histogram |
| `mlservice_errors_total` | errors by class |
| `mlservice_predictions_total` | rows scored (a batch of 50 adds 50) |
| `mlservice_prediction_score` | **score distribution — the drift signal** |
| `mlservice_batch_size` | how callers actually batch |
| `mlservice_model_loaded` | which version this replica is serving |

`mlservice_prediction_score` is the one that earns its keep. A model degrades
silently while latency and error rate stay flat; the score histogram moves first.

Routes are labelled with the **route template**, never the raw path. A metric
labelled with raw paths grows one timeseries per distinct URL, which is how a
scanner takes down a metrics backend.

**Tracing** is optional (`pip install '.[otel]'`). Every function in
`observability/tracing.py` degrades to a no-op when the packages are absent, so
the service's import graph does not depend on OpenTelemetry.

## Drift monitoring: extension points, not a product

`monitoring/` defines two seams and refuses to grow past them:

**`DriftSink`** — where scored predictions go. Three references ship: `NullSink`
(default), `LoggingSink`, `JsonlSink`. Write one class to send records to Kafka,
BigQuery, or a vendor SDK.

**`DriftDetector`** — how records become signals. One reference implementation
ships: `ScoreDistributionDrift`, computing PSI against the score histogram the
trainer recorded in artifact metadata. It is meant to run offline, against what
a sink wrote, not inline in the request path.

Between them sits `DriftReporter`, which the serving path uses and which
enforces two properties the route handler must not be trusted to remember:

- **Fail-open.** A sink that raises leaves `/predict` returning 200. Monitoring
  degrades to "unmonitored", never to "down". This is verified by a test.
- **Sampled.** `drift_sample_ratio` controls what fraction is emitted, because
  at volume the emit is the expensive part.

Feature values are withheld from records by default. Inference inputs are
frequently personal data, and a drift pipeline that quietly duplicates them into
a log file is a compliance incident, not a feature.

## What this blueprint deliberately does not do

- **No feature store.** Requests carry their own features. Adding one changes the
  service's failure modes fundamentally; that belongs in your fork, not here.
- **No online learning.** Artifacts are immutable. A new model is a new version.
- **No multi-model serving.** One process serves one artifact. Serve several by
  running several deployments — it keeps memory, metrics, and rollback per-model.
- **No auth.** Put the service behind your gateway or mesh. An auth layer baked
  into a template is one you will have to rip out.
- **No drift detection engine.** See above.
