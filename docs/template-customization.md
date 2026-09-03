# Customising the template

The point of this repository is that **replacing the model and the dataset should
not require rewriting any infrastructure**. This page is the map of what you
touch, in the order you touch it.

## The 15-minute version

Swapping in your own tabular binary classifier:

1. Write your data to a CSV. Point `configs/training.yaml` at it.
2. Set `data.target` to your label column.
3. Set `model.estimator` and `model.hyperparameters` in `configs/model.yaml`.
4. `make train` — read the metrics.
5. Set `gates:` slightly below what you just measured.
6. Rewrite `model_cards/TEMPLATE.md`.
7. `make golden-path`.

Everything else — schema inference, validation, versioning, promotion, the API,
metrics, the container — is unchanged.

## Files you will edit

| File | Why |
| --- | --- |
| `configs/training.yaml` | data path, target, split, gates |
| `configs/model.yaml` | estimator and hyperparameters |
| `scripts/make_dataset.py` | replace with your own extract, or delete it |
| `model_cards/TEMPLATE.md` | intended use, limitations, ethics — **do not ship the placeholder** |
| `.env.example` | if you add settings |
| `docs/deployment.md` | your measured performance numbers |
| `README.md` | your model, not the reference one |

## Files you will not edit

`src/mlservice/serving/`, `src/mlservice/registry/`, `src/mlservice/observability/`,
`Dockerfile`, `Makefile`, `.github/workflows/ci.yml`. If you find yourself
changing these to accommodate a normal tabular model, that is a gap in the
template worth reporting.

---

## 1. Your own dataset

`scripts/make_dataset.py` exists so the reference model needs no network. Yours
probably pulls from a warehouse:

```python
# scripts/make_dataset.py
import pandas as pd
from sqlalchemy import create_engine

def build_frame() -> pd.DataFrame:
    engine = create_engine(os.environ["WAREHOUSE_URL"])
    return pd.read_sql(QUERY, engine)
```

Keep the CSV boundary. The training pipeline reading a file rather than a
connection is what makes `metadata.dataset.content_sha256` meaningful: two
artifacts with the same digest were trained on byte-identical data, which is the
only version of "same data" that survives an argument.

```yaml
# configs/training.yaml
data:
  path: data/raw/your_table.csv
  target: churned
  positive_label: 1
  drop_columns: [customer_id, signup_timestamp]   # identifiers and leakage
```

`drop_columns` is your leakage guard. Anything that would not exist at prediction
time — an outcome timestamp, a downstream field, a post-hoc label — goes here.
The schema is derived after these are dropped, so a dropped column is not merely
unused: the service will **reject** a request that includes it.

### Non-CSV sources

`load_dataset` in `src/mlservice/training/data.py` is 15 lines. Point it at
Parquet if you like — but then hash the file you actually read, so provenance
stays honest.

## 2. Your own model

### Using a shipped estimator

```yaml
# configs/model.yaml
name: churn-classifier          # also the registry key
estimator: random_forest
hyperparameters:
  n_estimators: 500
  max_depth: 12
  min_samples_leaf: 20
  class_weight: balanced
calibrate: true                 # when scores are consumed as probabilities
```

`mlservice estimators` lists what is available. Hyperparameters are passed
straight to the constructor; a bad one fails at build time with the estimator's
own message, not at fit time.

### Adding an estimator (XGBoost, LightGBM, CatBoost)

Two edits.

```python
# src/mlservice/training/model_factory.py
from xgboost import XGBClassifier

_BUILDERS: dict[str, Callable[..., BaseEstimator]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "xgboost": XGBClassifier,          # <-- add
}
_SEEDED = {..., "xgboost"}             # <-- if it takes random_state
```

```python
# src/mlservice/config/model.py
EstimatorName = Literal[..., "xgboost"]
```

Add the dependency to `pyproject.toml` and run `make lock`.

Config files name estimators rather than carrying import paths on purpose: a
YAML file can then never cause arbitrary code to be imported, and the supported
set stays explicit and reviewable.

### A model that is not scikit-learn at all

The pipeline must satisfy: `fit(X, y)`, `predict_proba(X) -> (n, 2)`, and
picklability. Wrap a PyTorch or ONNX model in a small `BaseEstimator` subclass
and register it in the factory. Everything downstream — schema, registry,
service, metrics — is indifferent.

If your model has no `predict_proba`, override `ModelArtifact._score`.

### Multiclass or regression

The blueprint is binary-classification-shaped in four specific places, and they
are the ones to change:

| Place | What to change |
| --- | --- |
| `ModelArtifact._score` | takes `predict_proba[:, 1]`; return the full matrix or a point estimate |
| `training/evaluate.py` | binary metrics; swap for macro-F1, or MAE/RMSE |
| `training/pipeline.py::_binarise` | maps `positive_label` to 1/0; drop for multiclass |
| `serving/schemas.py::Prediction` | `score: float` + `label: int`; widen to a class-probability map |

The decision-threshold machinery (`choose_threshold`, `decision_threshold`,
per-request `threshold`) is binary-specific and should be removed for regression
rather than left inert.

## 3. Preprocessing

The `ColumnTransformer` is built in `training/preprocessing.py` from the
**schema**, not from re-sniffing dtypes at fit time — which is what keeps the
artifact's declared contract and its actual behaviour identical.

Configurable without code:

```yaml
preprocessing:
  numeric_imputation: median          # mean | median | most_frequent
  categorical_imputation: most_frequent
  scale_numeric: true
  one_hot_min_frequency: 0.01         # collapse rare categories
```

For a custom step — a log transform, a target encoder, a text vectoriser — add it
to the relevant branch of `build_preprocessor`. It must be a fitted transformer
so it serializes into the artifact. **Do not preprocess before `load_dataset`:**
anything you do outside the pipeline is something the service will not do, and
that is exactly the skew this design exists to prevent.

## 4. Gates

```yaml
gates:
  min_roc_auc: 0.82
  min_recall: 0.75
```

Set them from a measured baseline. Train once with all gates `null`, read the
numbers, then set each gate slightly below what you achieved. A gate above your
current model blocks every deploy; a gate far below catches nothing.

Choose gates that match the cost of being wrong. A fraud model gates on recall.
A model whose false positives cost a human review gates on precision. Gating on
accuracy with an imbalanced label is a way to ship a model that predicts the
majority class.

## 5. The service

### Adding an endpoint

```python
# src/mlservice/serving/routes.py
@router.post("/explain", response_model=ExplainResponse, tags=["inference"])
def explain(payload: PredictRequest, holder: HolderDep) -> ExplainResponse:
    artifact = holder.artifact
    validated = artifact.feature_schema.validate_frame(pd.DataFrame(payload.instances))
    ...
```

`HolderDep`, `ServiceConfigDep`, `ObservabilityConfigDep`, and `DriftReporterDep`
are the dependency aliases. Errors raised from `mlservice.artifacts.schema` or
the registry already have handlers, so your endpoint gets the standard error body
for free.

### Adding auth

There is none by design. Add a dependency:

```python
async def require_api_key(x_api_key: Annotated[str, Header()]) -> None:
    if not secrets.compare_digest(x_api_key, os.environ["MLSERVICE_API_KEY"]):
        raise HTTPException(status_code=401, detail="invalid api key")

app.include_router(router, dependencies=[Depends(require_api_key)])
```

Leave `/health`, `/ready`, and `/metrics` unauthenticated, or your probes and
your scraper will fail.

### Changing the request shape

`instances: list[dict]` is free-form because the feature set is known only at
load time, from the artifact. If your callers need a fixed typed payload,
generate a Pydantic model from `FeatureSchema` at startup — but you then have to
handle the schema changing when a new version is promoted, which is precisely
what the free-form shape avoids.

## 6. Registry backend

Subclass `ModelRegistry` and implement seven methods; `rollback`, `load_stage`,
`latest_version`, and `describe` come from the base class.

```python
class S3Registry(ModelRegistry):
    def register(self, artifact, *, version=None, allow_failed_gates=False): ...
    def list_models(self): ...
    def list_versions(self, model_name): ...
    def get_version(self, model_name, version): ...
    def load(self, model_name, version): ...
    def resolve_stage(self, model_name, stage): ...
    def promote(self, model_name, version, stage, *, reason="", actor="unknown"): ...
    def transitions(self, model_name, stage=None): ...
    def delete_version(self, model_name, version): ...
```

Then inject it: `create_app(registry=S3Registry(...))`.

Two invariants a backend must preserve, or rollback breaks: versions are
immutable once registered, and every stage change is appended to `transitions`.

## 7. Drift monitoring

Point the shipped seam at your real pipeline:

```python
# your_package/sinks.py
class KafkaSink(DriftSink):
    def __init__(self, producer, topic: str) -> None: ...
    def emit(self, record: PredictionRecord) -> None:
        self.producer.send(self.topic, record.model_dump_json().encode())
    def emit_batch(self, records): ...   # override: batch produce
    def flush(self): self.producer.flush()
```

Register it in `monitoring/reporter.py::build_sink` and add the name to
`ObservabilityConfig.drift_sink`.

Two properties `DriftReporter` gives you and that your sink must not undo:
inference stays up when the sink fails, and emission is sampled. Do not bypass
the reporter to call a sink directly from a route.

For detection, implement `DriftDetector` and run it offline against what the sink
wrote. `ScoreDistributionDrift` is a working example, including how to rebuild
the training-time baseline from artifact metadata.

## 8. Tests

The fixtures in `tests/conftest.py` are the ones to change:

- `synthetic_frame` — a small mixed-type frame with real signal. Change it to
  match your feature shape; the schema and preprocessing tests follow.
- `training_config` / `model_config` — your estimator, fast settings.
- `trained_artifact` — session-scoped, so the whole suite fits in seconds.

Tests worth keeping verbatim, whatever your model is:

| Test | What it protects |
| --- | --- |
| `test_training_is_reproducible` | determinism — the property every metric comparison rests on |
| `test_round_trip_preserves_predictions` | serialization does not change behaviour |
| `test_batch_scoring_matches_row_by_row_scoring` | batching is a throughput choice, not a behavioural one |
| `test_a_tampered_model_file_is_rejected` | the digest check works |
| `test_rollback_returns_the_stage_to_the_previous_version` | your incident plan works |
| `test_a_broken_sink_does_not_break_inference` | monitoring cannot take down inference |
| `test_metrics_label_routes_by_template_not_by_path` | cardinality safety |

## 9. Renaming the package

`mlservice` appears in `pyproject.toml` (`name`, `packages`, `scripts`, mypy
`files`), the `src/mlservice/` directory, every import, the `MLSERVICE_` env
prefix, the `mlservice_*` metric names, and the Dockerfile `CMD`.

```bash
git grep -l mlservice | xargs sed -i '' 's/mlservice/yourname/g'
git mv src/mlservice src/yourname
```

Changing the metric prefix means rewriting your dashboards. Renaming is optional;
it is often not worth it.

## Checklist before you call it yours

- [ ] `configs/training.yaml` points at your data and target
- [ ] `drop_columns` lists every identifier and leaky field
- [ ] `configs/model.yaml` names your estimator
- [ ] Gates set from a measured baseline
- [ ] `model_cards/TEMPLATE.md` rewritten — placeholder text is gone
- [ ] `README.md` describes your model
- [ ] `tests/conftest.py` fixtures match your feature shape
- [ ] `make golden-path` passes
- [ ] Load numbers measured and recorded in `docs/deployment.md`
