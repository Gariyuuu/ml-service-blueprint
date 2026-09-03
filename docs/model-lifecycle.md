# Model lifecycle

How a model gets from a CSV to production, and back out again when it misbehaves.

## The stages

```
train ──► validate ──► register ──► promote to staging ──► promote to production
              │                                                    │
              └── gates fail: no artifact is registered            └── rollback
```

## 1. Train

```bash
make data          # materialise data/raw/breast_cancer.csv
make train         # train, evaluate, register
```

`mlservice train` runs, in this order:

1. **Seed** numpy and `random` from `training.seed`.
2. **Load** the CSV named in `data.path`, dropping `drop_columns`.
3. **Split** three ways, deterministically, from `split.random_state`.
4. **Derive the feature schema from the training split only.** Using the full
   frame would bake test-set ranges and categories into the artifact's declared
   contract and its drift baseline.
5. **Build** the pipeline: `ColumnTransformer` (driven by the schema) → estimator.
6. **Fit** on train.
7. **Select the decision threshold on validation**, maximising F1 by default.
8. **Evaluate on test** — the split that influenced neither fitting nor threshold selection.
9. **Check gates** against the test metrics.
10. **Package** the artifact: pipeline, schema, metrics, provenance, model card.
11. **Register**, if gates passed.

### The split discipline, and why it is three-way

| Split | Used for | Never used for |
| --- | --- | --- |
| train | fitting the pipeline | anything you report |
| validation | selecting the decision threshold | reporting final metrics |
| test | the metrics you report and gate on | any decision during training |

Tuning the threshold on test data leaks the test set into the deployed decision
rule and makes every reported number optimistic. That is why validation exists as
a separate split rather than "just use cross-validation" — the threshold is a
deployed artifact field, not a modelling detail.

`validation_size` and `test_size` are fractions of the **original** frame. With
both at 0.2 you get 60/20/20, not 60/20/16.

### Determinism

Same config + same data file + same seed → identical scores, verified by
`tests/integration/test_training_pipeline.py::test_training_is_reproducible`.

`split.random_state` is effectively immutable once you have registered models:
changing it reshuffles every partition, so metrics from before and after are not
comparable. If you must change it, treat every prior version's metrics as
belonging to a different experiment.

## 2. Validate — the gates

```yaml
# configs/training.yaml
gates:
  min_roc_auc: 0.95
  min_f1: 0.90
```

A run that misses a gate still writes its metrics and its model card. What it
cannot do is register:

```
error: evaluation gates failed: roc_auc=0.9312 below required 0.9500
```

The refusal lives in the registry (`register()` raises `RegistryError`), not in
the CLI, so no path — a notebook, a script, an Airflow task — can register a
model that missed its bar without passing `allow_failed_gates=True` explicitly.

**Set gates from a measured baseline, not from ambition.** Train once with no
gates, look at the numbers, then set the gate slightly below what you got. A gate
above your current model blocks every deploy; a gate far below it catches nothing.

`gates.check` reports every failure at once, and flags a gate whose metric was
never computed — which catches the case where a one-class holdout silently
dropped `roc_auc` from the metric set.

## 3. Register

```
registry/tabular-classifier/
    versions/v1/  v2/  v3/
    stages.json
    transitions.jsonl
```

Versions are `v1, v2, v3…`, allocated by an exclusive `mkdir` so concurrent
trainers cannot collide. **Registered versions are immutable.** Retraining never
overwrites; it adds.

```bash
mlservice registry list
mlservice registry versions tabular-classifier
mlservice registry show tabular-classifier v2      # full metadata as JSON
```

## 4. Promote

Stages are pointers. Promotion writes one small JSON file — no bytes move.

```bash
mlservice registry promote tabular-classifier v2 staging --reason "beats v1 on recall"
mlservice registry promote tabular-classifier v2 production --reason "staging soak clean"
```

Or in one step from training:

```bash
mlservice train --promote-to staging
```

The service resolves its stage pointer **once, at startup**. Promoting does not
change what a running replica serves; a restart does. That is deliberate: a model
swapping underneath in-flight requests makes an incident unreadable.

### Suggested flow

| Stage | Who points at it | What it is for |
| --- | --- | --- |
| `staging` | a staging deployment | soak against real traffic shape |
| `production` | the production deployment | live traffic |

Stage names are arbitrary strings — add `canary` or `shadow` if you use them.

## 5. Roll back

```bash
mlservice registry rollback tabular-classifier production --reason "score drift after v3"
```

This reads `transitions.jsonl`, finds the last change to that stage, and promotes
back to its `from_version`. Then restart the deployment to pick it up.

Rollback is a pointer move, which is why it is fast and safe: the old artifact
was never deleted, never modified, and still passes its own digest check.

```bash
mlservice registry history tabular-classifier --stage production
```

```
2026-02-01T09:14:22+00:00  production   - -> v1     ci     initial
2026-02-08T11:02:10+00:00  production   v1 -> v2    alice  beats v1 on recall
2026-02-08T15:41:55+00:00  production   v2 -> v1    bob    (rollback) score drift after v3
```

Rollback refuses in two cases, both loudly: no transition history for that stage,
and a stage that has only ever pointed at one version.

## 6. Retire

```bash
mlservice registry promote tabular-classifier v5 production   # move the pointer first
python -c "
from mlservice.registry.local import LocalFilesystemRegistry
LocalFilesystemRegistry('registry').delete_version('tabular-classifier', 'v1')"
```

`delete_version` refuses while any stage still points at the version.

Keep more versions than feels necessary. The cheapest possible incident response
is `rollback`, and it only works if the artifact is still there.

## Retraining

Retraining is just training again. The registry does the rest.

```bash
make data && make train        # new version, gates enforced
mlservice registry versions tabular-classifier   # compare against the incumbent
mlservice registry promote tabular-classifier v4 staging
```

Compare like with like: metrics are only comparable across versions if
`split.random_state` and the data file are unchanged. `metadata.dataset.content_sha256`
tells you whether the data moved — two versions with different digests were
evaluated on different rows, and their metrics are not a fair comparison.

### When to retrain

The blueprint gives you the signals; the policy is yours.

| Signal | Where it comes from |
| --- | --- |
| Score distribution shifted | `mlservice_prediction_score` in Prometheus, or `ScoreDistributionDrift` over sink records |
| Input distribution shifted | per-feature comparison against `metadata.feature_schema` (min/max/categories/null rate are recorded for exactly this) |
| Outcome quality dropped | your labels, joined to prediction records by `event_id` — outside this repo |
| Schema violations rising | `mlservice_errors_total{kind="client_error"}` plus 422 details in the logs |
| Calendar | a schedule, because the world moves even when nothing alerts |

A rising 422 rate deserves a specific mention: it usually means an upstream
producer changed a column, not that the model degraded. Fixing that upstream is
cheaper than retraining around it.

## Model cards

Every artifact gets `model_card.md`, generated from its own metadata. The
provenance, data, metrics, and schema sections cannot go stale — they are read
from the artifact.

The judgement sections come from `model_cards/TEMPLATE.md`: intended use,
limitations, ethical considerations, and who gets paged. Edit that file; the next
training run picks it up.

Served live at `GET /model-card`, so the card describing the model currently
handling traffic is one request away.

## Incident checklist

A model is producing bad predictions in production.

```bash
# 1. What is actually deployed?
curl -s localhost:8000/model-info | jq '{model_version, stage, decision_threshold, git_commit, trained_at}'

# 2. What changed, and when?
mlservice registry history tabular-classifier --stage production

# 3. Was it trained from clean code?
mlservice registry show tabular-classifier v3 | jq '.provenance'

# 4. Was it trained on the data you think?
mlservice registry show tabular-classifier v3 | jq '.dataset'

# 5. Roll back, then restart the deployment.
mlservice registry rollback tabular-classifier production --reason "INC-1234"
```

Then diagnose. The rollback is cheap and reversible; the diagnosis is not urgent
once traffic is healthy again.
