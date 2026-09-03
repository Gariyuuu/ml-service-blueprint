<!--
Human-authored sections of the model card.

Everything above the horizontal rule in a generated card (provenance, data,
metrics, schema) is produced from artifact metadata and cannot go stale. These
sections require judgement, so they are yours to write. Edit this file; the next
training run picks it up.
-->

## Intended use

This model is the blueprint's **reference example**, not a clinical tool. It
predicts the `target` label of the Wisconsin breast cancer diagnostic dataset
from cell-nucleus measurements. Its purpose in this repository is to exercise
the training → artifact → registry → service path with a real, non-synthetic
dataset small enough to run in CI.

**Do not use this model, or this dataset, for any medical decision.**

Replace this section with:

- the decisions your model is authorised to inform
- the decisions it must not inform
- the population and time window it is valid for
- the human review step, if any, between a score and an action

## Limitations

- 569 rows. Metrics on a 114-row test split carry a confidence interval wide
  enough that small version-to-version differences are noise, not improvement.
- Features are laboratory measurements from a single 1990s study. Nothing about
  this model transfers to another measurement process.
- The decision threshold is selected to maximise F1 on the validation split,
  which weighs a false positive and a false negative equally. Most real
  deployments should not.

Replace this section with your model's known failure modes, the populations your
training data under-represents, and the conditions that should trigger a retrain.

## Ethical considerations

The reference dataset contains no personal identifiers. Your dataset probably
does. Before shipping a fork, record here:

- what personal data enters the model, and the lawful basis for it
- which protected attributes are used, proxied, or correlated
- the fairness metric you measure and its threshold
- how a person contests a decision this model influenced

## Monitoring and retraining

- Owner: _team or individual_
- Retrain trigger: _schedule, drift threshold, or data volume_
- Rollback owner: _who runs `mlservice registry rollback` at 3am_
