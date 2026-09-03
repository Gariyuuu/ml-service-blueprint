# Load testing

Two equivalent profiles are provided; use whichever your team already runs.

| | k6 | Locust |
| --- | --- | --- |
| File | `k6_smoke.js` | `locustfile.py` |
| Install | `brew install k6` | `pip install -e '.[loadtest]'` |
| Run | `k6 run loadtest/k6_smoke.js` | `make loadtest-locust` |
| CI-friendly | yes (exit code from thresholds) | yes (`--headless`, exit code from the `quitting` hook) |

Both scripts fetch a valid instance from `/model-info` at start, so they work
against any model in this template without being edited.

## This repository publishes no throughput numbers

The thresholds in both files are **placeholders**, marked as such in the source.
Latency and throughput for an in-process scikit-learn model are dominated by
things this repository knows nothing about: your estimator, your feature count,
your batch size, your CPU, and whether the service shares a core with anything
else. A number measured on the maintainer's laptop would be worse than no number
at all, because it would look like a specification.

## Establishing your own baseline

1. Deploy the service the way you actually deploy it — same image, same CPU
   limit, same replica count. Numbers from `make serve` on a laptop do not
   transfer to a container with a 500m CPU limit.
2. Run the profile at a load your service comfortably handles, and record
   p50/p95/p99 and error rate.
3. Replace the placeholder thresholds with your measured p95 and p99 plus
   headroom (a common starting point is p95 × 1.5).
4. Write the numbers, the date, and the hardware into `docs/deployment.md`.
5. Re-measure whenever the model changes. A larger estimator changes inference
   cost even though not one line of service code moved.

## Knobs

| Variable | Both | Meaning |
| --- | --- | --- |
| `BASE_URL` / `--host` | ✓ | Target service |
| `BATCH_SIZE` | ✓ | Rows per `/predict` call |

Batch size is the most important variable and the one most often left at 1.
Per-row cost falls sharply with batching because the fixed per-request overhead
(JSON parsing, frame construction, validation) is amortised. If your callers can
batch, measure batched: it is usually the difference between needing one replica
and needing ten.

## Watching the right things during a run

Scrape `/metrics` while the load runs:

- `mlservice_request_duration_seconds` — server-side latency, excluding transport.
  Compare it against the client-side number to see how much is network.
- `mlservice_prediction_score` — the score histogram. It must not shift when the
  only thing that changed is load. If it does, something is non-deterministic.
- `mlservice_batch_size` — confirms the load generator is sending what you think.
- `mlservice_errors_total` — a `client_error` spike under load usually means your
  generator is malformed, not that the service broke.
