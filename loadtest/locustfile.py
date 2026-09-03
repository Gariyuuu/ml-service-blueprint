"""Locust load profile — the Python alternative to loadtest/k6_smoke.js.

    make serve                                              # in one shell
    locust -f loadtest/locustfile.py --host http://127.0.0.1:8000

Like the k6 script, this defines a profile, not a benchmark result. It fetches a
valid instance from ``/model-info`` at start, so it works against any model
without editing. Measure your own numbers; do not copy anyone else's.
"""

from __future__ import annotations

import os

from locust import HttpUser, between, events, task

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))


class InferenceUser(HttpUser):
    """A client that mostly predicts and occasionally checks health."""

    wait_time = between(0.05, 0.2)

    def on_start(self) -> None:
        response = self.client.get("/model-info", name="/model-info")
        response.raise_for_status()
        info = response.json()
        self.instance = info["example_instance"]
        self.model_version = info["model_version"]

    @task(20)
    def predict(self) -> None:
        payload = {"instances": [self.instance] * BATCH_SIZE}
        with self.client.post("/predict", json=payload, name="/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}: {r.text[:200]}")
                return
            body = r.json()
            if len(body["predictions"]) != BATCH_SIZE:
                r.failure(f"expected {BATCH_SIZE} predictions, got {len(body['predictions'])}")
            elif body["model_version"] != self.model_version:
                # A version change mid-run means a deploy happened underneath
                # you; the numbers before and after are not comparable.
                r.failure(f"model version changed to {body['model_version']}")

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")


@events.quitting.add_listener
def _fail_on_bad_run(environment, **_: object) -> None:
    """Make a bad run exit non-zero so CI can gate on it."""
    stats = environment.stats.total
    if stats.num_requests == 0:
        environment.process_exit_code = 1
        return

    # PLACEHOLDER p95 — replace with your measured baseline plus headroom.
    too_slow = stats.get_response_time_percentile(0.95) > 250
    if stats.fail_ratio > 0.01 or too_slow:
        environment.process_exit_code = 1
