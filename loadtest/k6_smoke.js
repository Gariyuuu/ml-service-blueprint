// k6 load profile for the inference service.
//
//   make serve                       # in one shell
//   k6 run loadtest/k6_smoke.js      # in another
//
// This file defines a load PROFILE and PASS/FAIL thresholds. It does not claim
// any throughput number: the numbers you get depend on your model, your
// hardware, and your batch size. Run it against your own deployment and record
// the result in docs/deployment.md. The thresholds below are placeholders that
// you are expected to replace with your measured baseline plus headroom.

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://127.0.0.1:8000';
const BATCH_SIZE = parseInt(__ENV.BATCH_SIZE || '1', 10);

const predictLatency = new Trend('predict_latency_ms', true);
const schemaRejections = new Rate('schema_rejections');

export const options = {
  scenarios: {
    // Ramp gently: a cold process spends its first requests warming numpy and
    // filling the allocator, and counting those as steady state is misleading.
    ramp: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '15s', target: 5 },
        { duration: '30s', target: 20 },
        { duration: '30s', target: 20 },
        { duration: '15s', target: 0 },
      ],
      gracefulRampDown: '5s',
    },
  },
  thresholds: {
    // PLACEHOLDER VALUES — replace with your measured baseline.
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
    'http_req_duration{endpoint:predict}': ['p(95)<250', 'p(99)<500'],
    'http_req_duration{endpoint:health}': ['p(95)<50'],
  },
};

// Fetch one valid instance from the service itself, so this script works
// against ANY model without being edited.
export function setup() {
  const response = http.get(`${BASE_URL}/model-info`);
  if (response.status !== 200) {
    throw new Error(`/model-info returned ${response.status}; is the service up?`);
  }
  const info = response.json();
  return {
    instance: info.example_instance,
    modelVersion: info.model_version,
  };
}

export default function (data) {
  const instances = Array(BATCH_SIZE).fill(data.instance);

  const predict = http.post(
    `${BASE_URL}/predict`,
    JSON.stringify({ instances }),
    {
      headers: { 'Content-Type': 'application/json' },
      tags: { endpoint: 'predict' },
    },
  );

  predictLatency.add(predict.timings.duration);
  schemaRejections.add(predict.status === 422);

  check(predict, {
    'predict returns 200': (r) => r.status === 200,
    'predict returns one score per row': (r) =>
      r.status === 200 && r.json('predictions').length === BATCH_SIZE,
    'predict reports the expected model version': (r) =>
      r.status === 200 && r.json('model_version') === data.modelVersion,
    'predict echoes a request id': (r) => !!r.headers['X-Request-Id'],
  });

  const health = http.get(`${BASE_URL}/health`, { tags: { endpoint: 'health' } });
  check(health, { 'health returns 200': (r) => r.status === 200 });

  sleep(0.1);
}

export function teardown() {
  // Readiness after the run is the signal that load did not knock the model out.
  const ready = http.get(`${BASE_URL}/ready`);
  if (ready.status !== 200) {
    console.error(`service is not ready after the run: ${ready.status} ${ready.body}`);
  }
}
