# CI

`ci.yml` runs five independent jobs. They are separate on purpose: a container
build failure should not hide a lint failure, and the golden path should be
readable as its own result.

| Job | What it proves | Typical duration |
| --- | --- | --- |
| `quality` | ruff, ruff format, mypy (strict), pytest on 3.11 and 3.12 | ~2 min |
| `golden-path` | `scripts/verify_golden_path.py` — data → … → monitoring hooks | ~2 min |
| `package` | the wheel and sdist build and pass `twine check` | ~1 min |
| `container` | the image builds, boots non-root, becomes ready, and serves a prediction | ~5 min |
| `security` | `pip-audit`, lockfile freshness, Trivy filesystem scan | ~2 min |

## What CI does not do

- **It does not publish.** Add a release workflow when you have somewhere to
  publish to; a template that pushes images by default is a footgun.
- **It does not gate on model quality.** That gate lives in
  `configs/training.yaml` under `gates:` and is enforced by the registry, which
  refuses to register an artifact that missed a threshold. CI runs training, so
  a regression below your gate fails the `golden-path` job.
- **It does not run load tests.** Load numbers from a shared GitHub runner are
  noise. See `loadtest/README.md`.

## Adapting it

- Drop the 3.12 matrix entry if you only ship 3.11; keep both if your consumers
  might be on either.
- The `security` job's Trivy step is set to `exit-code: "0"` — it reports to the
  Security tab without blocking. Flip it to `"1"` once you have triaged the
  existing findings, otherwise the first unrelated CVE blocks every PR.
- `pip-audit --strict` does block. That is deliberate for a dependency set this
  small.
