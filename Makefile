# Task runner for the ML Service Blueprint.
#
# `make help` lists targets. `make golden-path` runs the whole thing end to end
# and is what CI runs; if it passes on a fresh clone, the template works.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin
IMAGE ?= ml-service-blueprint
TAG ?= local
PORT ?= 8000

MODEL_NAME ?= tabular-classifier
STAGE ?= production

.PHONY: help
help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- environment -------------------------------------------------------------

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Create the virtualenv and install the package with dev extras
	$(BIN)/pip install -e '.[dev,loadtest]'

.PHONY: install-otel
install-otel: $(BIN)/python ## Add the optional OpenTelemetry tracing extra
	$(BIN)/pip install -e '.[otel]'

.PHONY: lock
lock: ## Regenerate requirements.lock (needs uv)
	uv pip compile pyproject.toml --generate-hashes \
	  --python-version 3.11 --python-platform x86_64-unknown-linux-gnu \
	  -o requirements.lock

.PHONY: clean
clean: ## Remove caches, build output, and generated artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -rf dist build src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: clean-all
clean-all: clean ## Also remove the dataset, registry, and virtualenv
	rm -rf data var registry $(VENV)

# --- golden path -------------------------------------------------------------

.PHONY: data
data: ## Materialise the reference dataset
	$(BIN)/python scripts/make_dataset.py

.PHONY: train
train: ## Train, evaluate against gates, and register a new version
	$(BIN)/mlservice train

.PHONY: train-promote
train-promote: ## Train and promote the new version straight to $(STAGE)
	$(BIN)/mlservice train --promote-to $(STAGE)

.PHONY: promote
promote: ## Promote a version: make promote VERSION=v2 [STAGE=production]
	@test -n "$(VERSION)" || { echo "usage: make promote VERSION=v2"; exit 1; }
	$(BIN)/mlservice registry promote $(MODEL_NAME) $(VERSION) $(STAGE)

.PHONY: rollback
rollback: ## Return $(STAGE) to its previous version
	$(BIN)/mlservice registry rollback $(MODEL_NAME) $(STAGE) --reason "make rollback"

.PHONY: registry
registry: ## Show registered models, versions, and stage pointers
	$(BIN)/mlservice registry list
	@echo
	$(BIN)/mlservice registry versions $(MODEL_NAME) 2>/dev/null || true

.PHONY: serve
serve: ## Run the inference service locally
	$(BIN)/mlservice serve --port $(PORT)

.PHONY: serve-dev
serve-dev: ## Run the service with autoreload and human-readable logs
	MLSERVICE_OBS_LOG_FORMAT=console $(BIN)/mlservice serve --port $(PORT) --reload

.PHONY: golden-path
golden-path: ## Run the entire local path end to end and report what passed
	$(BIN)/python scripts/verify_golden_path.py

# --- quality -----------------------------------------------------------------

.PHONY: lint
lint: ## ruff check + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

.PHONY: format
format: ## Apply ruff formatting and safe autofixes
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

.PHONY: typecheck
typecheck: ## mypy in strict mode
	$(BIN)/mypy

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: test-fast
test-fast: ## Run the test suite, skipping slow tests
	$(BIN)/pytest -m "not slow and not container"

.PHONY: coverage
coverage: ## Run tests with a coverage report
	$(BIN)/pytest --cov --cov-report=term-missing --cov-report=xml

.PHONY: audit
audit: ## Check installed dependencies for known vulnerabilities
	$(BIN)/pip-audit --strict --desc

.PHONY: check
check: lint typecheck test ## lint + typecheck + test

# --- packaging and container -------------------------------------------------

.PHONY: build
build: ## Build the wheel and sdist
	$(BIN)/python -m build

.PHONY: docker-build
docker-build: ## Build the container image (requires a registry: run `make train-promote` first)
	@test -d registry || { echo "no registry/ directory; run 'make train-promote' first"; exit 1; }
	docker build -t $(IMAGE):$(TAG) .

.PHONY: docker-run
docker-run: ## Run the container image on $(PORT)
	docker run --rm -p $(PORT):8000 \
	  -e MLSERVICE_MODEL_NAME=$(MODEL_NAME) \
	  -e MLSERVICE_MODEL_STAGE=$(STAGE) \
	  $(IMAGE):$(TAG)

.PHONY: docker-smoke
docker-smoke: ## Build, boot, probe, and tear down the container
	$(BIN)/pytest tests/container -m container -v

# --- load testing ------------------------------------------------------------

.PHONY: loadtest
loadtest: ## k6 smoke against a running service (needs k6 and `make serve`)
	k6 run loadtest/k6_smoke.js

.PHONY: loadtest-locust
loadtest-locust: ## Locust UI against a running service
	$(BIN)/locust -f loadtest/locustfile.py --host http://127.0.0.1:$(PORT)
