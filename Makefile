PY ?= .venv/bin/python

.PHONY: setup lint format typecheck test check env-capture

setup:
	python3.12 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -c constraints.txt -e '.[dev]'

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

typecheck:
	$(PY) -m mypy

test:
	$(PY) -m pytest -m "not gpu"

check: lint typecheck test

# Writes versions and hardware info to results/env-<commit>.json
env-capture:
	$(PY) -m bench.env_capture --out results
