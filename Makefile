PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python
CLI    := $(PY) -m clstrat.cli

export PYTHONPATH := src

.PHONY: setup data features lint test all

setup: $(VENV)/.installed
$(VENV)/.installed: pyproject.toml
	@if command -v uv >/dev/null 2>&1; then \
		uv venv -q --allow-existing --python 3.11 $(VENV) && uv pip install -q --python $(PY) -e ".[dev]"; \
	else \
		$(PYTHON) -m venv $(VENV) && $(PY) -m pip install -q -e ".[dev]"; \
	fi
	touch $@

data: setup
	$(CLI) generate
	$(CLI) validate

features: setup
	$(CLI) features

lint: setup
	$(BIN)/ruff check .

test: setup
	$(BIN)/pytest

all: lint test
