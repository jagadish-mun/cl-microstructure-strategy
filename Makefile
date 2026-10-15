PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python
CLI    := $(PY) -m clstrat.cli

export PYTHONPATH := src

.PHONY: setup data lint test all

setup: $(BIN)/activate
$(BIN)/activate: pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e ".[dev]"
	touch $(BIN)/activate

data: setup
	$(CLI) generate
	$(CLI) validate

lint: setup
	$(BIN)/ruff check .

test: setup
	$(BIN)/pytest

all: lint test
