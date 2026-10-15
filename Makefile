PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin
CLI    := $(BIN)/clstrat

.PHONY: setup lint test all

setup: $(BIN)/activate
$(BIN)/activate: pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e ".[dev]"
	touch $(BIN)/activate

lint: setup
	$(BIN)/ruff check .

test: setup
	$(BIN)/pytest

all: lint test
