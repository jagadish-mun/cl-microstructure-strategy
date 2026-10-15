PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python
CLI    := $(PY) -m clstrat.cli

export PYTHONPATH := src

.PHONY: setup data features train backtest research holdout report lint test all clean

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

train: setup
	$(CLI) train

backtest: setup
	$(CLI) backtest

research: setup
	$(CLI) research

holdout: setup
	$(CLI) holdout

report: setup
	$(CLI) report

lint: setup
	$(BIN)/ruff check .

test: setup
	$(BIN)/pytest

# Full pipeline from an empty data/ directory, then lint and tests.
all: data features train backtest research holdout report lint test

# Remove generated data and reports (keeps the committed sample report).
clean:
	rm -rf data reports/*.csv reports/*.png reports/*.txt reports/*.md reports/*.yaml
