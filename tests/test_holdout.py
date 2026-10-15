"""The holdout uses the last days only, trains on the days just before, and runs once."""

import pandas as pd
import yaml

from clstrat.config import load_config
from clstrat.research.holdout import holdout_fold, run_holdout


def test_holdout_fold_uses_last_days_and_preceding_training_window() -> None:
    cfg = load_config()
    days = list(pd.date_range("2024-01-01", periods=60))
    fold = holdout_fold(days, cfg)
    assert fold.test_days == tuple(days[-10:])
    assert fold.train_days == tuple(days[30:50])


def test_holdout_is_never_rerun(tmp_path) -> None:
    cfg = load_config(overrides={"paths.data_dir": str(tmp_path)})
    (tmp_path / "models").mkdir()
    stored = {"backtest": {"net_pnl": 123.0}}
    (tmp_path / "models" / "holdout.yaml").write_text(yaml.safe_dump(stored))
    out = run_holdout(cfg)  # would fail without inputs if it tried to recompute
    assert out["backtest"]["net_pnl"] == 123.0 and "already run" in out["note"]
