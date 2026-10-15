"""Config loader, contract spec and session calendar."""

from datetime import date

import pytest
from typer.testing import CliRunner

from clstrat.cli import app
from clstrat.config import load_config, with_overrides
from clstrat.contract import CL, SessionCalendar


def test_tick_value_is_ten_dollars() -> None:
    assert CL.tick_value == pytest.approx(10.0)


def test_overrides_are_applied_and_unknown_keys_rejected() -> None:
    cfg = load_config(overrides={"strategy.theta_entry_ticks": 2.0})
    assert cfg["strategy"]["theta_entry_ticks"] == 2.0
    with pytest.raises(KeyError):
        with_overrides(cfg, {"strategy.nope": 1})


def test_calendar_weekdays_and_eia() -> None:
    cal = SessionCalendar.from_config(load_config())
    days = cal.trading_days("2024-01-05", 3)  # Friday
    assert days == [date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9)]
    assert cal.is_eia_day(date(2024, 1, 10))
    open_, close = cal.session_bounds(date(2024, 1, 10))
    assert str(open_.tz) == "America/New_York"
    assert (close - open_).total_seconds() == cal.session_seconds() == 5.5 * 3600


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "show-config" in result.output
