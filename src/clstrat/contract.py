"""CL (NYMEX WTI crude oil) contract specification and regular-hours session calendar."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

import pandas as pd

from clstrat.config import Config


@dataclass(frozen=True)
class ContractSpec:
    """Static contract terms. ``tick_value`` is dollars per tick per contract."""

    symbol: str = "CL"
    tick_size: float = 0.01
    multiplier: float = 1000.0

    @property
    def tick_value(self) -> float:
        """Dollar value of one tick for one contract ($10.00 for CL)."""
        return self.tick_size * self.multiplier


CL = ContractSpec()


def _parse_time(text: str) -> time:
    hour, minute = (int(x) for x in text.split(":"))
    return time(hour, minute)


@dataclass(frozen=True)
class SessionCalendar:
    """Weekday regular-hours sessions with a weekly EIA release."""

    start: time
    end: time
    timezone: str
    eia_time: time
    eia_weekday: int

    @classmethod
    def from_config(cls, cfg: Config) -> SessionCalendar:
        """Build the calendar from the ``data`` section of the config."""
        d = cfg["data"]
        return cls(
            start=_parse_time(d["session"][0]),
            end=_parse_time(d["session"][1]),
            timezone=d["timezone"],
            eia_time=_parse_time(d["eia_time"]),
            eia_weekday=int(d["eia_weekday"]),
        )

    def trading_days(self, start_date: str | date, n_days: int) -> list[date]:
        """First ``n_days`` weekdays on or after ``start_date`` (exchange holidays ignored)."""
        days = pd.bdate_range(start=start_date, periods=n_days)
        return [d.date() for d in days]

    def session_bounds(self, day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Timezone-aware session open and close for ``day``."""
        return self.at(day, self.start), self.at(day, self.end)

    def at(self, day: date, t: time) -> pd.Timestamp:
        """Timezone-aware timestamp for wall-clock time ``t`` on ``day``."""
        return pd.Timestamp.combine(day, t).tz_localize(self.timezone)

    def is_eia_day(self, day: date) -> bool:
        """True on the weekday of the EIA Weekly Petroleum Status Report."""
        return day.weekday() == self.eia_weekday

    def session_seconds(self) -> int:
        """Length of one session in seconds."""
        start = self.start.hour * 3600 + self.start.minute * 60
        end = self.end.hour * 3600 + self.end.minute * 60
        return end - start
