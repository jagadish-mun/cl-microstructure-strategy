"""Data adapters: every source exposes ``load_quotes(day)`` and ``load_trades(day)``.

Schema (all sources):

- quotes: ``ts`` (tz-aware ns), ``bid_px``, ``bid_sz``, ``ask_px``, ``ask_sz`` and optional
  ``bid_px_2..5`` / ``bid_sz_2..5`` / ``ask_px_2..5`` / ``ask_sz_2..5``
- trades: ``ts``, ``price``, ``size``, ``side`` (+1 buy aggressor, -1 sell aggressor)
- events: ``ts``, ``type``
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


class ParquetStore:
    """One parquet file per table per day: ``<root>/<table>/<YYYY-MM-DD>.parquet``.

    This is the layout written by the synthetic generator.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def days(self) -> list[date]:
        """Days available, sorted."""
        return sorted(date.fromisoformat(p.stem) for p in (self.root / "quotes").glob("*.parquet"))

    def _load(self, table: str, day: date) -> pd.DataFrame:
        return pd.read_parquet(self.root / table / f"{day.isoformat()}.parquet")

    def load_quotes(self, day: date) -> pd.DataFrame:
        """Quotes for ``day``."""
        return self._load("quotes", day)

    def load_trades(self, day: date) -> pd.DataFrame:
        """Trades for ``day``."""
        return self._load("trades", day)

    def load_events(self, day: date) -> pd.DataFrame:
        """Scheduled events (e.g. EIA) for ``day``."""
        return self._load("events", day)
