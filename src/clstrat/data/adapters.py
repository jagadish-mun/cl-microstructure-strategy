"""Data adapters: every source exposes ``days()``, ``load_quotes(day)`` and ``load_trades(day)``.

Schema (all sources):

- quotes: ``ts`` (tz-aware ns), ``bid_px``, ``bid_sz``, ``ask_px``, ``ask_sz`` and optional
  ``bid_px_2..5`` / ``bid_sz_2..5`` / ``ask_px_2..5`` / ``ask_sz_2..5``
- trades: ``ts``, ``price``, ``size``, ``side`` (+1 buy aggressor, -1 sell aggressor,
  0 if Lee-Ready could not classify it)
- events: ``ts``, ``type``

Contract rolls: if the raw files carry an ``expiry`` column with several contracts, the
adapters keep one front-month contract per day (volume-based roll, see ``front_month``).
Rolls therefore only happen at day boundaries, and every feature and target is computed
within a single day, so no return is ever computed across a roll.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from clstrat.config import Config


class DataSource(Protocol):
    """Interface shared by the synthetic store and real-data adapters."""

    def days(self) -> list[date]: ...

    def load_quotes(self, day: date) -> pd.DataFrame: ...

    def load_trades(self, day: date) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Synthetic store
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Lee-Ready trade classification
# ---------------------------------------------------------------------------


def tick_rule(price: pd.Series) -> np.ndarray:
    """+1 on an uptick or zero-uptick, -1 on a downtick or zero-downtick, 0 if unknown."""
    change = np.sign(price.diff().to_numpy())
    change[change == 0] = np.nan
    return pd.Series(change).ffill().fillna(0).to_numpy(dtype=np.int8)


def lee_ready(trades: pd.DataFrame, quotes: pd.DataFrame, lag_ms: float = 0) -> np.ndarray:
    """Classify trade aggressor side with the Lee-Ready (1991) algorithm.

    Quote rule against the prevailing mid (last quote strictly before ``ts - lag``): above the
    mid is a buy, below is a sell. Trades at the mid fall back to the tick rule.
    """
    q = quotes[["ts", "bid_px", "ask_px"]].assign(ts=quotes["ts"].dt.as_unit("ns"))
    q = q.sort_values("ts")
    trade_ts = trades["ts"].dt.as_unit("ns")
    probe = trades[["ts"]].assign(_t=trade_ts - pd.Timedelta(milliseconds=lag_ms))
    probe = probe.reset_index().sort_values("_t")
    matched = pd.merge_asof(
        probe, q, left_on="_t", right_on="ts", allow_exact_matches=False, suffixes=("", "_q")
    ).set_index("index").sort_index()
    mid = ((matched["bid_px"] + matched["ask_px"]) / 2).to_numpy()
    px = trades["price"].to_numpy()
    side = np.sign(px - mid)
    side = np.where(np.isnan(side), 0, side).astype(np.int8)
    at_mid = side == 0
    side[at_mid] = tick_rule(trades["price"].reset_index(drop=True))[at_mid]
    return side


# ---------------------------------------------------------------------------
# Contract rolls
# ---------------------------------------------------------------------------


def front_month(daily_volume: pd.DataFrame) -> pd.Series:
    """Front contract per day from a ``day x expiry`` volume table (expiries sortable).

    Causal volume roll: day ``d`` trades the contract with the most volume on day ``d-1``,
    restricted to expiries no earlier than the contract held on ``d-1`` (never roll back).
    The first day uses the nearest expiry.
    """
    vol = daily_volume.sort_index().reindex(sorted(daily_volume.columns), axis=1).fillna(0)
    expiries = list(vol.columns)
    held = expiries[0]
    out = {}
    for i, day in enumerate(vol.index):
        if i > 0:
            prev = vol.iloc[i - 1]
            eligible = prev[expiries[expiries.index(held) :]]
            if eligible.max() > 0:
                held = eligible.idxmax()
        out[day] = held
    return pd.Series(out, name="expiry")


# ---------------------------------------------------------------------------
# File-based adapters
# ---------------------------------------------------------------------------


def _read_table(path_stem: Path) -> pd.DataFrame:
    """Read ``<stem>.parquet``, ``<stem>.csv`` or ``<stem>.csv.gz``, whichever exists."""
    readers = ((".parquet", pd.read_parquet), (".csv", pd.read_csv), (".csv.gz", pd.read_csv))
    for suffix, reader in readers:
        path = path_stem.with_name(path_stem.name + suffix)
        if path.exists():
            return reader(path)
    raise FileNotFoundError(f"no parquet/csv file for {path_stem}")


def _to_ts(raw: pd.Series, timezone: str) -> pd.Series:
    """Integer epoch nanoseconds or datetime strings to tz-aware timestamps in ``timezone``."""
    is_epoch = pd.api.types.is_integer_dtype(raw)
    ts = pd.to_datetime(raw, unit="ns", utc=True) if is_epoch else pd.to_datetime(raw)
    if ts.dt.tz is None:
        return ts.dt.tz_localize(timezone)
    return ts.dt.tz_convert(timezone)


class CsvAdapter:
    """Generic CSV/parquet adapter for files already in (or renamed to) the common schema.

    Layout: ``<root>/quotes/YYYY-MM-DD.<ext>`` and ``<root>/trades/YYYY-MM-DD.<ext>``.
    ``ts`` may be epoch nanoseconds or a datetime string; naive values are read in
    ``timezone``. A missing trade ``side`` is filled with Lee-Ready. An ``expiry`` column
    with several contracts triggers the volume-based front-month roll.
    """

    def __init__(
        self,
        root: str | Path,
        timezone: str,
        columns: dict[str, str] | None = None,
        lee_ready_lag_ms: float = 0,
    ) -> None:
        self.root = Path(root)
        self.timezone = timezone
        self.columns = columns or {}
        self.lee_ready_lag_ms = lee_ready_lag_ms
        self._front: pd.Series | None = None

    def days(self) -> list[date]:
        """Days with a quotes file, sorted."""
        stems = {p.name.split(".")[0] for p in (self.root / "quotes").iterdir()}
        return sorted(date.fromisoformat(s) for s in stems)

    def _raw(self, table: str, day: date) -> pd.DataFrame:
        df = _read_table(self.root / table / day.isoformat()).rename(columns=self.columns)
        df["ts"] = _to_ts(df["ts"], self.timezone)
        return df.sort_values("ts", kind="stable").reset_index(drop=True)

    def _front_month(self) -> pd.Series:
        """Front contract per day, computed once from daily trade volume by expiry."""
        if self._front is None:
            volumes = {}
            for day in self.days():
                t = self._raw("trades", day)
                if "expiry" in t:
                    volumes[day] = t.groupby("expiry")["size"].sum()
            self._front = front_month(pd.DataFrame(volumes).T) if volumes else pd.Series()
        return self._front

    def _select_contract(self, df: pd.DataFrame, day: date) -> pd.DataFrame:
        if "expiry" not in df:
            return df
        keep = df["expiry"] == self._front_month()[day]
        return df[keep].drop(columns="expiry").reset_index(drop=True)

    def load_quotes(self, day: date) -> pd.DataFrame:
        """Front-month quotes for ``day`` in the common schema."""
        return self._select_contract(self._raw("quotes", day), day)

    def load_trades(self, day: date) -> pd.DataFrame:
        """Front-month trades for ``day``; missing or zero ``side`` values come from Lee-Ready."""
        trades = self._select_contract(self._raw("trades", day), day)
        side = trades["side"].to_numpy() if "side" in trades else np.zeros(len(trades))
        if (side == 0).any():
            lr = lee_ready(trades, self.load_quotes(day), self.lee_ready_lag_ms)
            side = np.where(side != 0, side, lr)
        return trades.assign(side=side.astype(np.int8))[["ts", "price", "size", "side"]]


VENDOR_COLUMNS = {
    "ts_event": "ts",
    "bid_px_00": "bid_px",
    "ask_px_00": "ask_px",
    "bid_sz_00": "bid_sz",
    "ask_sz_00": "ask_sz",
    "symbol": "expiry",
}


class VendorFileAdapter(CsvAdapter):
    """Adapter for CME top-of-book (MBP-1) and trades files exported from a commercial vendor.

    Expected files, one per day under ``<root>/quotes`` and ``<root>/trades`` (parquet or CSV):

    - quotes: ``ts_event`` (epoch ns, UTC), ``bid_px_00``, ``ask_px_00``, ``bid_sz_00``,
      ``ask_sz_00``, ``symbol`` (e.g. ``CLK4``)
    - trades: ``ts_event``, ``price``, ``size``, ``side`` (``"A"`` = ask-side aggressor, i.e. a
      buy; ``"B"`` = sell; ``"N"`` = unknown, re-classified with Lee-Ready), ``symbol``

    Downloading is left to the vendor's own client. Credentials are never stored in this
    repository: a download step should read the key from the environment variable named in
    ``data.vendor.api_key_env``. ``symbol`` is mapped to ``expiry``; CL month codes
    (F G H J K M N Q U V X Z) are converted to sortable ``YYYY-MM`` strings for the roll.
    """

    MONTH_CODES = "FGHJKMNQUVXZ"

    def __init__(self, root: str | Path, timezone: str, lee_ready_lag_ms: float = 0) -> None:
        super().__init__(root, timezone, VENDOR_COLUMNS, lee_ready_lag_ms)

    @classmethod
    def symbol_to_expiry(cls, symbol: str) -> str:
        """``CLK4`` -> ``2024-05`` (single-digit years are taken as 202x)."""
        month = cls.MONTH_CODES.index(symbol[-2]) + 1
        return f"202{symbol[-1]}-{month:02d}"

    def _raw(self, table: str, day: date) -> pd.DataFrame:
        df = super()._raw(table, day)
        if "expiry" in df:
            df["expiry"] = df["expiry"].map(self.symbol_to_expiry)
        if "side" in df:
            df["side"] = df["side"].map({"A": 1, "B": -1}).fillna(0).astype(np.int8)
        return df


def make_source(cfg: Config) -> DataSource:
    """Build the data source named by ``data.source``."""
    d = cfg["data"]
    if d["source"] == "synthetic":
        return ParquetStore(Path(cfg["paths"]["data_dir"]) / "synthetic")
    if d["source"] == "csv":
        c = d["csv"]
        return CsvAdapter(c["root"], c["timezone"], c["columns"], d["lee_ready_lag_ms"])
    if d["source"] == "vendor":
        return VendorFileAdapter(d["vendor"]["root"], d["timezone"], d["lee_ready_lag_ms"])
    raise ValueError(f"unknown data source: {d['source']}")
