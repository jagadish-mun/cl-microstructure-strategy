"""Execution model: latency, market-order fills against the book, slippage and fees.

An order decided at bar ``t`` fills against the book as of ``t + latency``. Buys take the
ask, sells the bid. Size beyond the L1 queue walks L2..L5 when depth is available
(``slippage_model: l2_or_sqrt``); otherwise, or beyond the deepest level, the remainder pays
square-root impact ``coef * sqrt(remaining / L1 size)`` ticks on top of the last price used.

Costs are reported as separate dollar line items relative to the mid at fill time:
``spread_cost`` (L1 price vs mid), ``slippage`` (fill price vs L1 price) and ``fee``
(half the round-trip fee per contract on each side).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from clstrat.config import Config
from clstrat.contract import CL


@dataclass(frozen=True)
class Book:
    """One session's 1 s book as arrays; level arrays have shape ``(n_bars, n_levels)``."""

    ts: pd.DatetimeIndex
    mid: np.ndarray
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    quote_age: np.ndarray  # seconds since the last quote update at each bar end

    @classmethod
    def from_bars(cls, bars: pd.DataFrame) -> Book:
        """Collect L1 and any ``_2.._5`` depth columns from 1 s bars."""
        suffixes = [""] + [f"_{k}" for k in range(2, 6) if f"bid_px_{k}" in bars]

        def levels(prefix: str) -> np.ndarray:
            return np.column_stack([bars[f"{prefix}{s}"].to_numpy(float) for s in suffixes])

        return cls(
            ts=pd.DatetimeIndex(bars.index),
            mid=bars["mid"].to_numpy(float),
            bid_px=levels("bid_px"),
            bid_sz=levels("bid_sz"),
            ask_px=levels("ask_px"),
            ask_sz=levels("ask_sz"),
            quote_age=bars["quote_age"].to_numpy(float)
            if "quote_age" in bars
            else np.zeros(len(bars)),
        )

    def __len__(self) -> int:
        return len(self.mid)

    def spread_ticks(self, i: int) -> float:
        return round((self.ask_px[i, 0] - self.bid_px[i, 0]) / CL.tick_size)


@dataclass(frozen=True)
class Fill:
    """An executed market order. Costs are non-negative dollars."""

    ts: pd.Timestamp
    qty: int
    price: float
    mid: float
    spread_cost: float
    slippage: float
    fee: float


class ExecutionModel:
    """Market orders with latency, depth-aware slippage and per-contract fees."""

    def __init__(self, cfg: Config) -> None:
        e = cfg["execution"]
        self.latency_s = math.ceil(e["latency_ms"] / 1000)
        self.fee_per_side = e["fee_per_contract_rt"] / 2
        self.use_depth = e["slippage_model"] == "l2_or_sqrt"
        if e["slippage_model"] not in ("l2_or_sqrt", "sqrt"):
            raise ValueError(f"unknown slippage model: {e['slippage_model']}")
        self.impact_coef = e["sqrt_impact_coef"]

    def fill_index(self, book: Book, decision_idx: int) -> int:
        """Bar whose book an order decided at ``decision_idx`` executes against."""
        return min(decision_idx + self.latency_s, len(book) - 1)

    def _walk(self, px: np.ndarray, sz: np.ndarray, qty: int, side: int) -> list[tuple[int, float]]:
        """Split ``qty`` into (lots, price) pieces walking the visible levels on one side."""
        levels = len(px) if self.use_depth else 1
        pieces, left = [], qty
        for k in range(levels):
            take = int(min(left, sz[k]))
            if take > 0:
                pieces.append((take, px[k]))
                left -= take
            if left == 0:
                return pieces
        last = pieces[-1][1] if pieces else px[0]
        extra = self.impact_coef * math.sqrt(left / max(sz[0], 1.0)) * CL.tick_size
        return pieces + [(left, last + side * extra)]

    def market(self, book: Book, i: int, qty: int) -> Fill:
        """Execute a market order for signed ``qty`` against bar ``i`` of ``book``."""
        side = int(np.sign(qty))
        px, sz = (book.ask_px[i], book.ask_sz[i]) if side > 0 else (book.bid_px[i], book.bid_sz[i])
        pieces = self._walk(px, sz, abs(qty), side)
        lots = abs(qty)
        avg = sum(n * p for n, p in pieces) / lots
        mid = book.mid[i]
        return Fill(
            ts=book.ts[i],
            qty=qty,
            price=avg,
            mid=mid,
            spread_cost=lots * abs(px[0] - mid) * CL.multiplier,
            slippage=sum(n * abs(p - px[0]) for n, p in pieces) * CL.multiplier,
            fee=lots * self.fee_per_side,
        )
