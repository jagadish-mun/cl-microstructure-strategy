"""Prediction -> target position.

Entry needs a strong signal (``ŷ > θ_entry`` long, ``ŷ < -θ_entry`` short, or the percentile
equivalent) and, if ``require_edge``, a positive expected edge ``|ŷ| - cost`` where cost is the
live spread plus round-trip fees in ticks. A position is held until the signal weakens past
the (lower) exit threshold (hysteresis), reverses, or a stop-loss, take-profit or
maximum-holding-time exit fires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from clstrat.config import Config
from clstrat.contract import CL


@dataclass
class Position:
    """Open position: signed contracts, average entry price and entry time."""

    qty: int = 0
    entry_px: float = math.nan
    entry_ts: pd.Timestamp | None = None

    @property
    def side(self) -> int:
        return int(np.sign(self.qty))


@dataclass(frozen=True)
class Decision:
    """Inputs the signal rule sees at one decision time."""

    ts: pd.Timestamp
    pred: float  # ŷ in ticks
    pct: float  # percentile of ŷ in the training predictions (NaN in ticks mode)
    mid: float
    spread_ticks: float
    rv_ticks: float  # realized vol of the mid over the target horizon, in ticks


def percentile_of(pred: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Map predictions to [0, 1] using a training-prediction quantile grid."""
    return np.interp(pred, quantiles, np.linspace(0, 1, len(quantiles)))


class SignalRule:
    """Stateless mapping from a decision and the current position to a target position."""

    def __init__(self, cfg: Config) -> None:
        self.s = cfg["strategy"]
        self.fee_ticks = cfg["execution"]["fee_per_contract_rt"] / CL.tick_value

    def cost_ticks(self, d: Decision) -> float:
        """Estimated round-trip cost per contract: cross the spread once in total
        (half on entry, half on exit) plus the round-trip fee."""
        return d.spread_ticks + self.fee_ticks

    def _strong(self, d: Decision, side: int) -> bool:
        s = self.s
        if s["threshold_mode"] == "percentile":
            beyond = d.pct >= s["pct_entry"] if side > 0 else d.pct <= 1 - s["pct_entry"]
        else:
            beyond = side * d.pred > s["theta_entry_ticks"]
        edge_ok = not s["require_edge"] or side * d.pred - self.cost_ticks(d) > 0
        return bool(beyond and edge_ok)

    def _weak(self, d: Decision, side: int) -> bool:
        s = self.s
        if s["threshold_mode"] == "percentile":
            return d.pct < s["pct_exit"] if side > 0 else d.pct > 1 - s["pct_exit"]
        return side * d.pred < s["theta_exit_ticks"]

    def size(self, d: Decision) -> int:
        """Contracts for a new position."""
        s = self.s
        if s["sizing"] == "fixed":
            return int(s["base_size"])
        if s["sizing"] == "linear":
            raw = s["base_size"] * abs(d.pred) / s["theta_entry_ticks"]
        elif s["sizing"] == "vol_target":
            raw = s["vol_target_ticks"] / d.rv_ticks if d.rv_ticks > 0 else s["base_size"]
        else:
            raise ValueError(f"unknown sizing: {s['sizing']}")
        return int(np.clip(round(raw), 1, s["max_size"]))

    def _exit_reason(self, d: Decision, pos: Position) -> str | None:
        s = self.s
        pnl_ticks = pos.side * (d.mid - pos.entry_px) / CL.tick_size
        if pnl_ticks <= -s["stop_loss_ticks"]:
            return "stop_loss"
        if pnl_ticks >= s["take_profit_ticks"]:
            return "take_profit"
        if (d.ts - pos.entry_ts).total_seconds() >= s["max_hold_seconds"]:
            return "max_hold"
        return None

    def target(self, d: Decision, pos: Position) -> tuple[int, str]:
        """Target signed position and the reason for any change (empty if none)."""
        if pos.side == 0:
            for side in (1, -1):
                if self._strong(d, side):
                    return side * self.size(d), "entry"
            return 0, ""
        reason = self._exit_reason(d, pos)
        if reason:
            return 0, reason
        if self._strong(d, -pos.side):
            return -pos.side * self.size(d), "reversal"
        if self._weak(d, pos.side):
            return 0, "signal_exit"
        return pos.qty, ""
