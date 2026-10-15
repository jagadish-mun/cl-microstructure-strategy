"""Synthetic CL market simulator.

One session is built in four layers:

1. **Volatility**: target annual vol x daily regime (two-state Markov) x U-shaped intraday
   profile x mean-reverting log-vol x an EIA spike. Gives per-second sigma.
2. **Latent pressure** ``alpha``: an OU process with a ~45 s half-life. It adds a weak drift
   to the efficient price and tilts the sign of new order flow. This is the planted signal.
3. **Order flow**: Hawkes trade arrivals, Lillo-Mike-Farmer metaorder signs (long-memory sign
   autocorrelation), Pareto child sizes, and square-root impact split into a permanent part
   and a transient part that decays (giving short-term mean reversion).
4. **Book**: the observed mid is the latent price discretised to the tick grid with a 1-3 tick
   spread. L1 queue sizes shrink on the side the latent price is close to (queue depletion),
   so book imbalance and microprice carry information about the next mid move.

All parameters are in the ``generator`` section of the config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numba import njit

from clstrat.config import Config
from clstrat.contract import CL, SessionCalendar

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class DayData:
    """One simulated session."""

    quotes: pd.DataFrame
    trades: pd.DataFrame
    events: pd.DataFrame
    close_price: float


# ---------------------------------------------------------------------------
# Volatility, regime and latent signal (numpy, per second)
# ---------------------------------------------------------------------------


def _stationary_vol_prob(stay_prob: list[float]) -> float:
    """Stationary probability of the volatile state."""
    p_calm, p_vol = stay_prob
    return (1 - p_calm) / ((1 - p_calm) + (1 - p_vol))


def simulate_regimes(rng: np.random.Generator, n_days: int, stay_prob: list[float]) -> np.ndarray:
    """Daily two-state Markov chain (0 = calm, 1 = volatile), started from its stationary law."""
    states = np.empty(n_days, dtype=np.int64)
    states[0] = int(rng.random() < _stationary_vol_prob(stay_prob))
    for i in range(1, n_days):
        prev = states[i - 1]
        states[i] = prev if rng.random() < stay_prob[prev] else 1 - prev
    return states


def regime_norm(vol_mult: list[float], stay_prob: list[float]) -> float:
    """Scale so that the stationary mean of ``vol_mult**2`` is one."""
    pi_vol = _stationary_vol_prob(stay_prob)
    return math.sqrt((1 - pi_vol) * vol_mult[0] ** 2 + pi_vol * vol_mult[1] ** 2)


def intraday_profile(n_sec: int, p: Config) -> np.ndarray:
    """U-shaped multiplier, normalised so its mean square over the session is one."""
    s = np.arange(n_sec, dtype=float)
    u = (
        1.0
        + p["open_amp"] * np.exp(-s / (60 * p["open_tau_min"]))
        + p["close_amp"] * np.exp(-(n_sec - s) / (60 * p["close_tau_min"]))
    )
    return u / np.sqrt(np.mean(u**2))


def event_bump(n_sec: int, at: int | None, amp: float, decay_s: float) -> np.ndarray:
    """``1 + amp * exp(-(s - at) / decay)`` from ``at`` onwards; ones when there is no event."""
    out = np.ones(n_sec)
    if at is not None:
        lag = np.arange(n_sec - at, dtype=float)
        out[at:] += amp * np.exp(-lag / decay_s)
    return out


def stochastic_vol(rng: np.random.Generator, n_sec: int, p: Config) -> np.ndarray:
    """Mean-reverting log-vol on a 1-minute grid, held constant within each minute.

    Normalised so ``E[mult**2] = 1``.
    """
    n_min = n_sec // 60 + 1
    phi, sig = p["phi_per_min"], p["sigma"]
    var = sig**2 / (1 - phi**2)
    h = np.empty(n_min)
    h[0] = rng.normal(0, math.sqrt(var))
    eps = rng.normal(0, sig, n_min)
    for k in range(1, n_min):
        h[k] = phi * h[k - 1] + eps[k]
    return np.repeat(np.exp(h - var), 60)[:n_sec]


def ou_process(rng: np.random.Generator, n: int, half_life: float) -> np.ndarray:
    """Unit-variance AR(1) with the given half-life (in steps)."""
    phi = 0.5 ** (1.0 / half_life)
    eps = rng.normal(0, math.sqrt(1 - phi**2), n)
    out = np.empty(n)
    out[0] = rng.normal()
    for k in range(1, n):
        out[k] = phi * out[k - 1] + eps[k]
    return out


def efficient_log_path(
    rng: np.random.Generator,
    sigma_sec: np.ndarray,
    alpha: np.ndarray,
    drift_strength: float,
    fine_dt: float,
    jump: tuple[int, float] | None,
) -> np.ndarray:
    """Latent log-price on the fine grid, relative to the session open.

    Per second the increment is ``sigma * (drift_strength * alpha + z)`` minus the Ito term,
    so the price is a martingale apart from the planted alpha drift. ``jump`` is
    ``(second, log_size)`` for the EIA release.
    """
    steps_per_sec = round(1 / fine_dt)
    sig = np.repeat(sigma_sec, steps_per_sec)
    a = np.repeat(alpha, steps_per_sec)
    z = rng.normal(0, math.sqrt(fine_dt), sig.size)
    inc = sig * (drift_strength * a * fine_dt + z) - 0.5 * sig**2 * fine_dt
    if jump is not None:
        inc[jump[0] * steps_per_sec] += jump[1]
    return np.concatenate([[0.0], np.cumsum(inc)])


# ---------------------------------------------------------------------------
# Event-level kernels (numba)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _hawkes_times(seed: int, mu: np.ndarray, jump: float, beta: float) -> np.ndarray:
    """Ogata thinning for a Hawkes process with kernel ``jump * exp(-beta * t)``.

    ``mu`` is the baseline intensity for each second of the session.
    """
    np.random.seed(seed)
    n_sec = mu.shape[0]
    mu_max = mu.max()
    out = np.empty(int(n_sec * mu_max * 4) + 1024)
    t, exc, n = 0.0, 0.0, 0
    while True:
        lam_bar = mu_max + exc
        w = np.random.exponential(1.0 / lam_bar)
        t += w
        exc *= math.exp(-beta * w)
        if t >= n_sec:
            break
        if np.random.random() * lam_bar <= mu[int(t)] + exc:
            if n == out.shape[0]:
                out = np.concatenate((out, np.empty(out.shape[0])))
            out[n] = t
            n += 1
            exc += jump
    return out[:n]


@njit(cache=True)
def _pareto_int(u: float, tail: float, cap: int) -> int:
    """Discrete Pareto draw ``floor(u ** (-1 / tail))`` capped at ``cap``."""
    return min(cap, int(u ** (-1.0 / tail)))


@njit(cache=True)
def _simulate_book(
    seed: int,
    eff_log: np.ndarray,
    fine_dt: float,
    trade_t: np.ndarray,
    alpha: np.ndarray,
    p_wide: np.ndarray,
    open_ticks: float,
    quote_dt: float,
    n_active: int,
    length_tail: float,
    max_length: int,
    noise_frac: float,
    flow_loading: float,
    size_tail: float,
    max_lots: int,
    perm_k: float,
    trans_k: float,
    trans_tau: float,
    p_three: float,
    refresh_prob: float,
    depth_log_mu: float,
    depth_log_sigma: float,
    depth_phi: float,
    min_frac: float,
    n_levels: int,
    level_growth: float,
    deep_noise: float,
    alpha_tilt: float,
):
    """Walk scheduled book refreshes and trades in time order.

    Prices are integer ticks. Returns trade arrays ``(t, px, size, side)``, quote arrays
    ``(t, bid, ask, bid_sz[:, level], ask_sz[:, level])`` and the closing latent price in
    ticks.
    """
    np.random.seed(seed)
    n_sec = alpha.shape[0]
    n_grid = int(n_sec / quote_dt)
    n_tr = trade_t.shape[0]
    nq = n_grid + n_tr
    q_t = np.empty(nq)
    q_bid = np.empty(nq, np.int64)
    q_ask = np.empty(nq, np.int64)
    q_bsz = np.empty((nq, n_levels), np.int64)
    q_asz = np.empty((nq, n_levels), np.int64)
    tr_px = np.empty(n_tr, np.int64)
    tr_sz = np.empty(n_tr, np.int64)
    tr_side = np.empty(n_tr, np.int64)

    meta_sign = np.empty(n_active, np.int64)
    meta_left = np.empty(n_active, np.int64)
    p0 = 1.0 / (1.0 + math.exp(-2.0 * flow_loading * alpha[0]))
    for j in range(n_active):
        meta_sign[j] = 1 if np.random.random() < p0 else -1
        meta_left[j] = _pareto_int(np.random.random(), length_tail, max_length)

    perm, trans, last = 0.0, 0.0, 0.0
    spread = 1
    log_db, log_da = depth_log_mu, depth_log_mu
    shock = depth_log_sigma * math.sqrt(1 - depth_phi**2)
    deep_base = math.exp(depth_log_mu)
    bid, ask = 0, 0
    gi, ti, qi = 0, 0, 0
    c = open_ticks
    while gi < n_grid or ti < n_tr:
        is_trade = ti < n_tr and (gi >= n_grid or trade_t[ti] < gi * quote_dt)
        t = trade_t[ti] if is_trade else gi * quote_dt
        trans *= math.exp(-(t - last) / trans_tau)
        last = t
        a = alpha[min(int(t), n_sec - 1)]
        if is_trade:
            p_buy = 1.0 / (1.0 + math.exp(-2.0 * flow_loading * a))
            if np.random.random() < noise_frac:
                sign = 1 if np.random.random() < p_buy else -1
            else:
                j = np.random.randint(0, n_active)
                sign = meta_sign[j]
                meta_left[j] -= 1
                if meta_left[j] <= 0:
                    meta_sign[j] = 1 if np.random.random() < p_buy else -1
                    meta_left[j] = _pareto_int(np.random.random(), length_tail, max_length)
            size = _pareto_int(np.random.random(), size_tail, max_lots)
            tr_px[ti] = ask if sign > 0 else bid
            tr_sz[ti] = size
            tr_side[ti] = sign
            root = math.sqrt(size)
            perm += perm_k * sign * root
            trans += trans_k * sign * root
            tq = t
            ti += 1
        else:
            if np.random.random() < refresh_prob:
                if np.random.random() < p_wide[min(int(t), n_sec - 1)]:
                    spread = 3 if np.random.random() < p_three else 2
                else:
                    spread = 1
            log_db = depth_phi * log_db + (1 - depth_phi) * depth_log_mu
            log_db += shock * np.random.normal()
            log_da = depth_phi * log_da + (1 - depth_phi) * depth_log_mu
            log_da += shock * np.random.normal()
            tq = t
            gi += 1

        k = min(int(t / fine_dt), eff_log.shape[0] - 1)
        c = open_ticks * math.exp(eff_log[k]) + perm + trans
        bid = int(math.floor(c - (spread - 1) / 2.0))
        ask = bid + spread
        frac = (c - bid) / spread
        q_t[qi] = tq
        q_bid[qi] = bid
        q_ask[qi] = ask
        q_bsz[qi, 0] = max(1, round(math.exp(log_db) * (min_frac + (1 - min_frac) * frac)))
        q_asz[qi, 0] = max(1, round(math.exp(log_da) * (min_frac + (1 - min_frac) * (1 - frac))))
        for lvl in range(1, n_levels):
            base = deep_base * level_growth**lvl
            tilt = alpha_tilt * a
            q_bsz[qi, lvl] = max(1, round(base * math.exp(deep_noise * np.random.normal() + tilt)))
            q_asz[qi, lvl] = max(1, round(base * math.exp(deep_noise * np.random.normal() - tilt)))
        qi += 1
    return trade_t, tr_px, tr_sz, tr_side, q_t, q_bid, q_ask, q_bsz, q_asz, c


# ---------------------------------------------------------------------------
# Session assembly
# ---------------------------------------------------------------------------


def _relative_vol(
    g: Config, n_sec: int, regime: int, sv: np.ndarray, eia_sec: int | None
) -> np.ndarray:
    """Per-second vol relative to the unconditional per-second target."""
    reg = g["regime"]
    return (
        reg["vol_mult"][regime]
        / regime_norm(reg["vol_mult"], reg["stay_prob"])
        * intraday_profile(n_sec, g["intraday"])
        * sv
        * event_bump(n_sec, eia_sec, g["eia"]["vol_amp"], g["eia"]["vol_decay_s"])
    )


def _wide_spread_prob(g: Config, rel_vol: np.ndarray, eia_sec: int | None) -> np.ndarray:
    """Probability that a spread refresh draws a 2-3 tick spread, per second."""
    sp = g["spread"]
    p = np.minimum(sp["p_wide_max"], sp["p_wide_base"] * rel_vol ** sp["vol_exponent"])
    if eia_sec is not None:
        lo, hi = sp["eia_window_s"]
        window = slice(max(0, eia_sec + lo), eia_sec + hi)
        p[window] = np.maximum(p[window], sp["eia_p_wide"])
    return p


def _trade_intensity(g: Config, n_sec: int, regime: int, eia_sec: int | None) -> np.ndarray:
    """Hawkes baseline intensity per second: U-shape x regime x EIA, scaled to the mean rate."""
    h, reg = g["hawkes"], g["regime"]
    shape = intraday_profile(n_sec, g["intraday"])
    shape = shape / shape.mean()
    reg_mult = reg["vol_mult"][regime] / regime_norm(reg["vol_mult"], reg["stay_prob"])
    eia = event_bump(n_sec, eia_sec, g["eia"]["rate_amp"], g["eia"]["rate_decay_s"])
    return h["mean_rate"] * (1 - h["branching"]) * shape * reg_mult * eia


def simulate_day(
    cfg: Config, cal: SessionCalendar, day: date, regime: int, open_price: float, seed: int
) -> DayData:
    """Simulate one regular-hours session starting at ``open_price``."""
    g = cfg["generator"]
    rng = np.random.default_rng(seed)
    n_sec = cal.session_seconds()
    open_ts, _ = cal.session_bounds(day)
    eia_sec = None
    if cfg["data"]["eia_events"] and cal.is_eia_day(day):
        eia_sec = int((cal.at(day, cal.eia_time) - open_ts).total_seconds())

    sv = stochastic_vol(rng, n_sec, g["stoch_vol"])
    rel_vol = _relative_vol(g, n_sec, regime, sv, eia_sec)
    base_sigma = cfg["data"]["annual_vol"] / math.sqrt(TRADING_DAYS_PER_YEAR * n_sec)
    alpha = ou_process(rng, n_sec, g["alpha"]["half_life_s"])
    jump = None
    if eia_sec is not None:
        size = rng.uniform(*g["eia"]["jump_pct"]) / 100 * rng.choice([-1.0, 1.0])
        jump = (eia_sec, math.log1p(size))
    eff = efficient_log_path(
        rng, g["diffusion_scale"] * base_sigma * rel_vol, alpha,
        g["alpha"]["drift_strength"], g["fine_dt"], jump,
    )  # fmt: skip

    beta = 1.0 / g["hawkes"]["decay_s"]
    trade_t = _hawkes_times(
        int(rng.integers(2**31)),
        _trade_intensity(g, n_sec, regime, eia_sec),
        g["hawkes"]["branching"] * beta,
        beta,
    )
    m, sz, imp, sp, dp = g["metaorders"], g["sizes"], g["impact"], g["spread"], g["depth"]
    out = _simulate_book(
        int(rng.integers(2**31)), eff, g["fine_dt"], trade_t, alpha,
        _wide_spread_prob(g, rel_vol, eia_sec), open_price / CL.tick_size, g["quote_interval"],
        m["n_active"], m["length_tail"], m["max_length"], m["noise_frac"],
        g["alpha"]["flow_loading"], sz["tail"], sz["max_lots"],
        imp["permanent"], imp["transient"], imp["transient_decay_s"],
        sp["p_three_given_wide"], sp["refresh_prob"],
        math.log(dp["mean_lots"]), dp["log_sigma"], dp["ar_phi"], dp["min_frac"],
        dp["levels"], dp["level_growth"], dp["deep_noise"], dp["alpha_tilt"],
    )  # fmt: skip
    return _to_frames(out, open_ts, eia_sec)


def _timestamps(open_ts: pd.Timestamp, seconds: np.ndarray) -> pd.DatetimeIndex:
    """Session-relative seconds to tz-aware nanosecond timestamps."""
    return open_ts + pd.to_timedelta(np.round(seconds * 1e9).astype(np.int64), unit="ns")


def _to_frames(out: tuple, open_ts: pd.Timestamp, eia_sec: int | None) -> DayData:
    """Convert kernel arrays (integer ticks) to the parquet schema (prices in dollars)."""
    tr_t, tr_px, tr_sz, tr_side, q_t, q_bid, q_ask, q_bsz, q_asz, close_ticks = out
    tick = CL.tick_size
    trades = pd.DataFrame(
        {
            "ts": _timestamps(open_ts, tr_t),
            "price": np.round(tr_px * tick, 2),
            "size": tr_sz.astype(np.int32),
            "side": tr_side.astype(np.int8),
        }
    )
    quotes = {"ts": _timestamps(open_ts, q_t)}
    for lvl in range(q_bsz.shape[1]):
        sfx = "" if lvl == 0 else f"_{lvl + 1}"
        quotes[f"bid_px{sfx}"] = np.round((q_bid - lvl) * tick, 2)
        quotes[f"bid_sz{sfx}"] = q_bsz[:, lvl].astype(np.int32)
        quotes[f"ask_px{sfx}"] = np.round((q_ask + lvl) * tick, 2)
        quotes[f"ask_sz{sfx}"] = q_asz[:, lvl].astype(np.int32)
    eia = [] if eia_sec is None else [eia_sec]
    events = pd.DataFrame(
        {
            "ts": _timestamps(open_ts, np.array(eia, dtype=float)),
            "type": pd.Series(["EIA"] * len(eia), dtype="string"),
        }
    )
    return DayData(pd.DataFrame(quotes), trades, events, float(close_ticks * tick))


def generate_dataset(cfg: Config, out_dir: Path) -> list[date]:
    """Simulate ``data.n_days`` sessions and write one parquet file per table per day.

    The data/generator config, seed and daily regimes are saved as ``generator_config.yaml``.
    """
    cal = SessionCalendar.from_config(cfg)
    days = cal.trading_days(cfg["data"]["start_date"], cfg["data"]["n_days"])
    regime_ss, *day_ss = np.random.SeedSequence(cfg["seed"]).spawn(len(days) + 1)
    regimes = simulate_regimes(
        np.random.default_rng(regime_ss), len(days), cfg["generator"]["regime"]["stay_prob"]
    )
    for table in ("quotes", "trades", "events"):
        (out_dir / table).mkdir(parents=True, exist_ok=True)
    price = cfg["data"]["start_price"]
    for day, regime, ss in zip(days, regimes, day_ss, strict=True):
        d = simulate_day(cfg, cal, day, int(regime), price, int(ss.generate_state(1)[0]))
        for table in ("quotes", "trades", "events"):
            path = out_dir / table / f"{day.isoformat()}.parquet"
            getattr(d, table).to_parquet(path, index=False)
        price = d.close_price
    meta = {
        "seed": cfg["seed"],
        "data": cfg["data"],
        "generator": cfg["generator"],
        "regimes": {d.isoformat(): int(r) for d, r in zip(days, regimes, strict=True)},
    }
    with open(out_dir / "generator_config.yaml", "w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    return days
