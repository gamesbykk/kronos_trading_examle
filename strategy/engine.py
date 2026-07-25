"""
Signal engine: combines the Kronos forecast (directional edge + confidence)
with standard TA indicators (regime / momentum filters) into a single daily
target position. TA acts as a gate on the model's forecast rather than a
separate vote -- the philosophy is "only act on Kronos when the tape agrees".
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .forecaster import ForecastResult


@dataclass
class StrategyConfig:
    lookback: int = 400          # bars fed to Kronos as context
    pred_len: int = 10            # forecast horizon, trading days
    rebalance_every: int = 5      # re-run Kronos every N trading days
    sample_count: int = 30

    score_entry_long: float = 0.5     # forecast score threshold to go long
    score_entry_short: float = -0.5   # forecast score threshold to go short
    score_exit: float = 0.1           # |score| below this closes the position
    prob_up_min_long: float = 0.55
    prob_up_max_short: float = 0.45

    rsi_overbought: float = 75.0
    rsi_oversold: float = 25.0
    adx_trend_min: float = 15.0   # below this, ADX says "no real trend" -> stand aside

    allow_short: bool = True

    atr_stop_mult: float = 3.0
    max_hold_days: int = 2000


def _regime_ok_for_long(row: pd.Series, cfg: StrategyConfig) -> bool:
    return bool(
        row["trend_bull"]
        and row["rsi"] < cfg.rsi_overbought
        and row["adx"] >= cfg.adx_trend_min
    )


def _regime_ok_for_short(row: pd.Series, cfg: StrategyConfig) -> bool:
    return bool(
        row["trend_bear"]
        and row["rsi"] > cfg.rsi_oversold
        and row["adx"] >= cfg.adx_trend_min
    )


def decide_target_position(
    row: pd.Series,
    current_position: int,
    forecast: ForecastResult,
    cfg: StrategyConfig,
) -> tuple[int, str]:
    """Returns (target_position, reason). Position is -1 / 0 / +1 (direction;
    sizing is handled separately in the backtester via ATR risk-sizing)."""
    score = forecast.score

    if current_position == 0:
        if (
            score >= cfg.score_entry_long
            and forecast.prob_up >= cfg.prob_up_min_long
            and _regime_ok_for_long(row, cfg)
        ):
            return 1, f"enter long (score={score:.2f}, prob_up={forecast.prob_up:.2f}, TA confirms uptrend)"
        if (
            cfg.allow_short
            and score <= cfg.score_entry_short
            and forecast.prob_up <= cfg.prob_up_max_short
            and _regime_ok_for_short(row, cfg)
        ):
            return -1, f"enter short (score={score:.2f}, prob_up={forecast.prob_up:.2f}, TA confirms downtrend)"
        return 0, "no edge / regime not confirmed"

    if current_position == 1:
        if score <= cfg.score_exit or not row["trend_bull"]:
            return 0, f"exit long (score decayed to {score:.2f} or trend broke)"
        return 1, "hold long"

    if current_position == -1:
        if score >= -cfg.score_exit or not row["trend_bear"]:
            return 0, f"exit short (score decayed to {score:.2f} or trend broke)"
        return -1, "hold short"

    return 0, "flat"
