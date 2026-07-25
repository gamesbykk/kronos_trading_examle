"""
Standard technical-analysis indicators, computed causally (every value at row t
uses only data available up to and including t, so nothing here can leak
future information into the backtest).
"""
from __future__ import annotations

import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands


def add_indicators(
    df: pd.DataFrame,
    ema_fast: int = 50,
    ema_slow: int = 200,
    rsi_len: int = 14,
    atr_len: int = 14,
    bb_len: int = 20,
    bb_std: float = 2.0,
    adx_len: int = 14,
) -> pd.DataFrame:
    """Return a copy of df with TA columns appended."""
    out = df.copy()

    out["ema_fast"] = EMAIndicator(out["close"], window=ema_fast).ema_indicator()
    out["ema_slow"] = EMAIndicator(out["close"], window=ema_slow).ema_indicator()
    out["trend_bull"] = (out["ema_fast"] > out["ema_slow"]) & (out["close"] > out["ema_fast"])
    out["trend_bear"] = (out["ema_fast"] < out["ema_slow"]) & (out["close"] < out["ema_fast"])

    out["rsi"] = RSIIndicator(out["close"], window=rsi_len).rsi()

    macd = MACD(out["close"])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    out["atr"] = AverageTrueRange(out["high"], out["low"], out["close"], window=atr_len).average_true_range()

    bb = BollingerBands(out["close"], window=bb_len, window_dev=bb_std)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_pctb"] = bb.bollinger_pband()

    out["adx"] = ADXIndicator(out["high"], out["low"], out["close"], window=adx_len).adx()

    return out
