"""
Data loading utilities.

Loads raw K-line CSV data (5-min bars, in the format Kronos itself expects:
timestamps, open, high, low, close, volume, amount) and resamples to daily
bars for swing-style backtesting.
"""
from __future__ import annotations

import pandas as pd


REQUIRED_COLS = ["open", "high", "low", "close", "volume"]
import yfinance as yf
import pandas as pd
def load_daily_bars_yfinance(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """Load daily OHLCV data from Yahoo Finance.
    
    Args:
        ticker: stock ticker (e.g. "AAPL", "0700.HK" for HK stocks)
        start: start date as string "YYYY-MM-DD" (default: 5 years ago)
        end: end date as string "YYYY-MM-DD" (default: today)
    
    Returns:
        DataFrame with columns [open, high, low, close, volume, amount]
    """
    if start is None:
        start = pd.Timestamp.now() - pd.Timedelta(days=5*365)
    if end is None:
        end = pd.Timestamp.now()
    
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.sort_index()
    df.index.name = "date"
    df.columns=REQUIRED_COLS
    
    # Rename yfinance columns to match our format
    df = df.rename(columns={
        "Open": "open",
        "High": "high", 
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    
    # Keep only OHLCV
    df = df[["open", "high", "low", "close", "volume"]]
    
    # Add "amount" (notional volume) like the CSV loader does
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    
    # Drop any NaN rows
    df = df.dropna()
    df = df[df["volume"] > 0]
    print(df)
    return df[["open", "high", "low", "close", "volume", "amount"]]

def load_intraday_csv(path: str, timestamp_col: str = "timestamps") -> pd.DataFrame:
    """Load a raw 5-min (or similar) K-line CSV into a clean, sorted DataFrame."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col])
    df = df.set_index(timestamp_col)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df[REQUIRED_COLS + ["amount"]]


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample intraday bars to daily OHLCV bars (causal, no look-ahead)."""
    daily = df.resample("1D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }
    )
    daily = daily.dropna(subset=["open", "high", "low", "close"])
    daily = daily[daily["volume"] > 0]  # drop empty/weekend rows
    return daily


def load_daily_bars(path: str) -> pd.DataFrame:
    """Convenience: load intraday CSV and return clean daily bars."""
    intraday = load_intraday_csv(path)
    return resample_to_daily(intraday)
