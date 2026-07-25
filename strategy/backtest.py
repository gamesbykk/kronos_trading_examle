"""
Walk-forward backtest engine.

Design choices, stated explicitly because they matter for trusting the
results:

  * Signals are computed from information available through day t's CLOSE.
    Orders are filled at day t+1's OPEN. No same-bar lookahead.
  * The Kronos forecast is only re-run every `rebalance_every` trading days
    (it's expensive) and cached in between; TA-based stop-loss / trend-break
    exits are still checked every single day.
  * Position sizing is risk-based: each new trade risks
    `risk_per_trade` of current equity against its ATR-based stop, not a
    fixed share count. Gross exposure is capped at `max_gross_exposure`.
  * Commission + slippage are charged in bps of notional on every fill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine import StrategyConfig, decide_target_position
from .forecaster import ForecastResult


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.2       # fraction of equity risked per trade (to stop)
    max_gross_exposure: float = 1.0    # no leverage by default
    cost_bps: float = 5.0              # commission + slippage, bps of notional per fill


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trades: list = field(default_factory=list)
    signal_log: pd.DataFrame = None


def run_backtest(
    df: pd.DataFrame,
    forecaster,
    strat_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
) -> BacktestResult:
    n = len(df)
    start_idx = strat_cfg.lookback + 1
    if start_idx >= n - 1:
        raise ValueError("Not enough bars for the requested lookback + horizon")

    equity = bt_cfg.initial_capital
    cash = bt_cfg.initial_capital
    position = 0          # -1, 0, 1
    shares = 0.0
    entry_price = None
    entry_atr = None
    entry_date = None
    days_held = 0

    equity_dates, equity_vals = [], []
    trades: list[Trade] = []
    log_rows = []
    current_forecast: ForecastResult | None = None

    for t in range(start_idx, n - 1):
        row = df.iloc[t]
        date_t = df.index[t]
        next_row = df.iloc[t + 1]
        next_date = df.index[t + 1]

        # mark-to-market equity using today's close before any action
        if position == 1:
            equity = cash + shares * row["close"]  # long: cash + position value
        elif position == -1:
            equity = cash - shares * row["close"]  # short: cash - liability value
        else:
            equity = cash

        # --- intraday-style stop check (uses info from day t only) ---
        stopped_out = False
        if position == 1 and entry_atr is not None:
            stop_price = entry_price - strat_cfg.atr_stop_mult * entry_atr
            if row["low"] <= stop_price:
                stopped_out = True
        elif position == -1 and entry_atr is not None:
            stop_price = entry_price + strat_cfg.atr_stop_mult * entry_atr
            if row["high"] >= stop_price:
                stopped_out = True

        max_hold_hit = position != 0 and days_held >= strat_cfg.max_hold_days

        # --- refresh Kronos forecast on schedule ---
        is_rebalance_day = (t - start_idx) % strat_cfg.rebalance_every == 0
        if is_rebalance_day or current_forecast is None:
            window = df.iloc[t - strat_cfg.lookback + 1 : t + 1]
            current_forecast = forecaster.forecast(
                window, pred_len=strat_cfg.pred_len, sample_count=strat_cfg.sample_count
            )

        target_pos, reason = decide_target_position(row, position, current_forecast, strat_cfg)
        if stopped_out:
            target_pos, reason = 0, "stop-loss hit"
        elif max_hold_hit and target_pos == position:
            target_pos, reason = 0, "max holding period reached"

        log_rows.append(
            {
                "date": date_t, "position": position, "target": target_pos,
                "score": current_forecast.score, "prob_up": current_forecast.prob_up,
                "reason": reason,
            }
        )

        # --- execute any position change at next day's OPEN ---
        if target_pos != position:
            fill_price = next_row["open"]
            cost_mult = bt_cfg.cost_bps / 10_000.0

            if position != 0:
                # close existing leg
                gross = shares * position * (fill_price - entry_price)  # PnL from position
                notional = shares * fill_price
                
                if position == 1:
                    # closing a long: we sell and receive fill_price per share
                    cash += notional - notional * cost_mult
                elif position == -1:
                    # closing a short: we buy back at fill_price per share (negative cash flow)
                    cash -= notional + notional * cost_mult
                
                trades.append(
                    Trade(entry_date, next_date, position, entry_price, fill_price,
                          shares, gross - notional * cost_mult, reason)
                )
                position, shares, entry_price, entry_atr, entry_date, days_held = 0, 0.0, None, None, None, 0

            if target_pos != 0:
                stop_dist = strat_cfg.atr_stop_mult * row["atr"]
                risk_dollars = equity * bt_cfg.risk_per_trade
                sized_shares = risk_dollars / max(stop_dist, 1e-6)
                max_shares_by_exposure = (equity * bt_cfg.max_gross_exposure) / fill_price
                new_shares = min(sized_shares, max_shares_by_exposure)
                notional = new_shares * fill_price
                
                if target_pos == 1:
                    # entering a long: pay cash for shares
                    cash -= notional + notional * cost_mult
                elif target_pos == -1:
                    # entering a short: receive cash for borrowed shares (minus cost)
                    cash += notional - notional * cost_mult
                
                position, shares = target_pos, new_shares
                entry_price, entry_atr, entry_date, days_held = fill_price, row["atr"], next_date, 0
        else:
            if position != 0:
                days_held += 1

        equity_dates.append(date_t)
        equity_vals.append(equity)

    equity_curve = pd.Series(equity_vals, index=equity_dates, name="equity")
    bench = df["close"].loc[equity_curve.index]
    benchmark_curve = bt_cfg.initial_capital * bench / bench.iloc[0]

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        trades=trades,
        signal_log=pd.DataFrame(log_rows).set_index("date"),
    )


def performance_summary(equity_curve: pd.Series, trades: list, periods_per_year: int = 252) -> dict:
    rets = equity_curve.pct_change().dropna()
    n_days = len(rets)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (periods_per_year / max(n_days, 1)) - 1
    ann_vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = (rets.mean() * periods_per_year) / ann_vol if ann_vol > 0 else np.nan
    downside = rets[rets < 0]
    sortino = (rets.mean() * periods_per_year) / (downside.std() * np.sqrt(periods_per_year)) if len(downside) > 0 else np.nan
    cum = equity_curve / equity_curve.cummax()
    max_dd = cum.min() - 1
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    wins = [tr.pnl for tr in trades if tr.pnl > 0]
    losses = [tr.pnl for tr in trades if tr.pnl <= 0]
    win_rate = len(wins) / len(trades) if trades else np.nan
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else np.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "n_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }
