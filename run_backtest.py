"""
Kronos + TA hybrid strategy -- backtest runner.

Usage:
    python run_backtest.py                      # surrogate forecaster (works anywhere)
    python run_backtest.py --real-kronos         # real Kronos-small from HF Hub
                                                  # (needs torch + internet + ideally GPU)

Data: defaults to the sample 5-min HK Alibaba (09988) K-line data shipped in
the Kronos repo, resampled to daily bars. Point --csv at your own 5-min (or
finer) OHLCV CSV to use another instrument -- any columns named
timestamps/open/high/low/close/volume(/amount) will work.
"""
from __future__ import annotations

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from strategy.data_utils import load_daily_bars
from strategy.indicators import add_indicators
from strategy.forecaster import get_forecaster
from strategy.engine import StrategyConfig
from strategy.backtest import BacktestConfig, run_backtest, performance_summary

DEFAULT_CSV = "kronos_src/finetune_csv/data/HK_ali_09988_kline_5min_all.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="CSV file path (optional, uses yfinance if not provided)")
    parser.add_argument("--ticker", default="0700.HK", help="Yahoo Finance ticker (e.g. AAPL, 0700.HK)")
    parser.add_argument("--start", default=None, help="start date YYYY-MM-DD (yfinance only)")
    parser.add_argument("--end", default=None, help="end date YYYY-MM-DD (yfinance only)")
    parser.add_argument("--real-kronos", action="store_true")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    if args.csv:
        print(f"Loading data from {args.csv} ...")
        daily = load_daily_bars(args.csv)
    else:
        print(f"Loading {args.ticker} from Yahoo Finance ({args.start or 'default'} to {args.end or 'default'}) ...")
        from strategy.data_utils import load_daily_bars_yfinance
        daily = load_daily_bars_yfinance(args.ticker, start=args.start, end=args.end)
        if args.start:
            daily = daily.loc[args.start:]
    print(f"Loaded {len(daily)} daily bars: {daily.index.min().date()} -> {daily.index.max().date()}")

    print("Computing TA indicators ...")
    daily = add_indicators(daily).dropna()

    print(f"Setting up forecaster (real_kronos={args.real_kronos}) ...")
    forecaster = get_forecaster(use_real_kronos=args.real_kronos)

    strat_cfg = StrategyConfig()
    bt_cfg = BacktestConfig()

    print("Running walk-forward backtest ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_backtest(daily, forecaster, strat_cfg, bt_cfg)

    stats = performance_summary(result.equity_curve, result.trades)
    bench_stats = performance_summary(result.benchmark_curve, [])

    print("\n" + "=" * 60)
    print("STRATEGY vs BUY & HOLD")
    print("=" * 60)
    header = f"{'metric':<16}{'strategy':>15}{'buy & hold':>15}"
    print(header)
    for key in ["total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar"]:
        s_val = stats[key]
        b_val = bench_stats[key]
        fmt = "{:>15.2%}" if abs(s_val) < 10 else "{:>15.2f}"
        print(f"{key:<16}{fmt.format(s_val)}{fmt.format(b_val)}")
    print(f"{'n_trades':<16}{stats['n_trades']:>15}")
    print(f"{'win_rate':<16}{stats['win_rate']:>15.2%}" if pd.notna(stats["win_rate"]) else f"{'win_rate':<16}{'n/a':>15}")
    print(f"{'profit_factor':<16}{stats['profit_factor']:>15.2f}" if pd.notna(stats["profit_factor"]) else f"{'profit_factor':<16}{'n/a':>15}")
    print("=" * 60)

    # --- plots ---
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(result.equity_curve, label="Kronos + TA strategy", linewidth=1.5)
    axes[0].plot(result.benchmark_curve, label="Buy & hold", linewidth=1.2, alpha=0.7)
    axes[0].set_ylabel("Equity ($)")
    axes[0].legend()
    axes[0].set_title("Kronos-forecast + TA-filtered strategy vs Buy & Hold")

    dd = result.equity_curve / result.equity_curve.cummax() - 1
    axes[1].fill_between(dd.index, dd.values * 100, 0, color="crimson", alpha=0.4)
    axes[1].set_ylabel("Drawdown (%)")
    fig.tight_layout()

    out_path = f"{args.out_dir}/equity_curve.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nSaved equity curve plot to {out_path}")

    trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
    trades_path = f"{args.out_dir}/trade_log.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"Saved trade log ({len(trades_df)} trades) to {trades_path}")

    signal_path = f"{args.out_dir}/signal_log.csv"
    result.signal_log.to_csv(signal_path)
    print(f"Saved daily signal log to {signal_path}")


if __name__ == "__main__":
    main()
