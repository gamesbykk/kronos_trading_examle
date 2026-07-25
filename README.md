# Kronos + TA Hybrid Strategy

A swing-trading strategy that uses the [Kronos](https://github.com/shiyu-coder/Kronos)
financial foundation model as its directional forecast source, gated by
standard technical-analysis filters, backtested with a walk-forward engine
that charges realistic costs and sizes positions off ATR risk.

## How it works

**1. Forecast (the "what")** — Every `rebalance_every` trading days, the last
`lookback` daily bars are fed to Kronos, which samples `sample_count`
possible future price paths over the next `pred_len` days. From those paths
we derive:
- `expected_return` — mean terminal return across sampled paths
- `prob_up` — fraction of paths that finish positive
- `dispersion` — spread of terminal returns (the model's own uncertainty)
- `score = expected_return / dispersion` — a risk-adjusted forecast strength;
  this is the number the strategy actually acts on, so a confident small
  edge can outrank a large but highly uncertain one.

**2. Filter (the "should we act on it")** — standard TA indicators gate entries:
- **Trend regime**: EMA50 vs EMA200 + price position (only take longs in a
  confirmed uptrend, shorts in a confirmed downtrend)
- **ADX**: skip trading in a directionless/choppy tape (ADX < 15)
- **RSI**: don't chase a long that's already overbought (>75) or a short
  that's already oversold (<25)

Kronos without a regime filter will happily "forecast" continuation of a
move that's about to exhaust itself; TA without a forward-looking model is
purely reactive. Gating one with the other is the point of combining them.

**3. Risk (the "how much")** — each trade risks a fixed `risk_per_trade`
fraction of equity (default 1%) against an ATR-based stop
(`entry ± atr_stop_mult × ATR`), so position size scales inversely with
volatility. A `max_hold_days` cap forces an exit if neither the stop nor the
signal has closed the trade.

**4. Backtest** — walk-forward, day by day:
- signals use only information through day *t*'s close; fills happen at day
  *t+1*'s open (no same-bar lookahead)
- stop-loss / trend-break exits are checked every day; the Kronos forecast
  itself is only refreshed every `rebalance_every` days (it's an expensive
  model to run — no strategy re-runs a foundation model every single bar in
  practice)
- commission + slippage charged in bps on every fill

## ⚠️ Important: the forecaster you're running right now is a surrogate

This sandbox can reach PyPI and GitHub but **not** huggingface.co, so the
real Kronos weights (`NeoQuasar/Kronos-small` etc.) can't be downloaded
here. `strategy/forecaster.py` ships two interchangeable implementations:

- `SurrogateForecaster` — block-bootstraps historical daily returns plus a
  damped momentum term. This is what `run_backtest.py` uses by default. It's
  useful for confirming the strategy/backtest *mechanics* are correct
  (sizing, stops, no-lookahead, metrics), but it has no real predictive
  skill — don't read anything into its returns as evidence Kronos "works."
- `KronosForecaster` — the real thing, calling the actual model via
  `kronos_src/`.

**To run with the real model** on a machine with internet + ideally a GPU:

```bash
pip install -r kronos_src/requirements.txt
python run_backtest.py --real-kronos
```

One thing to fix before relying on it: `KronosPredictor.predict()` as
shipped by the Kronos repo averages `sample_count` paths internally and
returns a single blended path, which throws away the very dispersion this
strategy uses for its `score`. For genuine multi-path uncertainty, call
`predictor.generate(...)` directly (same file, `model/kronos.py`) and keep
each sampled path instead of averaging — `KronosForecaster.forecast()` has a
comment marking exactly where to make that change.

## Files

```
strategy/
  data_utils.py   # CSV loading, 5-min -> daily resampling
  indicators.py   # EMA/RSI/MACD/ATR/Bollinger/ADX via the `ta` package
  forecaster.py   # SurrogateForecaster + KronosForecaster, common interface
  engine.py       # StrategyConfig + combined signal logic
  backtest.py     # walk-forward loop, ATR position sizing, performance metrics
run_backtest.py   # CLI entry point
kronos_src/       # trimmed copy of the Kronos repo (model code + sample data)
```

## Running it

```bash
python run_backtest.py                        # surrogate, sample HK data, full history
python run_backtest.py --start 2022-01-01      # limit the backtest window
python run_backtest.py --csv path/to/ohlcv.csv # any 5-min+ OHLCV CSV
python run_backtest.py --real-kronos           # real model (needs internet/GPU)
```

Outputs land in `outputs/`: `equity_curve.png` (equity + drawdown vs
buy-and-hold), `trade_log.csv` (every closed trade), `signal_log.csv` (daily
score/position/reasoning — useful for sanity-checking *why* the strategy did
or didn't act on a given day).

## Tuning knobs

All in `strategy/engine.StrategyConfig` and `strategy/backtest.BacktestConfig`:
`lookback`, `pred_len`, `rebalance_every`, `sample_count` (forecast); score
and RSI/ADX thresholds (entry/exit gating); `atr_stop_mult`, `max_hold_days`,
`risk_per_trade`, `max_gross_exposure`, `cost_bps` (risk/costs).

## Honest caveats

- 9 trades over ~5.7 years on one instrument is not statistically meaningful
  either way — this demonstrates the pipeline works, not that the strategy
  has edge. Real validation needs the actual model, multiple instruments,
  and out-of-sample/walk-forward robustness checks across regimes.
- The surrogate's block-bootstrap has no genuine forward-looking skill, so
  don't compare its Sharpe/CAGR to buy-and-hold as if it were the real
  model's performance.
- No portfolio-level risk controls (correlation across positions, sector
  exposure) since this is single-instrument; extend `backtest.py` before
  running multi-asset.
