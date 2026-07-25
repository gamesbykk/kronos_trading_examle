"""
Forecast engine wrapping Kronos.

Two implementations behind one interface:

  * KronosForecaster    - the real thing. Loads NeoQuasar/Kronos-* from the
                          Hugging Face Hub and runs KronosPredictor.predict().
                          Requires torch + huggingface_hub + network access to
                          huggingface.co, and ideally a GPU.

  * SurrogateForecaster - a lightweight statistical stand-in with the exact
                          same interface, used for development/testing when
                          the real model/weights aren't reachable. It
                          block-bootstraps historical daily log-returns
                          (preserves volatility clustering better than iid
                          normal noise) plus a small momentum drift term, and
                          produces the same multi-path forecast object the
                          real model would.

Both return a ForecastResult, so strategy code never needs to know which one
it's talking to. Swap via get_forecaster(use_real_kronos=True/False).
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KRONOS_SRC_PATH = Path(__file__).resolve().parent.parent / "kronos_src"


@dataclass
class ForecastResult:
    samples: np.ndarray      # shape (sample_count, pred_len) -- simulated close prices
    last_close: float
    expected_return: float   # mean terminal return across samples
    prob_up: float           # fraction of samples with positive terminal return
    dispersion: float        # std of terminal returns across samples (uncertainty)

    @property
    def score(self) -> float:
        """Risk-adjusted forecast strength: expected return per unit of
        forecast uncertainty. This is the single number the strategy acts on."""
        eps = 1e-9
        return self.expected_return / (self.dispersion + eps)


def _summarize_paths(samples_close: np.ndarray, last_close: float) -> ForecastResult:
    terminal = samples_close[:, -1]
    terminal_ret = terminal / last_close - 1.0
    return ForecastResult(
        samples=samples_close,
        last_close=last_close,
        expected_return=float(np.mean(terminal_ret)),
        prob_up=float(np.mean(terminal_ret > 0)),
        dispersion=float(np.std(terminal_ret)),
    )


class SurrogateForecaster:
    """Block-bootstrap statistical surrogate for Kronos. Same call signature,
    no GPU/network/model weights required. Good enough to validate that the
    strategy and backtest engine behave sensibly; NOT a substitute for the
    real model's forecast skill."""

    def __init__(self, block_size: int = 5, momentum_window: int = 20, momentum_weight: float = 0.15, seed: int = 7):
        self.block_size = block_size
        self.momentum_window = momentum_window
        self.momentum_weight = momentum_weight
        self.rng = np.random.default_rng(seed)

    def forecast(self, df_window: pd.DataFrame, pred_len: int, sample_count: int = 30, **_) -> ForecastResult:
        close = df_window["close"].values.astype(float)
        last_close = close[-1]
        log_ret = np.diff(np.log(close))
        if len(log_ret) < self.block_size * 2:
            raise ValueError("df_window too short for surrogate forecaster")

        # small drift term from recent momentum, damped so it doesn't dominate
        recent = log_ret[-self.momentum_window:]
        drift = float(np.mean(recent)) * self.momentum_weight

        n_blocks_needed = int(np.ceil(pred_len / self.block_size))
        paths = np.empty((sample_count, pred_len))
        for s in range(sample_count):
            blocks = []
            for _ in range(n_blocks_needed):
                start = self.rng.integers(0, len(log_ret) - self.block_size)
                blocks.append(log_ret[start:start + self.block_size])
            path_ret = np.concatenate(blocks)[:pred_len] + drift
            paths[s] = last_close * np.exp(np.cumsum(path_ret))

        return _summarize_paths(paths, last_close)


class KronosForecaster:
    """Real Kronos model wrapper. Requires torch, huggingface_hub, and
    network access to huggingface.co (not available in this sandbox, but
    will work in a normal environment with `pip install -r
    kronos_src/requirements.txt`)."""

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-small",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        max_context: int = 512,
        device: str | None = None,
    ):
        if str(KRONOS_SRC_PATH) not in sys.path:
            sys.path.insert(0, str(KRONOS_SRC_PATH))
        try:
            from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Could not import Kronos model code. Make sure kronos_src/ is "
                "present and torch/einops/huggingface_hub are installed."
            ) from e

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        model = Kronos.from_pretrained(model_name)
        self.predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)

    def forecast(
        self,
        df_window: pd.DataFrame,
        pred_len: int,
        sample_count: int = 30,
        T: float = 1.0,
        top_p: float = 0.9,
    ) -> ForecastResult:
        x_df = df_window[["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = pd.Series(df_window.index)
        last_ts = df_window.index[-1]
        y_timestamp = pd.Series(pd.bdate_range(last_ts, periods=pred_len + 1)[1:])

        pred_dfs = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
        )

        last_close = float(df_window["close"].iloc[-1])

        paths = np.stack(
            [df["close"].values for df in pred_dfs],
            axis=0,
        )

        return _summarize_paths(paths, last_close)


def get_forecaster(use_real_kronos: bool = False, **kwargs):
    """Factory. Tries the real model if requested; falls back to the
    surrogate with a clear warning if that's not possible in this
    environment."""
    if use_real_kronos:
        try:
            return KronosForecaster(**kwargs)
        except Exception as e:
            warnings.warn(
                f"Falling back to SurrogateForecaster -- real Kronos unavailable ({e})"
            )
    return SurrogateForecaster()
