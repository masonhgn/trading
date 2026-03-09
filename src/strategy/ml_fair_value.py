"""ML-based fair value model for Kalshi 15-min crypto prediction contracts.

Drop-in replacement for compute_fair_value() in fair_value.py.
Uses a LightGBM model trained on historical Coinbase candle data
to predict P(spot_close > spot_open) for a 15-minute window.

Falls back to the GBM analytical model if no trained model is available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "fv_lgbm_v1.txt"

# Lazy-loaded model singleton
_model = None
_model_loaded = False


def _load_model(path: Path | None = None) -> bool:
    """Load the trained LightGBM model from disk."""
    global _model, _model_loaded
    if _model_loaded:
        return _model is not None

    _model_loaded = True
    model_path = path or DEFAULT_MODEL_PATH
    if not model_path.exists():
        logger.warning("ML model not found at %s, using GBM fallback", model_path)
        return False

    try:
        import lightgbm as lgb
        _model = lgb.Booster(model_file=str(model_path))
        logger.info("Loaded ML fair value model from %s", model_path)
        return True
    except Exception as e:
        logger.error("Failed to load ML model: %s", e)
        return False


def compute_features(
    spot_now: float,
    spot_at_open: float,
    vol_15m: float,
    time_remaining_sec: float,
    asset: str = "",
    kalshi_mid: float | None = None,
    kalshi_spread: float | None = None,
    vol_short: float | None = None,
    momentum_60s: float | None = None,
) -> np.ndarray:
    """Compute feature vector for the ML model.

    Core features derived from the same inputs as the GBM model,
    plus optional market microstructure features.

    Returns:
        1D numpy array of features in model-expected order.
    """
    if spot_at_open <= 0:
        spot_at_open = spot_now

    # Core features (always available)
    log_return = np.log(spot_now / spot_at_open) if spot_at_open > 0 else 0.0
    abs_return = abs(log_return)
    time_frac = time_remaining_sec / 900.0  # fraction of 15 min
    time_frac = max(time_frac, 0.0)

    # Vol features
    vol = max(vol_15m, 1e-8)
    sigma_t = vol * np.sqrt(time_frac) if time_frac > 0 else 1e-8
    return_z = log_return / sigma_t if sigma_t > 1e-8 else 0.0

    # Vol ratio (short/long) — indicates vol regime changes
    v_short = vol_short if vol_short is not None else vol
    vol_ratio = v_short / vol if vol > 1e-8 else 1.0

    # Momentum
    mom = momentum_60s if momentum_60s is not None else 0.0

    # Asset encoding (BTC=0, ETH=1, SOL=2)
    asset_map = {"BTC": 0, "ETH": 1, "SOL": 2}
    asset_code = asset_map.get(asset, 0)

    # Kalshi features (0 if not available — model handles missing)
    k_mid = kalshi_mid / 100.0 if kalshi_mid is not None else 0.5
    k_spread = kalshi_spread if kalshi_spread is not None else 0.0

    features = np.array([
        log_return,        # 0: signed return from open
        abs_return,        # 1: magnitude of move
        time_frac,         # 2: time remaining as fraction
        vol,               # 3: 15-min vol estimate
        return_z,          # 4: return normalized by vol*sqrt(T)
        vol_ratio,         # 5: short-term / long-term vol
        mom,               # 6: 60s momentum
        asset_code,        # 7: asset identifier
        k_mid,             # 8: kalshi implied probability
        k_spread,          # 9: kalshi spread (liquidity proxy)
    ], dtype=np.float64)

    return features


def compute_fair_value(
    spot_now: float,
    spot_at_open: float,
    vol_15m: float,
    time_remaining_sec: float,
    asset: str = "",
    kalshi_mid: float | None = None,
    kalshi_spread: float | None = None,
    vol_short: float | None = None,
    momentum_60s: float | None = None,
    model_path: Path | None = None,
) -> float:
    """Compute fair probability that spot finishes above spot_at_open.

    Same interface as fair_value.compute_fair_value() with optional
    extra features. Falls back to GBM model if ML model unavailable.

    Returns:
        Probability in [0, 1] (multiply by 100 for cents).
    """
    if time_remaining_sec <= 0:
        return 1.0 if spot_now > spot_at_open else 0.0

    if not _load_model(model_path):
        from strategy.fair_value import compute_fair_value as gbm_fv
        return gbm_fv(spot_now, spot_at_open, vol_15m, time_remaining_sec)

    features = compute_features(
        spot_now, spot_at_open, vol_15m, time_remaining_sec,
        asset=asset, kalshi_mid=kalshi_mid, kalshi_spread=kalshi_spread,
        vol_short=vol_short, momentum_60s=momentum_60s,
    )

    prob = _model.predict(features.reshape(1, -1))[0]
    return float(np.clip(prob, 0.001, 0.999))


def reset_model() -> None:
    """Force reload of model on next call (useful for retraining)."""
    global _model, _model_loaded
    _model = None
    _model_loaded = False
