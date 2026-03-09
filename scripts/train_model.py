#!/usr/bin/env python3
"""Train the ML fair value model for Kalshi 15-min crypto contracts.

Pipeline:
1. Fetch historical 1-minute candles from Coinbase REST API
2. Reconstruct 15-minute contract windows with settlement labels
3. Engineer features at multiple time-remaining snapshots per window
4. Train LightGBM binary classifier
5. Evaluate and save model

Usage:
    python scripts/train_model.py                   # 7 days, default
    python scripts/train_model.py --days 30          # 30 days history
    python scripts/train_model.py --eval-only        # just evaluate existing model
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_model")

DATA_DIR = Path(__file__).parent.parent / "data" / "candles"
MODEL_DIR = Path(__file__).parent.parent / "models"


# ── Step 1: Fetch Coinbase Candles ──────────────────────────────────────

def fetch_coinbase_candles(
    symbol: str,
    days: int = 7,
    granularity: str = "ONE_MINUTE",
    key_file: str = "cdp_api_key.json",
) -> pd.DataFrame:
    """Fetch historical 1-minute candles from Coinbase Advanced Trade API.

    Returns DataFrame with columns: [ts, open, high, low, close, volume].
    Handles pagination (max 300 candles per request).
    """
    from coinbase.rest import RESTClient

    cache_path = DATA_DIR / f"{symbol}_{days}d_{granularity}.parquet"
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 1:
            logger.info("Using cached candles for %s (%.1fh old)", symbol, age_hours)
            return pd.read_parquet(cache_path)

    client = RESTClient(key_file=key_file)

    now = int(time.time())
    start = now - days * 86400

    # 1-minute candles: 300 per request = 5 hours
    chunk_sec = 300 * 60  # 5 hours in seconds
    all_candles = []

    t = start
    request_count = 0
    while t < now:
        end_t = min(t + chunk_sec, now)
        try:
            resp = client.get_candles(
                product_id=symbol,
                start=str(t),
                end=str(end_t),
                granularity=granularity,
            )
            candles = resp.get("candles", resp) if isinstance(resp, dict) else resp
            # Handle both dict and object responses
            if hasattr(candles, "candles"):
                candles = candles.candles

            for c in candles:
                if isinstance(c, dict):
                    row = c
                else:
                    row = {
                        "start": getattr(c, "start", "0"),
                        "open": getattr(c, "open", "0"),
                        "high": getattr(c, "high", "0"),
                        "low": getattr(c, "low", "0"),
                        "close": getattr(c, "close", "0"),
                        "volume": getattr(c, "volume", "0"),
                    }
                all_candles.append({
                    "ts": int(row.get("start", 0)),
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                })
            request_count += 1
            if request_count % 10 == 0:
                logger.info("  %s: fetched %d candles (%d requests)", symbol, len(all_candles), request_count)

        except Exception as e:
            logger.warning("Candle fetch error for %s at %d: %s", symbol, t, e)
            time.sleep(1)

        t = end_t
        # Rate limit: ~10 req/sec is safe
        time.sleep(0.15)

    if not all_candles:
        logger.error("No candles fetched for %s", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Fetched %d candles for %s (%d days, %d requests)",
                len(df), symbol, days, request_count)
    return df


# ── Step 2: Reconstruct 15-Min Windows ─────────────────────────────────

def reconstruct_windows(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Reconstruct Kalshi 15-minute contract windows from 1-min candles.

    Each window:
    - Starts at a 15-minute boundary (xx:00, xx:15, xx:30, xx:45)
    - Lasts 15 minutes
    - Settlement = "yes" if close > open, "no" otherwise

    Returns DataFrame with one row per window.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("ts").reset_index(drop=True)

    # Align to 15-minute boundaries
    min_ts = df["ts"].min()
    max_ts = df["ts"].max()

    # Round down to nearest 15 min
    window_start = min_ts - (min_ts % 900)

    windows = []
    while window_start + 900 <= max_ts:
        window_end = window_start + 900

        # Get candles within this window
        mask = (df["ts"] >= window_start) & (df["ts"] < window_end)
        window_candles = df[mask]

        if len(window_candles) >= 5:  # need reasonable coverage
            open_price = window_candles.iloc[0]["open"]
            close_price = window_candles.iloc[-1]["close"]
            high_price = window_candles["high"].max()
            low_price = window_candles["low"].min()
            total_volume = window_candles["volume"].sum()

            settlement = "yes" if close_price > open_price else "no"

            windows.append({
                "window_start": window_start,
                "window_end": window_end,
                "asset": asset,
                "open_price": open_price,
                "close_price": close_price,
                "high_price": high_price,
                "low_price": low_price,
                "volume": total_volume,
                "settlement": settlement,
                "label": 1 if settlement == "yes" else 0,
                "n_candles": len(window_candles),
            })

        window_start += 900  # next 15-min window

    return pd.DataFrame(windows)


# ── Step 3: Feature Engineering ────────────────────────────────────────

def engineer_features(
    candles: pd.DataFrame,
    windows: pd.DataFrame,
    snapshots_per_window: int = 12,
) -> pd.DataFrame:
    """Generate feature snapshots at multiple time points within each window.

    For each window, we sample at regular intervals from t=60s to t=840s
    (1 min to 14 min into the window), computing features as if we were
    evaluating the signal at that moment.
    """
    if candles.empty or windows.empty:
        return pd.DataFrame()

    candles = candles.sort_values("ts").reset_index(drop=True)

    # Pre-compute arrays for fast lookup
    candle_ts = candles["ts"].values
    candle_close = candles["close"].values
    candle_open = candles["open"].values
    candle_high = candles["high"].values
    candle_low = candles["low"].values

    rows = []

    for _, w in windows.iterrows():
        ws = w["window_start"]
        we = w["window_end"]
        open_price = w["open_price"]
        label = w["label"]
        asset = w["asset"]

        # Sample at regular intervals within the window
        for i in range(1, snapshots_per_window + 1):
            elapsed = i * (840 / snapshots_per_window)
            eval_ts = ws + elapsed
            time_remaining = we - eval_ts

            if time_remaining < 30 or time_remaining > 870:
                continue

            # Find the candle closest to eval_ts
            idx = np.searchsorted(candle_ts, eval_ts, side="right") - 1
            if idx < 0 or idx >= len(candle_ts):
                continue

            spot = candle_close[idx]
            if spot <= 0 or open_price <= 0:
                continue

            # Core features
            log_return = np.log(spot / open_price)
            abs_return = abs(log_return)
            time_frac = time_remaining / 900.0

            # Vol from recent candles (rolling 5-min window)
            vol_window = 300  # 5 minutes
            vol_mask = (candle_ts >= eval_ts - vol_window) & (candle_ts <= eval_ts)
            vol_indices = np.where(vol_mask)[0]

            if len(vol_indices) >= 3:
                vol_prices = candle_close[vol_indices]
                log_rets = np.diff(np.log(vol_prices))
                if len(log_rets) > 0:
                    dt_avg = vol_window / len(vol_indices)
                    var_per_sec = np.var(log_rets) / max(dt_avg, 1)
                    vol_long = np.sqrt(var_per_sec * 900)
                else:
                    vol_long = 0.001
            else:
                vol_long = 0.001

            # Short-term vol (last 60s)
            short_mask = (candle_ts >= eval_ts - 60) & (candle_ts <= eval_ts)
            short_indices = np.where(short_mask)[0]

            if len(short_indices) >= 2:
                short_prices = candle_close[short_indices]
                short_rets = np.diff(np.log(short_prices))
                if len(short_rets) > 0:
                    dt_short = 60 / len(short_indices)
                    vol_short = np.sqrt(np.var(short_rets) / max(dt_short, 1) * 900)
                else:
                    vol_short = vol_long
            else:
                vol_short = vol_long

            vol_ratio = vol_short / vol_long if vol_long > 1e-8 else 1.0

            # Z-score
            sigma_t = vol_long * np.sqrt(time_frac) if time_frac > 0 else 1e-8
            return_z = log_return / sigma_t if sigma_t > 1e-8 else 0.0

            # Momentum (60s price change)
            if len(short_indices) >= 2:
                mom_60s = np.log(candle_close[short_indices[-1]] / candle_close[short_indices[0]])
            else:
                mom_60s = 0.0

            # Asset encoding
            asset_map = {"BTC": 0, "ETH": 1, "SOL": 2}
            asset_code = asset_map.get(asset, 0)

            # Kalshi features not available in historical data — use neutral values
            k_mid = 0.5
            k_spread = 0.0

            rows.append({
                "log_return": log_return,
                "abs_return": abs_return,
                "time_frac": time_frac,
                "vol_long": vol_long,
                "return_z": return_z,
                "vol_ratio": vol_ratio,
                "momentum_60s": mom_60s,
                "asset_code": asset_code,
                "kalshi_mid": k_mid,
                "kalshi_spread": k_spread,
                # Metadata (not features)
                "label": label,
                "asset": asset,
                "window_start": ws,
                "time_remaining": time_remaining,
                "spot": spot,
                "open_price": open_price,
            })

    return pd.DataFrame(rows)


# ── Step 4: Train Model ───────────────────────────────────────────────

FEATURE_COLS = [
    "log_return", "abs_return", "time_frac", "vol_long", "return_z",
    "vol_ratio", "momentum_60s", "asset_code", "kalshi_mid", "kalshi_spread",
]


def train_model(df: pd.DataFrame) -> tuple:
    """Train LightGBM binary classifier on feature data.

    Returns (model, metrics_dict).
    """
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

    X = df[FEATURE_COLS].values
    y = df["label"].values

    logger.info("Training data: %d samples, %.1f%% positive (YES)",
                len(y), y.mean() * 100)

    # Time-based split: train on older data, test on newer
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info("Train: %d samples, Test: %d samples", len(y_train), len(y_test))

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    train_set = lgb.Dataset(X_train, y_train, feature_name=FEATURE_COLS)
    valid_set = lgb.Dataset(X_test, y_test, feature_name=FEATURE_COLS, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50),
        ],
    )

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_logloss": log_loss(y_train, y_pred_train),
        "test_logloss": log_loss(y_test, y_pred_test),
        "train_auc": roc_auc_score(y_train, y_pred_train),
        "test_auc": roc_auc_score(y_test, y_pred_test),
        "train_acc": accuracy_score(y_train, (y_pred_train > 0.5).astype(int)),
        "test_acc": accuracy_score(y_test, (y_pred_test > 0.5).astype(int)),
        "test_positive_rate": y_test.mean(),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "best_iteration": model.best_iteration,
    }

    return model, metrics


def evaluate_calibration(model, df: pd.DataFrame) -> None:
    """Check if predicted probabilities match actual outcomes."""
    from sklearn.metrics import roc_auc_score

    X = df[FEATURE_COLS].values
    y = df["label"].values
    preds = model.predict(X)

    print("\n  CALIBRATION (predicted prob vs actual win rate):")
    bins = [(0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]
    for lo, hi in bins:
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() > 0:
            actual = y[mask].mean()
            avg_pred = preds[mask].mean()
            print(f"    Pred {lo:.0%}-{hi:.0%}: n={mask.sum():>5}, "
                  f"avg_pred={avg_pred:.3f}, actual={actual:.3f}, "
                  f"diff={actual - avg_pred:+.3f}")

    # By time remaining
    print("\n  ACCURACY BY TIME REMAINING:")
    for lo, hi, label in [(0.03, 0.2, "30-180s"), (0.2, 0.5, "3-7min"), (0.5, 0.8, "7-12min"), (0.8, 1.0, "12-15min")]:
        mask = (df["time_frac"].values >= lo) & (df["time_frac"].values < hi)
        if mask.sum() > 0:
            acc = ((preds[mask] > 0.5).astype(int) == y[mask]).mean()
            print(f"    {label}: n={mask.sum():>5}, accuracy={acc:.3f}")

    # By asset
    print("\n  ACCURACY BY ASSET:")
    for code, name in [(0, "BTC"), (1, "ETH"), (2, "SOL")]:
        mask = df["asset_code"].values == code
        if mask.sum() > 0:
            acc = ((preds[mask] > 0.5).astype(int) == y[mask]).mean()
            auc = roc_auc_score(y[mask], preds[mask]) if y[mask].std() > 0 else 0
            print(f"    {name}: n={mask.sum():>5}, accuracy={acc:.3f}, AUC={auc:.3f}")


def print_feature_importance(model) -> None:
    """Show which features matter most."""
    importance = model.feature_importance(importance_type="gain")
    total = importance.sum()
    print("\n  FEATURE IMPORTANCE:")
    for name, imp in sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1]):
        print(f"    {name:<20s} {imp/total*100:6.1f}%")


# ── Step 5: Compare vs GBM Baseline ──────────────────────────────────

def compare_vs_gbm(df: pd.DataFrame, ml_model) -> None:
    """Compare ML model vs the existing GBM analytical model."""
    from sklearn.metrics import log_loss, accuracy_score
    from strategy.fair_value import compute_fair_value as gbm_fv

    X = df[FEATURE_COLS].values
    y = df["label"].values

    ml_preds = ml_model.predict(X)

    # GBM predictions
    gbm_preds = np.array([
        gbm_fv(
            row["spot"], row["open_price"],
            row["vol_long"], row["time_remaining"],
        )
        for _, row in df.iterrows()
    ])

    ml_ll = log_loss(y, np.clip(ml_preds, 1e-6, 1 - 1e-6))
    gbm_ll = log_loss(y, np.clip(gbm_preds, 1e-6, 1 - 1e-6))
    ml_acc = accuracy_score(y, (ml_preds > 0.5).astype(int))
    gbm_acc = accuracy_score(y, (gbm_preds > 0.5).astype(int))

    print("\n" + "=" * 60)
    print("  ML vs GBM COMPARISON (test set)")
    print("=" * 60)
    print(f"  {'Metric':<20s} {'ML':>10s} {'GBM':>10s} {'Diff':>10s}")
    print(f"  {'Log Loss':<20s} {ml_ll:>10.4f} {gbm_ll:>10.4f} {ml_ll - gbm_ll:>+10.4f}")
    print(f"  {'Accuracy':<20s} {ml_acc:>10.4f} {gbm_acc:>10.4f} {ml_acc - gbm_acc:>+10.4f}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="Train ML fair value model")
    parser.add_argument("--days", type=int, default=7, help="Days of history to fetch")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing model")
    args = parser.parse_args()

    # Load config for API key
    config_path = Path(__file__).parent.parent / "config.yaml"
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    key_file = cfg.get("coinbase", {}).get("key_file", "cdp_api_key.json")
    assets = cfg.get("assets", ["BTC", "ETH", "SOL"])

    print("=" * 60)
    print(f"  ML FAIR VALUE MODEL TRAINING")
    print(f"  Assets: {assets} | History: {args.days} days")
    print("=" * 60)

    # Step 1: Fetch candles
    all_candles = {}
    for asset in assets:
        symbol = f"{asset}-USD"
        logger.info("Fetching candles for %s...", symbol)
        all_candles[asset] = fetch_coinbase_candles(
            symbol, days=args.days, key_file=key_file,
        )

    # Step 2: Reconstruct windows
    all_windows = []
    for asset, candles in all_candles.items():
        windows = reconstruct_windows(candles, asset)
        all_windows.append(windows)
        logger.info("%s: %d windows, %.1f%% YES",
                    asset, len(windows),
                    windows["label"].mean() * 100 if len(windows) > 0 else 0)

    windows_df = pd.concat(all_windows, ignore_index=True)
    windows_df = windows_df.sort_values("window_start").reset_index(drop=True)
    logger.info("Total windows: %d", len(windows_df))

    # Step 3: Engineer features
    all_features = []
    for asset, candles in all_candles.items():
        asset_windows = windows_df[windows_df["asset"] == asset]
        features = engineer_features(candles, asset_windows, snapshots_per_window=12)
        all_features.append(features)
        logger.info("%s: %d feature snapshots", asset, len(features))

    features_df = pd.concat(all_features, ignore_index=True)
    features_df = features_df.sort_values("window_start").reset_index(drop=True)
    logger.info("Total training samples: %d", len(features_df))

    if len(features_df) < 100:
        logger.error("Not enough data to train. Need more history.")
        return

    # Save features for analysis
    features_path = DATA_DIR / "training_features.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(features_path, index=False)
    logger.info("Saved features to %s", features_path)

    if args.eval_only:
        model_path = MODEL_DIR / "fv_lgbm_v1.txt"
        if not model_path.exists():
            logger.error("No model found at %s", model_path)
            return
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
        from sklearn.metrics import roc_auc_score
        evaluate_calibration(model, features_df)
        print_feature_importance(model)
        compare_vs_gbm(features_df, model)
        return

    # Step 4: Train
    model, metrics = train_model(features_df)

    print("\n" + "=" * 60)
    print("  TRAINING RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25s} {v:.4f}")
        else:
            print(f"  {k:<25s} {v}")

    # Step 5: Evaluate
    from sklearn.metrics import roc_auc_score
    split_idx = int(len(features_df) * 0.8)
    test_df = features_df.iloc[split_idx:]
    evaluate_calibration(model, test_df)
    print_feature_importance(model)
    compare_vs_gbm(test_df, model)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "fv_lgbm_v1.txt"
    model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    print(f"\n  Model saved to {model_path}")
    print(f"  To use in live strategy, set 'strategy.model: ml' in config.yaml")


if __name__ == "__main__":
    main()
