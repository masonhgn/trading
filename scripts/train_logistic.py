#!/usr/bin/env python3
"""Train the logistic regression fair-value model from tick-level data.

Replays recorded Coinbase trades + orderbook and Kalshi market data through
the streaming feature trackers (FlowTracker, BookImbalanceTracker, MomentumTracker),
builds labeled training samples at evaluation points within each 15-min window,
then fits the LogisticModel.

Usage:
    python scripts/train_logistic.py                        # all assets, default data
    python scripts/train_logistic.py --asset BTC            # single asset
    python scripts/train_logistic.py --data-dir src/data    # custom data dir
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strategy.fair_value import ContractInfo, VolEstimator, parse_contract_ticker
from strategy.features import FlowTracker, BookImbalanceTracker, MomentumTracker
from strategy.models.logistic import LogisticModel, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_logistic")


# -- Data loading -----------------------------------------------------------

def load_parquet(base: Path, source: str, dtype: str) -> pd.DataFrame:
    d = base / source / dtype
    if not d.exists():
        return pd.DataFrame()
    files = sorted(d.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pa.concat_tables([pq.read_table(f) for f in files]).to_pandas()


def build_mid_series(ob: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Build mid-price series from orderbook data."""
    s = ob[ob["symbol"] == symbol]
    if s.empty:
        return pd.DataFrame()

    bids = s[s["side"] == "bid"].groupby("ts")["price"].max()
    asks = s[s["side"] == "ask"].groupby("ts")["price"].min()
    bbo = pd.DataFrame({"bid": bids, "ask": asks}).dropna()
    bbo["mid"] = (bbo["bid"] + bbo["ask"]) / 2
    return bbo.sort_index()


def build_book_depth(ob: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Build bid/ask total depth series from L2 orderbook snapshots."""
    s = ob[ob["symbol"] == symbol].copy()
    if s.empty:
        return pd.DataFrame()

    # Sum size across all levels per side per timestamp
    depth = s.groupby(["ts", "side"])["size"].sum().unstack(fill_value=0)
    result = pd.DataFrame(index=depth.index)
    result["bid_depth"] = depth.get("bid", 0)
    result["ask_depth"] = depth.get("ask", 0)
    return result.sort_index()


# -- Contract discovery -----------------------------------------------------

def discover_contracts(ka_market: pd.DataFrame, asset_filter: str | None = None) -> list[ContractInfo]:
    contracts = []
    for ticker in ka_market["ticker"].unique():
        if "15M" not in ticker:
            continue
        info = parse_contract_ticker(ticker)
        if info is None:
            continue
        if asset_filter and info.asset != asset_filter:
            continue
        contracts.append(info)
    contracts.sort(key=lambda c: c.window_start)
    return contracts


# -- Feature extraction via event replay ------------------------------------

def replay_and_extract(
    cb_mid: pd.DataFrame,
    cb_trades: pd.DataFrame,
    cb_depth: pd.DataFrame,
    ka_market: pd.DataFrame,
    contracts: list[ContractInfo],
    asset: str,
    cb_symbol: str,
    vol_lookback: float = 600.0,
    eval_interval_sec: float = 30.0,
) -> pd.DataFrame:
    """Replay tick data through feature trackers and extract labeled samples.

    For each contract window, we evaluate at regular intervals (every eval_interval_sec)
    between t=60s and t=840s into the window.

    Returns DataFrame with columns matching FEATURE_NAMES + metadata.
    """
    # Initialize trackers
    flow = FlowTracker(lookback_sec=120.0)
    book_imb = BookImbalanceTracker()
    momentum = MomentumTracker(lookback_sec=60.0)
    vol_est = VolEstimator(vol_lookback)

    # Pre-sort data
    mid_ts = cb_mid.index.values
    mid_vals = cb_mid["mid"].values

    trades = cb_trades[cb_trades["symbol"] == cb_symbol].sort_values("ts")
    trade_ts = trades["ts"].values
    trade_size = trades["size"].values
    trade_side = trades["side"].values

    depth_ts = cb_depth.index.values if not cb_depth.empty else np.array([])
    bid_depths = cb_depth["bid_depth"].values if not cb_depth.empty else np.array([])
    ask_depths = cb_depth["ask_depth"].values if not cb_depth.empty else np.array([])

    # Build evaluation schedule: for each contract, sample at regular intervals
    eval_points = []
    for info in contracts:
        ws, we = info.window_start, info.window_end

        # Find spot at open
        idx = np.searchsorted(mid_ts, ws, side="right") - 1
        if idx < 0 or idx >= len(mid_vals):
            continue
        spot_at_open = mid_vals[idx]

        # Find spot at close (for label)
        close_idx = np.searchsorted(mid_ts, we, side="right") - 1
        if close_idx < 0 or close_idx >= len(mid_vals):
            continue
        spot_at_close = mid_vals[close_idx]
        label = 1 if spot_at_close > spot_at_open else 0

        # Generate eval timestamps
        t = ws + 60  # start 1 min in
        while t < we - 60:  # stop 1 min before end
            eval_points.append({
                "eval_ts": t,
                "ticker": info.ticker,
                "window_start": ws,
                "window_end": we,
                "spot_at_open": spot_at_open,
                "label": label,
            })
            t += eval_interval_sec

    if not eval_points:
        return pd.DataFrame()

    eval_points.sort(key=lambda x: x["eval_ts"])
    logger.info("  %d evaluation points across %d contracts", len(eval_points), len(contracts))

    # Replay events in time order, updating trackers
    # We process trades, depth updates, and mid updates as they occur
    trade_idx = 0
    depth_idx = 0
    mid_idx = 0
    current_spot = None

    samples = []

    for ep in eval_points:
        eval_ts = ep["eval_ts"]

        # Advance trade tracker up to eval_ts
        while trade_idx < len(trade_ts) and trade_ts[trade_idx] <= eval_ts:
            flow.update(trade_ts[trade_idx], trade_size[trade_idx], trade_side[trade_idx])
            trade_idx += 1

        # Advance depth tracker up to eval_ts
        while depth_idx < len(depth_ts) and depth_ts[depth_idx] <= eval_ts:
            book_imb.update(bid_depths[depth_idx], ask_depths[depth_idx])
            depth_idx += 1

        # Advance mid/vol/momentum up to eval_ts
        while mid_idx < len(mid_ts) and mid_ts[mid_idx] <= eval_ts:
            p = mid_vals[mid_idx]
            t = mid_ts[mid_idx]
            vol_est.update(t, p)
            momentum.update(t, p)
            current_spot = p
            mid_idx += 1

        if current_spot is None:
            continue

        spot = current_spot
        spot_open = ep["spot_at_open"]
        time_remaining = ep["window_end"] - eval_ts

        if spot <= 0 or spot_open <= 0 or time_remaining <= 0:
            continue

        # Extract features matching LogisticModel.FEATURE_NAMES
        current_return = np.log(spot / spot_open)
        vol_15m = vol_est.vol_15m()
        time_frac = time_remaining / 900.0
        flow_imb = flow.imbalance if flow.imbalance is not None else 0.0
        book_imb_val = book_imb.imbalance if book_imb.imbalance is not None else 0.0
        mom_1m = momentum.momentum if momentum.momentum is not None else 0.0

        samples.append({
            "current_return": current_return,
            "vol_15m": vol_15m,
            "time_frac": time_frac,
            "flow_imbalance": flow_imb,
            "book_imbalance": book_imb_val,
            "momentum_1m": mom_1m,
            # Metadata
            "label": ep["label"],
            "asset": asset,
            "ticker": ep["ticker"],
            "window_start": ep["window_start"],
            "eval_ts": eval_ts,
            "time_remaining": time_remaining,
            "spot": spot,
            "spot_at_open": spot_open,
        })

    return pd.DataFrame(samples)


# -- Main -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train logistic fair-value model")
    parser.add_argument("--data-dir", default="src/data", help="Path to tick data")
    parser.add_argument("--asset", default=None, help="Single asset (BTC, ETH, SOL), or all")
    parser.add_argument("--eval-interval", type=float, default=30.0, help="Seconds between eval points per window")
    parser.add_argument("--output", default=None, help="Model output path (default: models/fv_logistic_v1.pkl)")
    args = parser.parse_args()

    base = Path(args.data_dir)

    # Determine assets
    assets = [args.asset.upper()] if args.asset else ["BTC", "ETH", "SOL"]

    print("=" * 60)
    print("  LOGISTIC MODEL TRAINING (tick-level data)")
    print(f"  Assets: {assets}")
    print("=" * 60)

    # Load Kalshi market data
    logger.info("Loading Kalshi market data...")
    ka_market = load_parquet(base, "kalshi", "kalshi_market")
    if ka_market.empty:
        logger.error("No Kalshi market data found in %s", base / "kalshi" / "kalshi_market")
        return

    # Load Coinbase data
    logger.info("Loading Coinbase orderbook data...")
    cb_ob = load_parquet(base, "coinbase", "orderbook")
    logger.info("Loading Coinbase trade data...")
    cb_trades = load_parquet(base, "coinbase", "trade")

    if cb_ob.empty:
        logger.error("No Coinbase orderbook data found")
        return

    all_features = []

    for asset in assets:
        cb_symbol = f"{asset}-USD"
        logger.info("Processing %s...", asset)

        # Build mid-price series
        cb_mid = build_mid_series(cb_ob, cb_symbol)
        if cb_mid.empty:
            logger.warning("  No Coinbase data for %s, skipping", cb_symbol)
            continue
        logger.info("  %d mid-price observations", len(cb_mid))

        # Build depth series
        cb_depth = build_book_depth(cb_ob, cb_symbol)
        logger.info("  %d depth snapshots", len(cb_depth))

        # Discover contracts
        contracts = discover_contracts(ka_market, asset_filter=asset)
        logger.info("  %d contracts", len(contracts))

        # Filter to contracts with Coinbase coverage
        cb_start = cb_mid.index.min()
        cb_end = cb_mid.index.max()
        contracts = [c for c in contracts if c.window_start >= cb_start and c.window_end <= cb_end]
        logger.info("  %d contracts with full coverage", len(contracts))

        if not contracts:
            continue

        # Replay and extract
        features = replay_and_extract(
            cb_mid, cb_trades, cb_depth, ka_market,
            contracts, asset, cb_symbol,
            eval_interval_sec=args.eval_interval,
        )
        if not features.empty:
            all_features.append(features)
            logger.info("  %d training samples for %s", len(features), asset)

    if not all_features:
        logger.error("No training data generated.")
        return

    df = pd.concat(all_features, ignore_index=True)
    df = df.sort_values("window_start").reset_index(drop=True)
    logger.info("Total training samples: %d", len(df))

    # Summary stats
    print(f"\n  --- Data Summary ---")
    print(f"  Total samples:   {len(df)}")
    print(f"  Positive (YES):  {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"  Assets:          {df['asset'].value_counts().to_dict()}")

    print(f"\n  --- Feature Stats ---")
    for feat in FEATURE_NAMES:
        vals = df[feat]
        print(f"  {feat:<20s}  mean={vals.mean():+.6f}  std={vals.std():.6f}  "
              f"min={vals.min():+.6f}  max={vals.max():+.6f}")

    # Train
    X = df[FEATURE_NAMES].values
    y = df["label"].values

    model = LogisticModel()
    metrics = model.fit(X, y)

    print(f"\n  --- Training Results ---")
    print(f"  Samples:           {metrics['n_samples']}")
    print(f"  Positive:          {metrics['n_positive']} ({metrics['n_positive']/metrics['n_samples']*100:.1f}%)")
    print(f"  CV Accuracy:       {metrics['cv_accuracy_mean']:.4f} +/- {metrics['cv_accuracy_std']:.4f}")
    print(f"  CV Log Loss:       {metrics['cv_logloss_mean']:.4f} +/- {metrics['cv_logloss_std']:.4f}")
    print(f"  Intercept:         {metrics['intercept']:+.4f}")

    print(f"\n  --- Coefficients ---")
    for feat, coef in sorted(metrics["coefficients"].items(), key=lambda x: -abs(x[1])):
        print(f"  {feat:<20s}  {coef:+.6f}")

    # Compare vs GBM on the same data
    print(f"\n  --- vs GBM Baseline ---")
    from strategy.fair_value import compute_fair_value
    from sklearn.metrics import log_loss, accuracy_score

    # Logistic predictions
    from strategy.models.base import MarketState
    logistic_preds = []
    gbm_preds = []
    for _, row in df.iterrows():
        state = MarketState(
            spot=row["spot"],
            spot_at_open=row["spot_at_open"],
            vol_15m=row["vol_15m"],
            time_remaining_sec=row["time_remaining"],
            asset=row["asset"],
            flow_imbalance=row["flow_imbalance"],
            book_imbalance=row["book_imbalance"],
            momentum_1m=row["momentum_1m"],
        )
        logistic_preds.append(model.fair_value(state))
        gbm_preds.append(compute_fair_value(
            row["spot"], row["spot_at_open"],
            row["vol_15m"], row["time_remaining"],
        ))

    logistic_preds = np.array(logistic_preds)
    gbm_preds = np.array(gbm_preds)

    log_ll = log_loss(y, np.clip(logistic_preds, 1e-6, 1-1e-6))
    gbm_ll = log_loss(y, np.clip(gbm_preds, 1e-6, 1-1e-6))
    log_acc = accuracy_score(y, (logistic_preds > 0.5).astype(int))
    gbm_acc = accuracy_score(y, (gbm_preds > 0.5).astype(int))

    print(f"  {'Metric':<20s} {'Logistic':>10s} {'GBM':>10s}")
    print(f"  {'Log Loss':<20s} {log_ll:>10.4f} {gbm_ll:>10.4f}")
    print(f"  {'Accuracy':<20s} {log_acc:>10.4f} {gbm_acc:>10.4f}")

    # Calibration
    print(f"\n  --- Calibration (Logistic) ---")
    bins = [(0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 1.0)]
    for lo, hi in bins:
        mask = (logistic_preds >= lo) & (logistic_preds < hi)
        if mask.sum() > 0:
            actual = y[mask].mean()
            avg_pred = logistic_preds[mask].mean()
            print(f"  Pred {lo:.0%}-{hi:.0%}: n={mask.sum():>5}, "
                  f"avg_pred={avg_pred:.3f}, actual={actual:.3f}, "
                  f"diff={actual - avg_pred:+.3f}")

    # Save
    save_path = args.output or None
    model.save(save_path)
    print(f"\n  Model saved to {model._model_path}")
    print(f"  To backtest: python scripts/backtest.py --model logistic")


if __name__ == "__main__":
    main()
