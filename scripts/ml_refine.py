"""ML-based refinement of the Kalshi 15-min strategy.

Extracts rich features at each potential trade point and trains a model
to predict which trades will be profitable. Uses walk-forward validation
to avoid overfitting.

Usage:
    python scripts/ml_refine.py [--data-dir src/data]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strategy.fair_value import parse_contract_ticker, compute_fair_value, VolEstimator


# -- Data loading (reuse from backtest) ------------------------------------

def load_parquet(base: Path, source: str, dtype: str) -> pd.DataFrame:
    d = base / source / dtype
    if not d.exists():
        return pd.DataFrame()
    files = sorted(d.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pa.concat_tables([pq.read_table(f) for f in files]).to_pandas()


def build_mid_series(ob: pd.DataFrame, symbol: str) -> pd.DataFrame:
    s = ob[ob["symbol"] == symbol]
    if s.empty:
        return pd.DataFrame()
    bids = s[s["side"] == "bid"].groupby("ts")["price"].max()
    asks = s[s["side"] == "ask"].groupby("ts")["price"].min()
    bbo = pd.DataFrame({"bid": bids, "ask": asks}).dropna()
    bbo["mid"] = (bbo["bid"] + bbo["ask"]) / 2
    return bbo.sort_index()


def build_trade_flow(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    s = trades[trades["symbol"] == symbol].copy()
    if s.empty:
        return pd.DataFrame()
    s["signed_size"] = np.where(s["side"] == "buy", s["size"], -s["size"])
    s["dollar_vol"] = s["price"] * s["size"]
    return s.sort_values("ts")


# -- Feature extraction ----------------------------------------------------

def extract_features(
    cb_mid: pd.DataFrame,
    cb_flow: pd.DataFrame,
    ka_market: pd.DataFrame,
    asset: str,
) -> pd.DataFrame:
    """Extract features at every Kalshi market snapshot for 15-min contracts.

    Each row represents a potential trade decision point with features
    describing market state and a label (did this contract settle YES?).
    """
    cb_symbol = f"{asset}-USD"
    mid_ts = cb_mid.index.values
    mid_vals = cb_mid["mid"].values

    # Pre-compute flow in 1-second buckets
    flow_buckets = None
    if not cb_flow.empty:
        cb_flow_indexed = cb_flow.copy()
        cb_flow_indexed.index = pd.to_datetime(cb_flow_indexed["ts"], unit="s", utc=True)
        flow_1s = cb_flow_indexed["signed_size"].resample("1s").sum().fillna(0)
        # Convert back to epoch for fast lookups
        flow_ts = flow_1s.index.astype(np.int64) / 1e9
        flow_vals = flow_1s.values
    else:
        flow_ts = np.array([])
        flow_vals = np.array([])

    rows = []

    # Get all 15-min tickers
    tickers_15m = [t for t in ka_market["ticker"].unique() if "15M" in t and asset in t]

    for ticker in sorted(tickers_15m):
        info = parse_contract_ticker(ticker)
        if info is None:
            continue

        # Get Kalshi snapshots for this contract
        ka_rows = ka_market[ka_market["ticker"] == ticker].sort_values("ts")
        if len(ka_rows) < 3:
            continue

        # Find spot at window open
        idx_open = np.searchsorted(mid_ts, info.window_start, side="right") - 1
        if idx_open < 0 or idx_open >= len(mid_vals):
            continue
        spot_at_open = mid_vals[idx_open]

        # Find spot at window end (settlement)
        idx_close = np.searchsorted(mid_ts, info.window_end, side="right") - 1
        if idx_close < 0 or idx_close >= len(mid_vals):
            continue
        spot_at_close = mid_vals[idx_close]
        settled_yes = 1 if spot_at_close > spot_at_open else 0

        # Vol estimator
        vol_est = VolEstimator(lookback_sec=600.0)

        # Process each Kalshi snapshot as a potential trade point
        for _, ka_row in ka_rows.iterrows():
            ts = ka_row["ts"]
            yes_bid = ka_row["yes_bid"]
            yes_ask = ka_row["yes_ask"]

            if yes_bid <= 0 or yes_ask <= 0 or yes_ask <= yes_bid:
                continue

            time_remaining = info.window_end - ts
            if time_remaining < 60 or time_remaining > 900:
                continue

            # Current spot
            idx = np.searchsorted(mid_ts, ts, side="right") - 1
            if idx < 0:
                continue
            spot_now = mid_vals[idx]

            # Feed vol estimator with recent prices
            lookback_start = max(0, np.searchsorted(mid_ts, ts - 600, side="left"))
            for i in range(lookback_start, idx + 1, max(1, (idx - lookback_start) // 200)):
                vol_est.update(mid_ts[i], mid_vals[i])

            vol = vol_est.vol_15m()
            fv = compute_fair_value(spot_now, spot_at_open, vol, time_remaining)
            fv_cents = fv * 100

            kalshi_mid = (yes_bid + yes_ask) / 2
            kalshi_spread = yes_ask - yes_bid
            edge = fv_cents - kalshi_mid

            # Spot return since window open
            spot_return = (spot_now - spot_at_open) / spot_at_open * 100

            # Spot momentum features (returns over different lookbacks)
            momentum_features = {}
            for lb_sec in [5, 10, 30, 60, 120]:
                lb_idx = np.searchsorted(mid_ts, ts - lb_sec, side="right") - 1
                if lb_idx >= 0:
                    past_price = mid_vals[lb_idx]
                    momentum_features[f"ret_{lb_sec}s"] = (spot_now - past_price) / past_price * 100
                else:
                    momentum_features[f"ret_{lb_sec}s"] = 0.0

            # Realized vol over different windows
            vol_features = {}
            for vw in [60, 300]:
                vw_start = max(0, np.searchsorted(mid_ts, ts - vw, side="left"))
                vw_prices = mid_vals[vw_start:idx+1:max(1, (idx-vw_start)//100)]
                if len(vw_prices) > 5:
                    log_rets = np.diff(np.log(vw_prices))
                    vol_features[f"vol_{vw}s"] = np.std(log_rets) * 100
                else:
                    vol_features[f"vol_{vw}s"] = 0.0

            # Trade flow features
            flow_features = {}
            if len(flow_ts) > 0:
                for fw in [10, 30, 60]:
                    fw_start = np.searchsorted(flow_ts, ts - fw, side="left")
                    fw_end = np.searchsorted(flow_ts, ts, side="right")
                    if fw_end > fw_start:
                        flow_features[f"flow_{fw}s"] = float(np.sum(flow_vals[fw_start:fw_end]))
                    else:
                        flow_features[f"flow_{fw}s"] = 0.0
            else:
                for fw in [10, 30, 60]:
                    flow_features[f"flow_{fw}s"] = 0.0

            # Kalshi-specific features
            time_frac = time_remaining / (15 * 60)

            row = {
                "ticker": ticker,
                "ts": ts,
                "asset": asset,
                "time_remaining": time_remaining,
                "time_frac": time_frac,
                "spot_return_pct": spot_return,
                "fair_value": fv_cents,
                "kalshi_mid": kalshi_mid,
                "kalshi_spread": kalshi_spread,
                "edge": edge,
                "abs_edge": abs(edge),
                "edge_sign": 1 if edge > 0 else -1,
                "vol_15m": vol * 100,
                **momentum_features,
                **vol_features,
                **flow_features,
                # Label
                "settled_yes": settled_yes,
                # For P&L calculation
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "spot_at_open": spot_at_open,
                "spot_at_close": spot_at_close,
            }
            rows.append(row)

    return pd.DataFrame(rows)


# -- Model training and evaluation -----------------------------------------

def simulate_trade_pnl(row: pd.Series, fee: float = 5.0) -> float:
    """Simulate P&L for taking one trade based on the edge direction."""
    if row["edge"] > 0:
        # Buy YES at ask
        entry = row["yes_ask"]
        payout = 100.0 if row["settled_yes"] else 0.0
        return payout - entry - fee
    else:
        # Buy NO at (100 - bid)
        entry = 100 - row["yes_bid"]
        payout = 100.0 if not row["settled_yes"] else 0.0
        return payout - entry - fee


def walk_forward_analysis(df: pd.DataFrame, fee: float = 5.0) -> None:
    """Walk-forward validation: train on first N contracts, test on next M."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"{'=' * 70}")

    # Get unique tickers in time order
    tickers = df.sort_values("ts").groupby("ticker")["ts"].min().sort_values().index.tolist()
    n = len(tickers)
    print(f"\n  Total contracts: {n}")

    if n < 10:
        print("  Not enough contracts for walk-forward.")
        return

    # Feature columns
    feature_cols = [c for c in df.columns if c not in [
        "ticker", "ts", "asset", "settled_yes", "yes_bid", "yes_ask",
        "spot_at_open", "spot_at_close", "kalshi_mid", "fair_value",
    ]]

    # Simulate P&L for each row
    df = df.copy()
    df["trade_pnl"] = df.apply(lambda r: simulate_trade_pnl(r, fee), axis=1)
    df["profitable"] = (df["trade_pnl"] > 0).astype(int)

    # Walk-forward splits
    train_sizes = [15, 20, 25]
    all_test_preds = []

    for train_size in train_sizes:
        if train_size >= n - 3:
            continue

        train_tickers = set(tickers[:train_size])
        test_tickers = set(tickers[train_size:])

        train = df[df["ticker"].isin(train_tickers)]
        test = df[df["ticker"].isin(test_tickers)]

        if len(train) < 20 or len(test) < 10:
            continue

        X_train = train[feature_cols].values
        y_train = train["profitable"].values
        X_test = test[feature_cols].values
        y_test = test["profitable"].values

        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=42,
        )
        lgb_model.fit(X_train, y_train)
        lgb_preds = lgb_model.predict(X_test)
        lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
        lgb_acc = accuracy_score(y_test, lgb_preds)

        # Logistic regression (baseline)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_s, y_train)
        lr_preds = lr_model.predict(X_test_s)
        lr_acc = accuracy_score(y_test, lr_preds)

        # Naive baseline: always predict the majority class
        naive_acc = max(y_test.mean(), 1 - y_test.mean())

        # Simulate P&L with ML filter: only trade when model says profitable
        test_df = test.copy()
        test_df["lgb_pred"] = lgb_preds
        test_df["lgb_proba"] = lgb_proba

        # Compare strategies
        no_filter_pnl = test_df["trade_pnl"].sum()
        ml_filter_pnl = test_df[test_df["lgb_pred"] == 1]["trade_pnl"].sum()
        high_conf_pnl = test_df[test_df["lgb_proba"] > 0.6]["trade_pnl"].sum()

        no_filter_trades = len(test_df)
        ml_filter_trades = (test_df["lgb_pred"] == 1).sum()
        high_conf_trades = (test_df["lgb_proba"] > 0.6).sum()

        print(f"\n  --- Train on {train_size} contracts, test on {n - train_size} ---")
        print(f"  Train samples: {len(train)}, Test samples: {len(test)}")
        print(f"  Accuracy:  Naive={naive_acc:.1%}  LogReg={lr_acc:.1%}  LightGBM={lgb_acc:.1%}")
        print(f"\n  P&L comparison (test set):")
        print(f"  {'Strategy':<25s} {'Trades':>7s} {'P&L':>10s} {'$/trade':>10s}")
        print(f"  {'-'*25} {'-'*7} {'-'*10} {'-'*10}")

        for name, pnl, trades in [
            ("No filter (all trades)", no_filter_pnl, no_filter_trades),
            ("ML filter (pred=1)", ml_filter_pnl, ml_filter_trades),
            ("High conf (prob>0.6)", high_conf_pnl, high_conf_trades),
        ]:
            per_trade = pnl / trades if trades > 0 else 0
            print(f"  {name:<25s} {trades:>7d} {pnl:>+9.0f}c {per_trade:>+9.1f}c")

        # Feature importance
        importances = lgb_model.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
        print(f"\n  Top features (LightGBM importance):")
        for feat, imp in feat_imp[:10]:
            bar = "#" * int(imp / max(importances) * 20)
            print(f"    {feat:<25s} {imp:>4d}  {bar}")

        if train_size == train_sizes[-1] or train_size == max(t for t in train_sizes if t < n - 3):
            all_test_preds = list(zip(test_df["ticker"], test_df["lgb_proba"], test_df["trade_pnl"]))

    return all_test_preds


def analyze_edge_calibration(df: pd.DataFrame, fee: float = 5.0) -> None:
    """How well calibrated is our edge estimate?"""
    print(f"\n{'=' * 70}")
    print(f"  EDGE CALIBRATION ANALYSIS")
    print(f"{'=' * 70}")

    df = df.copy()
    df["trade_pnl"] = df.apply(lambda r: simulate_trade_pnl(r, fee), axis=1)

    # Bin by absolute edge
    bins = [0, 3, 5, 7, 10, 15, 20, 30, 50, 100]
    df["edge_bin"] = pd.cut(df["abs_edge"], bins=bins)

    print(f"\n  {'Edge Bin':>15s} {'N':>6s} {'Win%':>6s} {'Avg PnL':>8s} {'Total PnL':>10s}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8} {'-'*10}")

    for edge_bin in df["edge_bin"].cat.categories:
        subset = df[df["edge_bin"] == edge_bin]
        if len(subset) == 0:
            continue
        win_pct = (subset["trade_pnl"] > 0).mean() * 100
        avg_pnl = subset["trade_pnl"].mean()
        total_pnl = subset["trade_pnl"].sum()
        print(f"  {str(edge_bin):>15s} {len(subset):>6d} {win_pct:>5.0f}% {avg_pnl:>+7.1f}c {total_pnl:>+9.0f}c")

    # Bin by time remaining
    print(f"\n  {'Time Remaining':>15s} {'N':>6s} {'Win%':>6s} {'Avg PnL':>8s} {'Avg Edge':>9s}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8} {'-'*9}")

    time_bins = [(60, 180, "1-3 min"), (180, 360, "3-6 min"),
                 (360, 540, "6-9 min"), (540, 840, "9-14 min")]
    for lo, hi, label in time_bins:
        subset = df[(df["time_remaining"] >= lo) & (df["time_remaining"] < hi)]
        if len(subset) == 0:
            continue
        win_pct = (subset["trade_pnl"] > 0).mean() * 100
        avg_pnl = subset["trade_pnl"].mean()
        avg_edge = subset["abs_edge"].mean()
        print(f"  {label:>15s} {len(subset):>6d} {win_pct:>5.0f}% {avg_pnl:>+7.1f}c {avg_edge:>8.1f}c")

    # Bin by spot return magnitude
    print(f"\n  {'|Spot Return|':>15s} {'N':>6s} {'Win%':>6s} {'Avg PnL':>8s}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8}")

    df["abs_spot_ret"] = df["spot_return_pct"].abs()
    ret_bins = [(0, 0.01, "<0.01%"), (0.01, 0.03, "0.01-0.03%"),
                (0.03, 0.1, "0.03-0.1%"), (0.1, 1.0, ">0.1%")]
    for lo, hi, label in ret_bins:
        subset = df[(df["abs_spot_ret"] >= lo) & (df["abs_spot_ret"] < hi)]
        if len(subset) == 0:
            continue
        win_pct = (subset["trade_pnl"] > 0).mean() * 100
        avg_pnl = subset["trade_pnl"].mean()
        print(f"  {label:>15s} {len(subset):>6d} {win_pct:>5.0f}% {avg_pnl:>+7.1f}c")


def analyze_feature_correlations(df: pd.DataFrame, fee: float = 5.0) -> None:
    """Which features correlate most with trade profitability?"""
    print(f"\n{'=' * 70}")
    print(f"  FEATURE-PROFIT CORRELATIONS")
    print(f"{'=' * 70}")

    df = df.copy()
    df["trade_pnl"] = df.apply(lambda r: simulate_trade_pnl(r, fee), axis=1)

    feature_cols = [
        "abs_edge", "edge", "time_remaining", "time_frac", "spot_return_pct",
        "kalshi_spread", "vol_15m", "ret_5s", "ret_10s", "ret_30s", "ret_60s",
        "ret_120s", "vol_60s", "vol_300s", "flow_10s", "flow_30s", "flow_60s",
    ]

    correlations = []
    for col in feature_cols:
        if col in df.columns:
            corr = df[col].corr(df["trade_pnl"])
            correlations.append((col, corr))

    correlations.sort(key=lambda x: -abs(x[1]))

    print(f"\n  {'Feature':<25s} {'Corr with PnL':>15s}")
    print(f"  {'-'*25} {'-'*15}")
    for feat, corr in correlations:
        bar = "#" * int(abs(corr) / max(abs(c) for _, c in correlations) * 15)
        sign = "+" if corr > 0 else "-"
        print(f"  {feat:<25s} {corr:>+14.4f}  {sign}{bar}")


# -- Main -------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ML refinement for Kalshi strategy")
    parser.add_argument("--data-dir", default="src/data")
    parser.add_argument("--fee", type=float, default=5.0)
    args = parser.parse_args()

    base = Path(args.data_dir)

    print("Loading data...")
    cb_ob = load_parquet(base, "coinbase", "orderbook")
    cb_trades = load_parquet(base, "coinbase", "trade")
    ka_market = load_parquet(base, "kalshi", "kalshi_market")

    all_features = []
    for asset in ["BTC", "ETH", "SOL"]:
        cb_symbol = f"{asset}-USD"
        print(f"\nExtracting features for {asset}...")
        cb_mid = build_mid_series(cb_ob, cb_symbol)
        cb_flow = build_trade_flow(cb_trades, cb_symbol)

        if cb_mid.empty:
            print(f"  No data for {cb_symbol}")
            continue

        df = extract_features(cb_mid, cb_flow, ka_market, asset)
        print(f"  {len(df)} feature rows from {df['ticker'].nunique()} contracts")
        all_features.append(df)

    if not all_features:
        print("No features extracted.")
        return

    df_all = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal: {len(df_all)} feature rows across {df_all['ticker'].nunique()} contracts")

    # Analysis
    analyze_edge_calibration(df_all, args.fee)
    analyze_feature_correlations(df_all, args.fee)

    # Per-asset walk-forward
    for asset in df_all["asset"].unique():
        df_asset = df_all[df_all["asset"] == asset]
        if df_asset["ticker"].nunique() >= 10:
            print(f"\n  *** {asset} ***")
            walk_forward_analysis(df_asset, args.fee)

    # Combined walk-forward
    if df_all["ticker"].nunique() >= 20:
        print(f"\n  *** ALL ASSETS COMBINED ***")
        walk_forward_analysis(df_all, args.fee)


if __name__ == "__main__":
    main()
