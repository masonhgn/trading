"""Lead-lag analysis: Coinbase spot vs Kalshi prediction markets.

Measures how quickly Kalshi 15-min contracts react to Coinbase spot moves.
Identifies exploitable delays and quantifies signal strength.

Usage:
    python scripts/lead_lag.py [--data-dir src/data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# -- Data loading ------------------------------------------------------

def load_parquet(base: Path, source: str, dtype: str) -> pd.DataFrame:
    d = base / source / dtype
    if not d.exists():
        return pd.DataFrame()
    files = sorted(d.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pa.concat_tables([pq.read_table(f) for f in files]).to_pandas()


def build_coinbase_mid(ob: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build 1-second mid-price bars from orderbook data."""
    mids = {}
    for symbol in ob["symbol"].unique():
        s = ob[ob["symbol"] == symbol]
        bids = s[s["side"] == "bid"].groupby("ts")["price"].max().rename("best_bid")
        asks = s[s["side"] == "ask"].groupby("ts")["price"].min().rename("best_ask")
        bbo = pd.concat([bids, asks], axis=1).dropna()
        bbo["mid"] = (bbo["best_bid"] + bbo["best_ask"]) / 2

        # Resample to 1-second bars (last mid in each second)
        bbo.index = pd.to_datetime(bbo.index, unit="s", utc=True)
        bars = bbo["mid"].resample("1s").last().dropna()
        bars.name = "mid"
        mids[symbol] = bars.to_frame()
    return mids


def build_kalshi_series(market: pd.DataFrame, min_snapshots: int = 5, min_movement: float = 1.0) -> dict[str, pd.DataFrame]:
    """Build time series of Kalshi contract implied probabilities.

    Only returns contracts that have enough data and actual price movement.
    """
    series = {}
    for ticker in market["ticker"].unique():
        t = market[market["ticker"] == ticker].copy()
        t = t.sort_values("ts")
        t.index = pd.to_datetime(t["ts"], unit="s", utc=True)

        # Only keep rows where there's a real two-sided market
        has_market = (t["yes_bid"] > 0) & (t["yes_ask"] > 0)
        t = t[has_market]
        if len(t) < min_snapshots:
            continue

        t["implied_prob"] = (t["yes_bid"] + t["yes_ask"]) / 2

        # Skip contracts with no meaningful price movement
        if t["implied_prob"].max() - t["implied_prob"].min() < min_movement:
            continue

        series[ticker] = t[["implied_prob", "yes_bid", "yes_ask", "volume"]].copy()
    return series


def build_coinbase_flow(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build 1-second trade flow imbalance from trade data."""
    flows = {}
    for symbol in trades["symbol"].unique():
        s = trades[trades["symbol"] == symbol].copy()
        s["signed_size"] = np.where(s["side"] == "buy", s["size"], -s["size"])
        s["dollar_vol"] = s["price"] * s["size"]
        s.index = pd.to_datetime(s["ts"], unit="s", utc=True)

        bars = s.resample("1s").agg(
            net_flow=("signed_size", "sum"),
            volume=("size", "sum"),
            dollar_vol=("dollar_vol", "sum"),
            n_trades=("size", "count"),
        ).fillna(0)
        flows[symbol] = bars
    return flows


# -- Analysis functions ------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


ASSET_MAP = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}


def get_asset(ticker: str) -> str | None:
    for asset in ASSET_MAP:
        if asset in ticker:
            return asset
    return None


def analyze_time_alignment(
    cb_mids: dict[str, pd.DataFrame],
    kalshi_series: dict[str, pd.DataFrame],
) -> None:
    """Merge Coinbase and Kalshi on time axis, measure correlation at lags."""
    section("1. TIME-ALIGNED CORRELATION ANALYSIS")

    for kalshi_ticker, ka in kalshi_series.items():
        asset = get_asset(kalshi_ticker)
        if not asset or ASSET_MAP[asset] not in cb_mids:
            continue

        cb = cb_mids[ASSET_MAP[asset]]
        if len(ka) < 3 or len(cb) < 10:
            continue

        subsection(f"{kalshi_ticker} vs {ASSET_MAP[asset]}")

        # Forward-fill Kalshi to 1-second grid (since it updates every ~5s)
        ka_1s = ka["implied_prob"].resample("1s").last().ffill()

        # Align time ranges
        start = max(cb.index.min(), ka_1s.index.min())
        end = min(cb.index.max(), ka_1s.index.max())
        cb_aligned = cb.loc[start:end, "mid"]
        ka_aligned = ka_1s.loc[start:end]

        if len(cb_aligned) < 10 or len(ka_aligned) < 10:
            print("  Not enough overlapping data.")
            continue

        # Normalize both to changes
        cb_ret = cb_aligned.pct_change().dropna()
        ka_chg = ka_aligned.diff().dropna()

        # Align indices
        common = cb_ret.index.intersection(ka_chg.index)
        if len(common) < 10:
            print("  Not enough common timestamps after differencing.")
            continue

        cb_ret = cb_ret.loc[common]
        ka_chg = ka_chg.loc[common]

        print(f"  Overlapping period: {start} -> {end}")
        print(f"  Common 1s bars:     {len(common)}")
        print(f"  Coinbase mid range: ${cb_aligned.min():.2f} - ${cb_aligned.max():.2f}")
        print(f"  Kalshi prob range:  {ka_aligned.min():.0f} - {ka_aligned.max():.0f} cents")

        # Cross-correlation at different lags
        # Positive lag = Coinbase leads (we shift Coinbase backward)
        # Negative lag = Kalshi leads
        print(f"\n  Cross-correlation (Coinbase return vs Kalshi prob change):")
        print(f"  {'Lag (sec)':>10s}  {'Correlation':>12s}  {'Interpretation':>30s}")
        print(f"  {'-'*10}  {'-'*12}  {'-'*30}")

        best_corr = 0
        best_lag = 0
        lag_results = []

        for lag in range(-30, 61, 1):
            if lag < 0:
                # Kalshi leads: shift kalshi backward
                cb_shifted = cb_ret.iloc[-lag:]
                ka_shifted = ka_chg.iloc[:lag]
            elif lag > 0:
                # Coinbase leads: shift coinbase backward
                cb_shifted = cb_ret.iloc[:-lag] if lag < len(cb_ret) else cb_ret
                ka_shifted = ka_chg.iloc[lag:]
            else:
                cb_shifted = cb_ret
                ka_shifted = ka_chg

            min_len = min(len(cb_shifted), len(ka_shifted))
            if min_len < 10:
                continue

            cb_arr = cb_shifted.values[:min_len]
            ka_arr = ka_shifted.values[:min_len]

            # Filter out zero-variance
            if np.std(cb_arr) == 0 or np.std(ka_arr) == 0:
                continue

            corr = np.corrcoef(cb_arr, ka_arr)[0, 1]
            lag_results.append((lag, corr))

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        # Print key lags
        key_lags = [-10, -5, 0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60]
        lag_dict = dict(lag_results)
        for lag in key_lags:
            if lag not in lag_dict:
                continue
            corr = lag_dict[lag]
            if lag < 0:
                interp = f"Kalshi leads by {-lag}s"
            elif lag == 0:
                interp = "Simultaneous"
            else:
                interp = f"Coinbase leads by {lag}s"
            marker = " <-- BEST" if lag == best_lag else ""
            print(f"  {lag:>10d}  {corr:>12.4f}  {interp:>30s}{marker}")

        print(f"\n  Peak correlation:   {best_corr:.4f} at lag={best_lag}s")
        if best_lag > 0:
            print(f"  -> Coinbase LEADS Kalshi by ~{best_lag} seconds")
            print(f"  -> This is your exploitable window")
        elif best_lag < 0:
            print(f"  -> Kalshi LEADS Coinbase by ~{-best_lag} seconds (surprising!)")
        else:
            print(f"  -> Roughly simultaneous (no clear lead)")


def analyze_spot_move_events(
    cb_mids: dict[str, pd.DataFrame],
    kalshi_series: dict[str, pd.DataFrame],
) -> None:
    """Event study: when Coinbase moves X%, what happens to Kalshi?

    Uses actual Kalshi update points (not forward-filled) to measure
    real price reactions at the next N Kalshi observations.
    """
    section("2. EVENT STUDY: COINBASE SPOT MOVES -> KALSHI REACTION")

    for kalshi_ticker, ka in kalshi_series.items():
        asset = get_asset(kalshi_ticker)
        if not asset or ASSET_MAP[asset] not in cb_mids:
            continue

        cb = cb_mids[ASSET_MAP[asset]]
        if len(ka) < 5 or len(cb) < 60:
            continue

        subsection(f"{kalshi_ticker} vs {ASSET_MAP[asset]}")

        # Use raw Kalshi observations (not forward-filled) for reaction measurement
        ka_raw = ka["implied_prob"]

        # But use forward-filled for "value at event time"
        ka_ffill = ka["implied_prob"].resample("1s").last().ffill()

        start = max(cb.index.min(), ka_raw.index.min())
        end = min(cb.index.max(), ka_raw.index.max())
        cb_aligned = cb.loc[start:end, "mid"]

        if len(cb_aligned) < 60:
            continue

        print(f"  Kalshi raw observations: {len(ka_raw)}")
        avg_gap = ka_raw.index.to_series().diff().mean().total_seconds()
        print(f"  Avg time between Kalshi updates: {avg_gap:.0f}s")

        # Detect "events": Coinbase moves > threshold in rolling windows
        for window_sec in [10, 30, 60]:
            cb_rolling_ret = cb_aligned.pct_change(periods=window_sec).dropna() * 100

            for threshold in [0.02, 0.05, 0.10]:
                up_events = cb_rolling_ret[cb_rolling_ret > threshold]
                down_events = cb_rolling_ret[cb_rolling_ret < -threshold]

                if len(up_events) < 3 and len(down_events) < 3:
                    continue

                print(f"\n  Window={window_sec}s, threshold={threshold}%:")
                print(f"    Up moves (>{threshold}%):    {len(up_events)}")
                print(f"    Down moves (<-{threshold}%): {len(down_events)}")

                # Measure Kalshi reaction using next N actual Kalshi observations
                for direction, events in [("UP", up_events), ("DOWN", down_events)]:
                    if len(events) < 3:
                        continue

                    # For each event, find the Kalshi value at event time (ffill)
                    # and at the next 1, 2, 3, 5 actual Kalshi updates
                    reactions_by_update = {n: [] for n in [1, 2, 3, 5]}
                    reactions_by_time = {t: [] for t in [5, 10, 30, 60]}

                    ka_times = ka_raw.index
                    for event_time in events.index:
                        ka_at_event = ka_ffill.asof(event_time)
                        if pd.isna(ka_at_event):
                            continue

                        # Next N Kalshi observations after the event
                        future_obs = ka_times[ka_times > event_time]
                        for n_updates, lst in reactions_by_update.items():
                            if len(future_obs) >= n_updates:
                                future_val = ka_raw.loc[future_obs[n_updates - 1]]
                                lst.append(future_val - ka_at_event)

                        # Time-based (using ffill for exact time lookup)
                        for dt_sec, lst in reactions_by_time.items():
                            future_time = event_time + pd.Timedelta(seconds=dt_sec)
                            if future_time <= ka_ffill.index.max():
                                future_val = ka_ffill.asof(future_time)
                                if not pd.isna(future_val):
                                    lst.append(future_val - ka_at_event)

                    expected_sign = 1 if direction == "UP" else -1

                    print(f"    {direction} events -> Kalshi reaction (by next Kalshi update):")
                    print(f"    {'After':>12s}  {'Mean':>8s}  {'Median':>8s}  {'StdDev':>8s}  {'N':>5s}  {'Hit%':>6s}")
                    for n_updates in [1, 2, 3, 5]:
                        r = reactions_by_update[n_updates]
                        if not r:
                            continue
                        arr = np.array(r)
                        nonzero = arr[arr != 0]
                        hit_rate = np.mean(np.sign(nonzero) == expected_sign) * 100 if len(nonzero) > 0 else 0
                        label = f"+{n_updates} updates"
                        print(f"    {label:>12s}  {np.mean(arr):>+8.2f}  {np.median(arr):>+8.2f}  "
                              f"{np.std(arr):>8.2f}  {len(arr):>5d}  {hit_rate:>5.1f}%")

                    print(f"    {direction} events -> Kalshi reaction (by wall time):")
                    print(f"    {'After':>12s}  {'Mean':>8s}  {'Median':>8s}  {'StdDev':>8s}  {'N':>5s}  {'Hit%':>6s}")
                    for dt_sec in [5, 10, 30, 60]:
                        r = reactions_by_time[dt_sec]
                        if not r:
                            continue
                        arr = np.array(r)
                        nonzero = arr[arr != 0]
                        hit_rate = np.mean(np.sign(nonzero) == expected_sign) * 100 if len(nonzero) > 0 else 0
                        label = f"+{dt_sec}s"
                        print(f"    {label:>12s}  {np.mean(arr):>+8.2f}  {np.median(arr):>+8.2f}  "
                              f"{np.std(arr):>8.2f}  {len(arr):>5d}  {hit_rate:>5.1f}%")


def analyze_flow_signal(
    cb_flows: dict[str, pd.DataFrame],
    kalshi_series: dict[str, pd.DataFrame],
) -> None:
    """Does Coinbase trade flow imbalance predict Kalshi moves?"""
    section("3. TRADE FLOW -> KALSHI PREDICTION")

    for kalshi_ticker, ka in kalshi_series.items():
        asset = get_asset(kalshi_ticker)
        if not asset or ASSET_MAP[asset] not in cb_flows:
            continue

        flow = cb_flows[ASSET_MAP[asset]]
        if len(ka) < 3 or len(flow) < 30:
            continue

        subsection(f"Coinbase {ASSET_MAP[asset]} flow -> {kalshi_ticker}")

        # Forward-fill Kalshi
        ka_1s = ka["implied_prob"].resample("1s").last().ffill()

        start = max(flow.index.min(), ka_1s.index.min())
        end = min(flow.index.max(), ka_1s.index.max())

        flow_aligned = flow.loc[start:end]
        ka_aligned = ka_1s.loc[start:end]

        if len(flow_aligned) < 30:
            print("  Not enough overlapping data.")
            continue

        # Cumulative flow over rolling windows
        for window in [10, 30, 60]:
            cum_flow = flow_aligned["net_flow"].rolling(f"{window}s").sum().dropna()
            ka_future_chg = ka_aligned.diff(window).shift(-window).dropna()

            common = cum_flow.index.intersection(ka_future_chg.index)
            if len(common) < 10:
                continue

            x = cum_flow.loc[common].values
            y = ka_future_chg.loc[common].values

            if np.std(x) == 0 or np.std(y) == 0:
                continue

            corr = np.corrcoef(x, y)[0, 1]

            # Split into buy-heavy and sell-heavy periods
            buy_heavy = x > np.percentile(x, 75)
            sell_heavy = x < np.percentile(x, 25)

            ka_after_buy = y[buy_heavy]
            ka_after_sell = y[sell_heavy]

            print(f"\n  {window}s cumulative flow -> {window}s forward Kalshi change:")
            print(f"    Correlation:       {corr:+.4f}")
            if len(ka_after_buy) > 0:
                print(f"    After buy flow:    Kalshi moves {np.mean(ka_after_buy):+.2f} cents (n={len(ka_after_buy)})")
            if len(ka_after_sell) > 0:
                print(f"    After sell flow:   Kalshi moves {np.mean(ka_after_sell):+.2f} cents (n={len(ka_after_sell)})")


def analyze_kalshi_microstructure(
    kalshi_series: dict[str, pd.DataFrame],
    kalshi_trades: pd.DataFrame,
) -> None:
    """How do Kalshi prices move? Autocorrelation, mean-reversion, momentum."""
    section("4. KALSHI MICROSTRUCTURE")

    for ticker, ka in kalshi_series.items():
        if len(ka) < 5:
            continue

        subsection(ticker)

        prob = ka["implied_prob"]
        changes = prob.diff().dropna()

        print(f"  Observations:   {len(prob)}")
        print(f"  Prob range:     {prob.min():.0f} - {prob.max():.0f} cents")
        print(f"  Total movement: {prob.iloc[-1] - prob.iloc[0]:+.0f} cents")

        if len(changes) < 3:
            continue

        # Autocorrelation of changes
        changes_arr = changes.values
        if np.std(changes_arr) > 0 and len(changes_arr) > 2:
            ac1 = np.corrcoef(changes_arr[:-1], changes_arr[1:])[0, 1]
            print(f"  Autocorr(1):    {ac1:+.4f}  ({'momentum' if ac1 > 0.1 else 'mean-revert' if ac1 < -0.1 else 'noise'})")

        # Spread over time
        if "yes_bid" in ka.columns and "yes_ask" in ka.columns:
            liquid = ka[(ka["yes_bid"] > 0) & (ka["yes_ask"] > 0)]
            if not liquid.empty:
                spreads = liquid["yes_ask"] - liquid["yes_bid"]
                print(f"  Spread:         mean={spreads.mean():.1f}  min={spreads.min():.0f}  max={spreads.max():.0f}")

        # Trade analysis for this ticker
        if not kalshi_trades.empty:
            ticker_trades = kalshi_trades[kalshi_trades["symbol"] == ticker]
            if not ticker_trades.empty:
                print(f"  Trades:         {len(ticker_trades)}")
                print(f"  Trade prices:   {ticker_trades['price'].min():.0f} - {ticker_trades['price'].max():.0f}")
                print(f"  Avg trade size: {ticker_trades['size'].mean():.1f} contracts")

                # Trade direction impact
                if "side" in ticker_trades.columns:
                    for side in ticker_trades["side"].unique():
                        st = ticker_trades[ticker_trades["side"] == side]
                        print(f"  {side:>4s} trades:    {len(st)} ({st['size'].sum():.0f} contracts)")


def print_strategy_summary(
    cb_mids: dict[str, pd.DataFrame],
    kalshi_series: dict[str, pd.DataFrame],
) -> None:
    section("5. STRATEGY IMPLICATIONS")

    print("""
  LEAD-LAG STRATEGY CONCEPT:
  -----------------------------------------------------------------
  1. Monitor Coinbase BTC/ETH/SOL spot price in real-time (sub-second)
  2. When spot moves significantly over a short window:
     - Compute new fair value for Kalshi 15-min contract
     - If Kalshi hasn't adjusted yet, take the stale price
  3. Key parameters to optimize:
     - Spot move threshold (how big a move to trigger)
     - Lookback window (5s? 10s? 30s?)
     - How to estimate fair Kalshi value from spot move
     - Position sizing and max exposure

  FAIR VALUE ESTIMATION:
  -----------------------------------------------------------------
  For "will BTC go up in next 15 min?" contracts:
  - If BTC is UP +0.05% with 10 min remaining -> prob should increase
  - The magnitude depends on current vol and time remaining
  - Simple model: P(up) = Phi(current_return / (vol * sqrt(time_remaining)))
  - More sophisticated: use order flow as additional signal

  RISK CONSIDERATIONS:
  -----------------------------------------------------------------
  - Kalshi fees eat into edge (~3-7 cents per side)
  - Need sufficient edge to overcome spread + fees
  - Contract expiry creates gamma risk near settlement
  - Position limits on Kalshi
  - Execution risk: REST API latency for Kalshi orders
""")

    # Quantify the opportunity
    for ticker, ka in kalshi_series.items():
        if len(ka) < 5:
            continue
        prob = ka["implied_prob"]
        total_abs_move = prob.diff().abs().sum()
        duration_min = (ka.index[-1] - ka.index[0]).total_seconds() / 60
        if duration_min > 0:
            move_per_min = total_abs_move / duration_min
            print(f"  {ticker}:")
            print(f"    Total absolute movement:  {total_abs_move:.0f} cents in {duration_min:.0f} min")
            print(f"    Movement rate:            {move_per_min:.1f} cents/min")
            print(f"    If you capture 20% of moves net of fees:")
            net_per_contract = move_per_min * 0.20 - 0.5  # rough fee amortization
            print(f"      ~{net_per_contract:.1f} cents/min/contract")
            print(f"      ~${net_per_contract * 60 / 100:.2f}/hr per contract traded continuously")
    print()


# -- Main --------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Lead-lag analysis")
    parser.add_argument("--data-dir", default="src/data")
    args = parser.parse_args()
    base = Path(args.data_dir)

    print("Loading data...")
    cb_ob = load_parquet(base, "coinbase", "orderbook")
    cb_trades = load_parquet(base, "coinbase", "trade")
    ka_market = load_parquet(base, "kalshi", "kalshi_market")
    ka_trades = load_parquet(base, "kalshi", "trade")

    total = len(cb_ob) + len(cb_trades) + len(ka_market) + len(ka_trades)
    print(f"Loaded {total:,} rows.")

    print("Building time series...")
    cb_mids = build_coinbase_mid(cb_ob)
    kalshi_series = build_kalshi_series(ka_market)
    cb_flows = build_coinbase_flow(cb_trades)

    for sym, df in cb_mids.items():
        print(f"  {sym}: {len(df)} 1s bars")
    for ticker, df in kalshi_series.items():
        print(f"  {ticker}: {len(df)} snapshots")

    analyze_time_alignment(cb_mids, kalshi_series)
    analyze_spot_move_events(cb_mids, kalshi_series)
    analyze_flow_signal(cb_flows, kalshi_series)
    analyze_kalshi_microstructure(kalshi_series, ka_trades)
    print_strategy_summary(cb_mids, kalshi_series)


if __name__ == "__main__":
    main()
