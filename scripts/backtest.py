"""Backtest: Kalshi 15-min crypto prediction strategy.

Replays recorded Coinbase + Kalshi data through the fair value model
and simulates trading decisions. Reports P&L and performance metrics.

Usage:
    python scripts/backtest.py [--data-dir src/data] [--asset BTC]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strategy.fair_value import parse_contract_ticker, ContractInfo
from strategy.backtest_engine import BacktestEngine, BacktestConfig, ContractResult
from strategy.models import get_model


# -- Data loading ---------------------------------------------------------

def load_parquet(base: Path, source: str, dtype: str) -> pd.DataFrame:
    d = base / source / dtype
    if not d.exists():
        return pd.DataFrame()
    files = sorted(d.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pa.concat_tables([pq.read_table(f) for f in files]).to_pandas()


def build_mid_series(ob: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Build mid-price series from orderbook data at full resolution."""
    s = ob[ob["symbol"] == symbol]
    if s.empty:
        return pd.DataFrame()

    bids = s[s["side"] == "bid"].groupby("ts")["price"].max()
    asks = s[s["side"] == "ask"].groupby("ts")["price"].min()
    bbo = pd.DataFrame({"bid": bids, "ask": asks}).dropna()
    bbo["mid"] = (bbo["bid"] + bbo["ask"]) / 2
    return bbo.sort_index()


# -- Contract discovery ---------------------------------------------------

def discover_contracts(
    ka_market: pd.DataFrame,
    asset_filter: str | None = None,
) -> list[ContractInfo]:
    """Parse all 15-min contracts from Kalshi market data."""
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


# -- Event stream ---------------------------------------------------------

def build_event_stream(
    cb_mid: pd.DataFrame,
    ka_market: pd.DataFrame,
    contracts: list[ContractInfo],
    cb_symbol: str,
) -> list[tuple[float, str, dict]]:
    """Build time-sorted event stream from all data sources."""
    events = []

    # Coinbase mid updates (subsample to ~1/sec for speed)
    if not cb_mid.empty:
        # Take every Nth row to approximate 1 update/sec
        n_total = len(cb_mid)
        timestamps = cb_mid.index.values
        dt_median = np.median(np.diff(timestamps)) if n_total > 1 else 1.0
        step = max(1, int(1.0 / max(dt_median, 0.001)))  # ~1 per second
        sampled = cb_mid.iloc[::step]

        for ts, row in sampled.iterrows():
            events.append((ts, "cb_mid", {"mid": row["mid"]}))

    # Kalshi market snapshots
    for ticker in ka_market["ticker"].unique():
        if "15M" not in ticker:
            continue
        info = parse_contract_ticker(ticker)
        if info is None or info.cb_symbol != cb_symbol:
            continue

        rows = ka_market[ka_market["ticker"] == ticker].sort_values("ts")
        for _, row in rows.iterrows():
            events.append((row["ts"], "kalshi", {
                "ticker": ticker,
                "yes_bid": row["yes_bid"],
                "yes_ask": row["yes_ask"],
            }))

    # Contract lifecycle events
    for info in contracts:
        events.append((info.window_start, "contract_start", {"info": info}))
        events.append((info.window_end, "contract_end", {"info": info}))

    # Sort by timestamp
    events.sort(key=lambda e: e[0])
    return events


# -- Run backtest ---------------------------------------------------------

def run_backtest(
    events: list[tuple[float, str, dict]],
    contracts: list[ContractInfo],
    cb_mid: pd.DataFrame,
    config: BacktestConfig,
    asset: str,
    model_name: str = "gbm",
) -> BacktestEngine:
    """Process all events through the backtest engine."""
    model = get_model(model_name)
    engine = BacktestEngine(config, model=model)

    # Pre-register all contracts
    for info in contracts:
        engine.register_contract(info)

    # Pre-build a lookup for spot at any timestamp
    mid_ts = cb_mid.index.values
    mid_vals = cb_mid["mid"].values

    for ts, etype, data in events:
        if etype == "cb_mid":
            engine.on_spot_update(ts, asset, data["mid"])

        elif etype == "kalshi":
            engine.on_kalshi_update(
                ts, data["ticker"], data["yes_bid"], data["yes_ask"]
            )

        elif etype == "contract_start":
            info = data["info"]
            # Find spot at window start
            idx = np.searchsorted(mid_ts, info.window_start, side="right") - 1
            if 0 <= idx < len(mid_vals):
                engine._spot_at_open[info.ticker] = mid_vals[idx]

        elif etype == "contract_end":
            info = data["info"]
            # Find spot at window end
            idx = np.searchsorted(mid_ts, info.window_end, side="right") - 1
            if 0 <= idx < len(mid_vals):
                spot_close = mid_vals[idx]
                engine.settle_contract(info.ticker, spot_close)

    return engine


# -- Results reporting ----------------------------------------------------

def print_results(engine: BacktestEngine, config: BacktestConfig) -> None:
    results = engine.results
    if not results:
        print("No contract results.")
        return

    print(f"\n{'=' * 70}")
    print(f"  BACKTEST RESULTS")
    print(f"{'=' * 70}")

    print(f"\n  Config:")
    print(f"    Edge threshold:  {config.edge_threshold_cents}c")
    print(f"    Fee per side:    {config.fee_per_side_cents}c")
    print(f"    Max position:    {config.max_position_per_contract}")
    print(f"    Cooldown:        {config.cooldown_sec}s")
    print(f"    Trade window:    {config.min_time_remaining_sec}s - {config.max_time_remaining_sec}s remaining")
    print(f"    Vol lookback:    {config.vol_lookback_sec}s")
    print(f"    Max spot ret:    {config.max_spot_return_pct}% {'(off)' if config.max_spot_return_pct <= 0 else ''}")
    print(f"    Max vol filter:  {config.max_vol_15m} {'(off)' if config.max_vol_15m <= 0 else ''}")
    print(f"    Early only:      {config.prefer_early_entry}")

    # Aggregate stats
    total_pnl = sum(r.pnl_cents for r in results)
    total_trades = sum(len(r.trades) for r in results)
    contracts_traded = sum(1 for r in results if r.trades)
    contracts_total = len(results)

    all_trades = [t for r in results for t in r.trades]

    print(f"\n  --- Aggregate ---")
    print(f"  Contracts:         {contracts_total}")
    print(f"  Contracts traded:  {contracts_traded}")
    print(f"  Total trades:      {total_trades}")
    print(f"  Total P&L:         {total_pnl:+.1f} cents  (${total_pnl/100:+.2f})")

    if total_trades > 0:
        avg_pnl = total_pnl / total_trades
        print(f"  Avg P&L/trade:     {avg_pnl:+.2f} cents")

        # Win rate
        winning = sum(1 for r in results if r.pnl_cents > 0)
        losing = sum(1 for r in results if r.pnl_cents < 0)
        flat = contracts_traded - winning - losing
        if contracts_traded > 0:
            print(f"  Win/Lose/Flat:     {winning}/{losing}/{flat} ({winning/contracts_traded*100:.0f}% win rate)")

        # Per-trade P&L distribution
        trade_pnls = []
        for r in results:
            if not r.trades:
                continue
            pnl_per_trade = r.pnl_cents / len(r.trades) if r.trades else 0
            trade_pnls.extend([pnl_per_trade] * len(r.trades))

        if trade_pnls:
            arr = np.array(trade_pnls)
            print(f"  P&L per trade:     mean={np.mean(arr):+.2f}  std={np.std(arr):.2f}  median={np.median(arr):+.2f}")
            if np.std(arr) > 0:
                sharpe = np.mean(arr) / np.std(arr)
                print(f"  Sharpe (per trade): {sharpe:.3f}")

        # Edge at entry
        edges = [t.edge_cents for t in all_trades]
        print(f"  Avg edge at entry: {np.mean(edges):.2f} cents")

        # Time remaining at entry
        times = [t.time_remaining for t in all_trades]
        print(f"  Avg time remaining: {np.mean(times):.0f}s")

    # Settlement accuracy
    settled_yes = sum(1 for r in results if r.settled_yes)
    print(f"\n  Settlement:        {settled_yes} YES / {contracts_total - settled_yes} NO ({settled_yes/contracts_total*100:.0f}% YES)")

    # Cumulative P&L curve
    print(f"\n  --- Cumulative P&L ---")
    cum_pnl = 0.0
    max_pnl = 0.0
    max_dd = 0.0
    for r in results:
        cum_pnl += r.pnl_cents
        max_pnl = max(max_pnl, cum_pnl)
        dd = max_pnl - cum_pnl
        max_dd = max(max_dd, dd)

    print(f"  Final P&L:         {cum_pnl:+.1f} cents  (${cum_pnl/100:+.2f})")
    print(f"  Peak P&L:          {max_pnl:+.1f} cents")
    print(f"  Max drawdown:      {max_dd:.1f} cents")

    # Per-contract detail
    print(f"\n  --- Per-Contract Detail ---")
    from datetime import datetime, timezone
    print(f"  {'Ticker':<45s} {'Trades':>6s} {'P&L':>8s} {'Settled':>8s} {'Open$':>10s} {'Close$':>10s}")
    print(f"  {'-'*45} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    cum = 0.0
    for r in results:
        cum += r.pnl_cents
        n_trades = len(r.trades)
        settled = "YES" if r.settled_yes else "NO"
        pnl_str = f"{r.pnl_cents:+.0f}c" if n_trades > 0 else "-"
        print(f"  {r.ticker:<45s} {n_trades:>6d} {pnl_str:>8s} {settled:>8s} "
              f"{r.spot_at_open:>10.2f} {r.spot_at_close:>10.2f}")

    # Trade log for debugging
    if total_trades > 0 and total_trades <= 100:
        print(f"\n  --- Trade Log ---")
        print(f"  {'Time':>10s} {'Ticker':<35s} {'Side':>8s} {'Price':>6s} {'FV':>6s} {'Edge':>6s} {'Remain':>7s}")
        for r in results:
            for t in r.trades:
                dt = datetime.fromtimestamp(t.ts, tz=timezone.utc).strftime("%H:%M:%S")
                print(f"  {dt:>10s} {t.ticker:<35s} {t.side:>8s} {t.price_cents:>5.0f}c "
                      f"{t.fair_value:>5.1f}c {t.edge_cents:>5.1f}c {t.time_remaining:>6.0f}s")


def _sweep_run(events, contracts, cb_mid, asset, config):
    """Run one backtest config and return summary dict."""
    engine = run_backtest(events, contracts, cb_mid, config, asset)
    total_pnl = sum(r.pnl_cents for r in engine.results)
    total_trades = sum(len(r.trades) for r in engine.results)
    traded = [r for r in engine.results if r.trades]
    win = sum(1 for r in traded if r.pnl_cents > 0)
    win_pct = win / len(traded) * 100 if traded else 0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in engine.results:
        cum += r.pnl_cents
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    pnl_per = total_pnl / total_trades if total_trades else 0
    return {
        "trades": total_trades, "pnl": total_pnl, "pnl_per": pnl_per,
        "win_pct": win_pct, "max_dd": max_dd,
    }


def run_param_sweep(
    events: list[tuple[float, str, dict]],
    contracts: list[ContractInfo],
    cb_mid: pd.DataFrame,
    asset: str,
) -> None:
    """Sweep key parameters and show results."""
    print(f"\n{'=' * 70}")
    print(f"  PARAMETER SWEEP (with ML filters)")
    print(f"{'=' * 70}")

    header = (f"  {'Edge':>5s} {'Fee':>4s} {'MaxP':>4s} {'Cool':>4s} "
              f"{'SpRet':>5s} {'Early':>5s} "
              f"{'Trades':>7s} {'P&L':>10s} {'$/trade':>8s} {'Win%':>5s} {'MaxDD':>6s}")
    sep = (f"  {'-'*5} {'-'*4} {'-'*4} {'-'*4} "
           f"{'-'*5} {'-'*5} "
           f"{'-'*7} {'-'*10} {'-'*8} {'-'*5} {'-'*6}")
    print(f"\n{header}")
    print(sep)

    results_list = []

    for edge_thresh in [3, 5, 7, 10, 15]:
        for fee in [3, 5, 7]:
            for max_pos in [1, 5]:
                for cooldown in [5, 30]:
                    for max_spot_ret in [0.0, 0.05, 0.1]:
                        for early_only in [False, True]:
                            config = BacktestConfig(
                                edge_threshold_cents=edge_thresh,
                                fee_per_side_cents=fee,
                                max_position_per_contract=max_pos,
                                cooldown_sec=cooldown,
                                max_spot_return_pct=max_spot_ret,
                                prefer_early_entry=early_only,
                            )
                            r = _sweep_run(events, contracts, cb_mid, asset, config)
                            r["config"] = config
                            results_list.append(r)

                            sp_str = f"{max_spot_ret:.0%}" if max_spot_ret > 0 else "off"
                            early_str = "yes" if early_only else "no"
                            pnl_str = f"{r['pnl']:+.0f}c"
                            print(f"  {edge_thresh:>5d} {fee:>4d} {max_pos:>4d} {cooldown:>4d} "
                                  f"{sp_str:>5s} {early_str:>5s} "
                                  f"{r['trades']:>7d} {pnl_str:>10s} {r['pnl_per']:>+7.1f}c "
                                  f"{r['win_pct']:>4.0f}% {r['max_dd']:>5.0f}c")

    # Show top 10 by P&L
    results_list.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"\n  --- Top 10 Configurations by P&L ---")
    print(f"\n{header}")
    print(sep)
    for r in results_list[:10]:
        c = r["config"]
        sp_str = f"{c.max_spot_return_pct:.0%}" if c.max_spot_return_pct > 0 else "off"
        early_str = "yes" if c.prefer_early_entry else "no"
        pnl_str = f"{r['pnl']:+.0f}c"
        print(f"  {c.edge_threshold_cents:>5.0f} {c.fee_per_side_cents:>4.0f} "
              f"{c.max_position_per_contract:>4d} {c.cooldown_sec:>4.0f} "
              f"{sp_str:>5s} {early_str:>5s} "
              f"{r['trades']:>7d} {pnl_str:>10s} {r['pnl_per']:>+7.1f}c "
              f"{r['win_pct']:>4.0f}% {r['max_dd']:>5.0f}c")

    # Show top 10 by $/trade (min 10 trades)
    profitable = [r for r in results_list if r["trades"] >= 10]
    profitable.sort(key=lambda x: x["pnl_per"], reverse=True)
    print(f"\n  --- Top 10 Configurations by $/trade (min 10 trades) ---")
    print(f"\n{header}")
    print(sep)
    for r in profitable[:10]:
        c = r["config"]
        sp_str = f"{c.max_spot_return_pct:.0%}" if c.max_spot_return_pct > 0 else "off"
        early_str = "yes" if c.prefer_early_entry else "no"
        pnl_str = f"{r['pnl']:+.0f}c"
        print(f"  {c.edge_threshold_cents:>5.0f} {c.fee_per_side_cents:>4.0f} "
              f"{c.max_position_per_contract:>4d} {c.cooldown_sec:>4.0f} "
              f"{sp_str:>5s} {early_str:>5s} "
              f"{r['trades']:>7d} {pnl_str:>10s} {r['pnl_per']:>+7.1f}c "
              f"{r['win_pct']:>4.0f}% {r['max_dd']:>5.0f}c")


# -- Main -----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi 15-min strategy backtest")
    parser.add_argument("--data-dir", default="src/data")
    parser.add_argument("--asset", default="BTC", help="Asset to backtest (BTC, ETH, SOL)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--edge", type=float, default=5.0, help="Edge threshold (cents)")
    parser.add_argument("--fee", type=float, default=5.0, help="Fee per side (cents)")
    parser.add_argument("--max-pos", type=int, default=5, help="Max position per contract")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Cooldown seconds")
    parser.add_argument("--max-spot-ret", type=float, default=0.1, help="Max spot return %% filter (0=off)")
    parser.add_argument("--max-vol", type=float, default=0.0, help="Max vol_15m filter (0=off)")
    parser.add_argument("--early-only", action="store_true", help="Only trade with >9min remaining")
    parser.add_argument("--model", default="gbm", help="Model name (gbm, logistic, ...)")
    args = parser.parse_args()

    base = Path(args.data_dir)
    asset = args.asset.upper()
    cb_symbol = f"{asset}-USD"

    print(f"Loading data for {asset}...")
    cb_ob = load_parquet(base, "coinbase", "orderbook")
    ka_market = load_parquet(base, "kalshi", "kalshi_market")

    print(f"Building {cb_symbol} mid-price series...")
    cb_mid = build_mid_series(cb_ob, cb_symbol)
    if cb_mid.empty:
        print(f"No Coinbase data for {cb_symbol}")
        return
    print(f"  {len(cb_mid):,} mid-price observations")

    print("Discovering contracts...")
    contracts = discover_contracts(ka_market, asset_filter=asset)
    print(f"  {len(contracts)} contracts found")

    # Filter to contracts where we have Coinbase data covering the full window
    cb_start = cb_mid.index.min()
    cb_end = cb_mid.index.max()
    valid = [c for c in contracts if c.window_start >= cb_start and c.window_end <= cb_end]
    skipped = len(contracts) - len(valid)
    if skipped:
        print(f"  Skipping {skipped} contracts outside Coinbase data range")
    contracts = valid
    print(f"  {len(contracts)} contracts with full Coinbase coverage")

    if not contracts:
        print("No valid contracts to backtest.")
        return

    print("Building event stream...")
    events = build_event_stream(cb_mid, ka_market, contracts, cb_symbol)
    print(f"  {len(events):,} events")

    config = BacktestConfig(
        edge_threshold_cents=args.edge,
        fee_per_side_cents=args.fee,
        max_position_per_contract=args.max_pos,
        cooldown_sec=args.cooldown,
        max_spot_return_pct=args.max_spot_ret,
        max_vol_15m=args.max_vol,
        prefer_early_entry=args.early_only,
    )

    print(f"Running backtest with model={args.model}...")
    engine = run_backtest(events, contracts, cb_mid, config, asset, model_name=args.model)
    print_results(engine, config)

    if args.sweep:
        run_param_sweep(events, contracts, cb_mid, asset)


if __name__ == "__main__":
    main()
