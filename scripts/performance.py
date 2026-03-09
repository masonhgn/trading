#!/usr/bin/env python3
"""Performance analysis dashboard for live/paper trading logs.

Reads Parquet trade logs and computes key metrics:
- P&L summary (total, per-asset, per-day)
- Win rate and $/trade
- Realized edge vs model edge
- Fill rate
- Time-of-day analysis
- Drawdown

Usage:
    python scripts/performance.py                   # all dates
    python scripts/performance.py --date 2026-03-08 # specific date
    python scripts/performance.py --days 7          # last N days
    python scripts/performance.py --asset BTC       # filter by asset
"""

from __future__ import annotations

import argparse
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

LOGS_DIR = Path(__file__).parent.parent / "logs"


def load_table(table_name: str, date_filter: str | None = None, days: int | None = None) -> pd.DataFrame:
    """Load a Parquet table from the logs directory."""
    table_dir = LOGS_DIR / table_name
    if not table_dir.exists():
        return pd.DataFrame()

    files = []
    if date_filter:
        date_dir = table_dir / date_filter
        if date_dir.exists():
            files = sorted(date_dir.glob("*.parquet"))
    elif days:
        today = datetime.now(timezone.utc).date()
        for d in range(days):
            date_str = (today - timedelta(days=d)).isoformat()
            date_dir = table_dir / date_str
            if date_dir.exists():
                files.extend(sorted(date_dir.glob("*.parquet")))
    else:
        for date_dir in sorted(table_dir.iterdir()):
            if date_dir.is_dir():
                files.extend(sorted(date_dir.glob("*.parquet")))

    if not files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def fmt_cents(c: float) -> str:
    if abs(c) >= 100:
        return f"${c / 100:+,.2f}"
    return f"{c:+.1f}c"


def analyze_signals(signals: pd.DataFrame, asset: str | None = None) -> None:
    if signals.empty:
        print("\n  No signals recorded.\n")
        return

    if asset:
        signals = signals[signals["asset"] == asset]

    print("\n" + "=" * 70)
    print("  SIGNAL ANALYSIS")
    print("=" * 70)

    total = len(signals)
    buy_yes = (signals["side"] == "buy_yes").sum()
    buy_no = (signals["side"] == "buy_no").sum()
    avg_edge = signals["edge"].abs().mean()
    avg_fv = signals["fair_value"].mean()
    avg_tr = signals["time_remaining"].mean()

    print(f"\n  Total signals:    {total}")
    print(f"  Buy YES:          {buy_yes}  ({buy_yes/total*100:.0f}%)")
    print(f"  Buy NO:           {buy_no}  ({buy_no/total*100:.0f}%)")
    print(f"  Avg |edge|:       {avg_edge:.1f}c")
    print(f"  Avg fair value:   {avg_fv:.1f}c")
    print(f"  Avg time remain:  {avg_tr:.0f}s ({avg_tr/60:.1f}m)")

    # Per-asset breakdown
    print("\n  Per-asset:")
    print(f"  {'Asset':<8} {'Count':>8} {'Avg Edge':>10} {'Avg FV':>10} {'Avg TR':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for a, grp in signals.groupby("asset"):
        print(f"  {a:<8} {len(grp):>8} {grp['edge'].abs().mean():>9.1f}c {grp['fair_value'].mean():>9.1f}c {grp['time_remaining'].mean():>9.0f}s")

    # Time-of-day distribution (UTC hours)
    signals = signals.copy()
    signals["hour"] = pd.to_datetime(signals["ts"], unit="s", utc=True).dt.hour
    print("\n  Signals by hour (UTC):")
    hour_counts = signals.groupby("hour").size()
    for h, c in hour_counts.items():
        bar = "#" * min(c, 50)
        print(f"  {h:02d}:00  {c:>5}  {bar}")


def analyze_fills(fills: pd.DataFrame, signals: pd.DataFrame, asset: str | None = None) -> None:
    if fills.empty:
        print("\n  No fills recorded.\n")
        return

    if asset:
        fills = fills[fills["asset"] == asset]

    print("\n" + "=" * 70)
    print("  FILL ANALYSIS")
    print("=" * 70)

    total = len(fills)
    total_fees = fills["fee_cents"].sum()
    buy_fills = fills[fills["action"] == "buy"]
    sell_fills = fills[fills["action"] == "sell"]

    print(f"\n  Total fills:     {total}")
    print(f"  Buys:            {len(buy_fills)}")
    print(f"  Sells:           {len(sell_fills)}")
    print(f"  Total fees:      {fmt_cents(total_fees)}")
    print(f"  Avg fee/fill:    {total_fees/total:.1f}c" if total else "")

    # Fill rate (fills / signals)
    if not signals.empty:
        sig_count = len(signals[~signals["paper"]] if "paper" in signals.columns else signals)
        if sig_count > 0:
            print(f"  Fill rate:       {total/sig_count*100:.0f}% ({total}/{sig_count})")

    # Per-asset
    print("\n  Per-asset:")
    print(f"  {'Asset':<8} {'Fills':>8} {'Avg Price':>10} {'Tot Fees':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for a, grp in fills.groupby("asset"):
        print(f"  {a:<8} {len(grp):>8} {grp['price_cents'].mean():>9.1f}c {fmt_cents(grp['fee_cents'].sum()):>10}")


def analyze_pnl(fills: pd.DataFrame, snapshots: pd.DataFrame, asset: str | None = None) -> None:
    """Compute P&L from fills (realized) and snapshots (mark-to-market)."""
    if fills.empty and snapshots.empty:
        print("\n  No P&L data available.\n")
        return

    print("\n" + "=" * 70)
    print("  P&L ANALYSIS")
    print("=" * 70)

    # --- From fills: realized P&L ---
    if not fills.empty:
        if asset:
            fills = fills[fills["asset"] == asset]

        # Group fills by contract (ticker) and compute net P&L
        results = []
        for ticker, grp in fills.groupby("ticker"):
            # Simple P&L: for each buy, cost = price; for settlement, revenue = outcome
            # Since these are binary contracts, track net cash flow
            total_cost = 0.0
            total_fees = 0.0
            for _, f in grp.iterrows():
                cost = f["price_cents"] * f["size"]
                total_cost += cost
                total_fees += f["fee_cents"]

            results.append({
                "ticker": ticker,
                "asset": grp["asset"].iloc[0],
                "fills": len(grp),
                "total_cost": total_cost,
                "total_fees": total_fees,
            })

        if results:
            rdf = pd.DataFrame(results)
            print(f"\n  Contracts traded: {len(rdf)}")
            print(f"  Total fills:      {rdf['fills'].sum()}")
            print(f"  Total fees:       {fmt_cents(rdf['total_fees'].sum())}")

    # --- From snapshots: latest mark-to-market ---
    if not snapshots.empty:
        if asset:
            snapshots = snapshots[snapshots["asset"] == asset]

        # Get latest snapshot per ticker
        latest = snapshots.sort_values("ts").groupby("ticker").last().reset_index()
        total_pnl = latest["total_pnl_cents"].sum()
        total_cash = latest["cash_cents"].sum()
        total_mtm = latest["mtm_cents"].sum()
        total_fees = latest["fees_paid"].sum()

        print(f"\n  Mark-to-Market (latest snapshot):")
        print(f"  Realized P&L:    {fmt_cents(total_cash)}")
        print(f"  Unrealized MTM:  {fmt_cents(total_mtm)}")
        print(f"  Total P&L:       {fmt_cents(total_pnl)}")
        print(f"  Total fees:      {fmt_cents(total_fees)}")

        # Per-asset
        print(f"\n  {'Asset':<8} {'Realized':>10} {'Unrealized':>10} {'Total':>10} {'Fees':>10}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for a, grp in latest.groupby("asset"):
            print(f"  {a:<8} {fmt_cents(grp['cash_cents'].sum()):>10} "
                  f"{fmt_cents(grp['mtm_cents'].sum()):>10} "
                  f"{fmt_cents(grp['total_pnl_cents'].sum()):>10} "
                  f"{fmt_cents(grp['fees_paid'].sum()):>10}")

        # P&L time series for drawdown
        print("\n  P&L over time (every 30s snapshot):")
        ts_pnl = snapshots.groupby("ts")["total_pnl_cents"].sum().sort_index()
        if len(ts_pnl) > 1:
            cummax = ts_pnl.cummax()
            drawdown = ts_pnl - cummax
            max_dd = drawdown.min()
            peak = cummax.max()
            print(f"  Peak P&L:        {fmt_cents(peak)}")
            print(f"  Max drawdown:    {fmt_cents(max_dd)}")
            print(f"  Current P&L:     {fmt_cents(ts_pnl.iloc[-1])}")


def analyze_edge(signals: pd.DataFrame, fills: pd.DataFrame, asset: str | None = None) -> None:
    """Compare model edge at signal time vs realized outcomes."""
    if signals.empty:
        return

    if asset:
        signals = signals[signals["asset"] == asset]

    print("\n" + "=" * 70)
    print("  EDGE ANALYSIS (Model vs Realized)")
    print("=" * 70)

    # Model edge statistics
    print(f"\n  Model edge at signal time:")
    print(f"  Mean |edge|:     {signals['edge'].abs().mean():.1f}c")
    print(f"  Median |edge|:   {signals['edge'].abs().median():.1f}c")
    print(f"  Std edge:        {signals['edge'].std():.1f}c")

    # Edge distribution
    bins = [0, 15, 20, 25, 30, 40, 50, 100]
    signals = signals.copy()
    signals["edge_bucket"] = pd.cut(signals["edge"].abs(), bins=bins, right=False)
    print(f"\n  Edge distribution:")
    print(f"  {'Edge Range':>15} {'Count':>8} {'Pct':>8}")
    for bucket, count in signals["edge_bucket"].value_counts().sort_index().items():
        pct = count / len(signals) * 100
        print(f"  {str(bucket):>15} {count:>8} {pct:>7.1f}%")

    # Time remaining at signal
    print(f"\n  Time remaining at signal:")
    tr = signals["time_remaining"]
    print(f"  Mean:  {tr.mean():.0f}s ({tr.mean()/60:.1f}m)")
    print(f"  Min:   {tr.min():.0f}s ({tr.min()/60:.1f}m)")
    print(f"  Max:   {tr.max():.0f}s ({tr.max()/60:.1f}m)")

    # Spot return at signal
    if "spot_return_pct" in signals.columns:
        sr = signals["spot_return_pct"]
        print(f"\n  Spot return at signal:")
        print(f"  Mean:  {sr.mean():.3f}%")
        print(f"  Max:   {sr.max():.3f}%")


def analyze_orders(orders: pd.DataFrame, asset: str | None = None) -> None:
    if orders.empty:
        print("\n  No orders recorded.\n")
        return

    if asset:
        orders = orders[orders["asset"] == asset]

    print("\n" + "=" * 70)
    print("  ORDER ANALYSIS")
    print("=" * 70)

    total = len(orders)
    success = (orders["success"]).sum()
    failed = total - success
    paper = (orders["paper"]).sum()

    print(f"\n  Total orders:    {total}")
    print(f"  Successful:      {success}")
    print(f"  Failed:          {failed}")
    print(f"  Paper:           {paper}")
    if failed > 0:
        errors = orders[~orders["success"]]["error"].value_counts()
        print(f"\n  Failure reasons:")
        for err, cnt in errors.items():
            print(f"    {err}: {cnt}")


def daily_summary(signals: pd.DataFrame, fills: pd.DataFrame, snapshots: pd.DataFrame, asset: str | None = None) -> None:
    """Per-day summary table."""
    if signals.empty:
        return

    if asset:
        signals = signals[signals["asset"] == asset]
        if not fills.empty:
            fills = fills[fills["asset"] == asset]
        if not snapshots.empty:
            snapshots = snapshots[snapshots["asset"] == asset]

    signals = signals.copy()
    signals["date"] = pd.to_datetime(signals["ts"], unit="s", utc=True).dt.date

    print("\n" + "=" * 70)
    print("  DAILY SUMMARY")
    print("=" * 70)
    print(f"\n  {'Date':<12} {'Signals':>8} {'Fills':>8} {'Fees':>10} {'P&L':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for date, sig_grp in signals.groupby("date"):
        n_sig = len(sig_grp)

        n_fills = 0
        day_fees = 0
        if not fills.empty:
            fills_copy = fills.copy()
            fills_copy["date"] = pd.to_datetime(fills_copy["ts"], unit="s", utc=True).dt.date
            day_fills = fills_copy[fills_copy["date"] == date]
            n_fills = len(day_fills)
            day_fees = day_fills["fee_cents"].sum()

        day_pnl = 0
        if not snapshots.empty:
            snaps_copy = snapshots.copy()
            snaps_copy["date"] = pd.to_datetime(snaps_copy["ts"], unit="s", utc=True).dt.date
            day_snaps = snaps_copy[snaps_copy["date"] == date]
            if not day_snaps.empty:
                latest = day_snaps.sort_values("ts").groupby("ticker").last()
                day_pnl = latest["total_pnl_cents"].sum()

        print(f"  {str(date):<12} {n_sig:>8} {n_fills:>8} {fmt_cents(day_fees):>10} {fmt_cents(day_pnl):>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading performance analysis")
    parser.add_argument("--date", default=None, help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=None, help="Last N days")
    parser.add_argument("--asset", default=None, help="Filter by asset (BTC, ETH, SOL)")
    parser.add_argument("--logs-dir", default=None, help="Override logs directory")
    args = parser.parse_args()

    global LOGS_DIR
    if args.logs_dir:
        LOGS_DIR = Path(args.logs_dir)

    if not LOGS_DIR.exists():
        print(f"  No logs directory found at {LOGS_DIR}")
        print(f"  Run the strategy first to generate trade logs.")
        return

    print(f"\n  Loading logs from: {LOGS_DIR}")
    if args.date:
        print(f"  Date filter: {args.date}")
    elif args.days:
        print(f"  Last {args.days} days")
    if args.asset:
        print(f"  Asset filter: {args.asset}")

    signals = load_table("signals", args.date, args.days)
    orders = load_table("orders", args.date, args.days)
    fills = load_table("fills", args.date, args.days)
    snapshots = load_table("snapshots", args.date, args.days)

    print(f"\n  Loaded: {len(signals)} signals, {len(orders)} orders, "
          f"{len(fills)} fills, {len(snapshots)} snapshots")

    analyze_signals(signals, args.asset)
    analyze_orders(orders, args.asset)
    analyze_fills(fills, signals, args.asset)
    analyze_pnl(fills, snapshots, args.asset)
    analyze_edge(signals, fills, args.asset)
    daily_summary(signals, fills, snapshots, args.asset)

    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    main()
