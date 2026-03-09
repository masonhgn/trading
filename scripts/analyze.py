#!/usr/bin/env python3
"""Post-session analysis — the feedback loop.

Reads all Parquet trade logs and produces actionable insights:
- Settlement outcomes (win rate, model accuracy)
- Edge analysis (predicted vs realized)
- TP/SL effectiveness
- Parameter optimization suggestions

Usage:
    python scripts/analyze.py                    # analyze today
    python scripts/analyze.py --date 2026-03-08  # specific date
    python scripts/analyze.py --days 7           # last 7 days
    python scripts/analyze.py --sweep            # run TP/SL parameter sweep
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

LOGS_DIR = Path(__file__).parent.parent / "logs"


def load_table(name: str, date_filter: str | None = None, days: int = 1) -> pd.DataFrame:
    """Load a Parquet log table, optionally filtered by date."""
    table_dir = LOGS_DIR / name
    if not table_dir.exists():
        return pd.DataFrame()

    if date_filter:
        dirs = [table_dir / date_filter]
    elif days > 1:
        today = datetime.now(timezone.utc).date()
        dirs = [table_dir / (today - timedelta(days=i)).isoformat() for i in range(days)]
    else:
        dirs = [table_dir / datetime.now(timezone.utc).strftime("%Y-%m-%d")]

    dfs = []
    for d in dirs:
        if d.exists():
            for f in sorted(d.glob("*.parquet")):
                dfs.append(pd.read_parquet(f))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def analyze_settlements(df: pd.DataFrame) -> None:
    """Analyze settlement outcomes."""
    print("\n" + "=" * 70)
    print("  SETTLEMENT ANALYSIS")
    print("=" * 70)

    if df.empty:
        print("  No settlement data yet.")
        print("  (Settlements are recorded when contracts expire with a position)")
        return

    total = len(df)
    wins = df["model_correct"].sum()
    losses = total - wins
    win_rate = wins / total * 100

    total_pnl = df["pnl_cents"].sum()
    avg_pnl = df["pnl_cents"].mean()
    total_fees = df["fees_cents"].sum()

    print(f"  Total settled:    {total}")
    print(f"  Wins:             {wins} ({win_rate:.1f}%)")
    print(f"  Losses:           {losses} ({100 - win_rate:.1f}%)")
    print(f"  Total P&L:        {total_pnl:+.0f}c (${total_pnl/100:+.2f})")
    print(f"  Avg P&L/trade:    {avg_pnl:+.1f}c")
    print(f"  Total fees:       {total_fees:.0f}c")
    print()

    # By asset
    for asset, grp in df.groupby("asset"):
        n = len(grp)
        w = grp["model_correct"].sum()
        pnl = grp["pnl_cents"].sum()
        print(f"  {asset}: {n} trades, {w}/{n} wins ({w/n*100:.0f}%), P&L={pnl:+.0f}c")

    # By side
    print()
    for side, grp in df.groupby("entry_side"):
        n = len(grp)
        w = grp["model_correct"].sum()
        pnl = grp["pnl_cents"].sum()
        print(f"  {side}: {n} trades, {w}/{n} wins ({w/n*100:.0f}%), P&L={pnl:+.0f}c")


def analyze_signals(df: pd.DataFrame) -> None:
    """Analyze signal quality."""
    print("\n" + "=" * 70)
    print("  SIGNAL ANALYSIS")
    print("=" * 70)

    if df.empty:
        print("  No signal data.")
        return

    total = len(df)
    avg_edge = df["edge"].abs().mean()
    avg_tr = df["time_remaining"].mean()

    buy_yes = (df["side"] == "buy_yes").sum()
    buy_no = (df["side"] == "buy_no").sum()

    print(f"  Total evaluations: {total}")
    print(f"  Buy YES signals:   {buy_yes}")
    print(f"  Buy NO signals:    {buy_no}")
    print(f"  Avg |edge|:        {avg_edge:.1f}c")
    print(f"  Avg time remain:   {avg_tr:.0f}s ({avg_tr/60:.1f}m)")
    print()

    # Edge distribution
    print("  Edge distribution:")
    for threshold in [5, 10, 15, 20, 25, 30]:
        count = (df["edge"].abs() >= threshold).sum()
        print(f"    |edge| >= {threshold}c:  {count} ({count/total*100:.1f}%)")

    # By asset
    print()
    for asset, grp in df.groupby("asset"):
        ae = grp["edge"].abs().mean()
        print(f"  {asset}: {len(grp)} evals, avg |edge|={ae:.1f}c")


def analyze_fills(df: pd.DataFrame) -> None:
    """Analyze fill data."""
    print("\n" + "=" * 70)
    print("  FILL ANALYSIS")
    print("=" * 70)

    if df.empty:
        print("  No fill data.")
        return

    buys = df[df["action"] == "buy"]
    sells = df[df["action"] == "sell"]

    print(f"  Total fills: {len(df)}")
    print(f"  Buys:  {len(buys)}")
    print(f"  Sells: {len(sells)}")

    total_cost = (buys["price_cents"] * buys["size"]).sum()
    total_revenue = (sells["price_cents"] * sells["size"]).sum()
    total_fees = df["fee_cents"].sum()
    print(f"  Total cost:    {total_cost:.0f}c")
    print(f"  Total revenue: {total_revenue:.0f}c")
    print(f"  Total fees:    {total_fees:.0f}c")

    if not buys.empty:
        avg_buy = buys["price_cents"].mean()
        print(f"  Avg buy price: {avg_buy:.1f}c")
    if not sells.empty:
        avg_sell = sells["price_cents"].mean()
        print(f"  Avg sell price: {avg_sell:.1f}c")


def analyze_model_accuracy(signals: pd.DataFrame, settlements: pd.DataFrame) -> None:
    """Compare model predictions to settlement outcomes."""
    print("\n" + "=" * 70)
    print("  MODEL ACCURACY")
    print("=" * 70)

    if settlements.empty:
        print("  No settlement data to validate model against.")
        return

    if signals.empty:
        print("  No signal data.")
        return

    # For each settlement, find the signal with the closest timestamp
    # and compare fair value prediction to outcome
    results = []
    for _, s in settlements.iterrows():
        ticker = s["ticker"]
        ticker_sigs = signals[signals["ticker"] == ticker]
        if ticker_sigs.empty:
            continue

        # Use the signal closest to entry time
        fv = s.get("fair_value_at_entry", 0)
        result = s["settlement_result"]
        entry_side = s["entry_side"]

        # Model said YES probability = fv/100
        # Settlement: yes=100, no=0
        settlement_val = 100 if result == "yes" else 0
        model_error = abs(fv - settlement_val)

        results.append({
            "ticker": ticker,
            "asset": s["asset"],
            "fair_value": fv,
            "settlement": settlement_val,
            "model_error": model_error,
            "side": entry_side,
            "correct": s["model_correct"],
        })

    if not results:
        print("  Could not match signals to settlements.")
        return

    rdf = pd.DataFrame(results)
    avg_error = rdf["model_error"].mean()
    accuracy = rdf["correct"].mean() * 100

    print(f"  Matched trades:     {len(rdf)}")
    print(f"  Model accuracy:     {accuracy:.1f}%")
    print(f"  Avg model error:    {avg_error:.1f}c")
    print(f"  (lower error = better calibration)")
    print()

    # By FV bucket
    print("  Accuracy by fair value range:")
    bins = [(0, 25), (25, 40), (40, 60), (60, 75), (75, 100)]
    for lo, hi in bins:
        bucket = rdf[(rdf["fair_value"] >= lo) & (rdf["fair_value"] < hi)]
        if not bucket.empty:
            acc = bucket["correct"].mean() * 100
            print(f"    FV {lo}-{hi}c: {len(bucket)} trades, {acc:.0f}% accuracy")


def parameter_sweep(signals: pd.DataFrame, settlements: pd.DataFrame) -> None:
    """Sweep TP/SL thresholds to find optimal values.

    Uses settlement data to simulate different exit strategies.
    """
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (TP/SL optimization)")
    print("=" * 70)

    if settlements.empty:
        print("  Need settlement data to run parameter sweep.")
        print("  Run the strategy for a while and come back.")
        return

    # For each settlement, we know entry price, settlement result, and pnl
    # Simulate what would happen with different TP/SL combos
    # (This is approximate — we don't have tick-by-tick price history)
    print("  Note: This uses settlement outcomes only (no intra-window price data)")
    print("  For precise TP/SL optimization, we'd need tick-level price history")
    print()

    # Simple analysis: what if we only traded when edge was above X?
    if not signals.empty:
        print("  Edge threshold sensitivity:")
        print(f"  {'Threshold':>10} {'Signals':>8} {'Avg|Edge|':>10}")
        for threshold in [10, 15, 20, 22, 25, 30]:
            above = signals[signals["edge"].abs() >= threshold]
            if not above.empty:
                ae = above["edge"].abs().mean()
                print(f"  {threshold:>8}c {len(above):>8} {ae:>9.1f}c")
        print()

    # Win rate by entry price bucket
    print("  Win rate by entry price bucket:")
    bins = [(0, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 100)]
    for lo, hi in bins:
        bucket = settlements[
            (settlements["entry_price"] >= lo) & (settlements["entry_price"] < hi)
        ]
        if not bucket.empty:
            wr = bucket["model_correct"].mean() * 100
            avg_pnl = bucket["pnl_cents"].mean()
            print(f"    Entry {lo}-{hi}c: {len(bucket)} trades, "
                  f"WR={wr:.0f}%, avg P&L={avg_pnl:+.1f}c")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-session trade analysis")
    parser.add_argument("--date", type=str, default=None, help="Date to analyze (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=1, help="Number of days to analyze")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    print("=" * 70)
    if args.date:
        print(f"  TRADE ANALYSIS — {args.date}")
    else:
        print(f"  TRADE ANALYSIS — last {args.days} day(s)")
    print("=" * 70)

    # Load all data
    signals = load_table("signals", args.date, args.days)
    fills = load_table("fills", args.date, args.days)
    settlements = load_table("settlements", args.date, args.days)
    snapshots = load_table("snapshots", args.date, args.days)

    print(f"\n  Data loaded:")
    print(f"    Signals:     {len(signals):>6} rows")
    print(f"    Fills:       {len(fills):>6} rows")
    print(f"    Settlements: {len(settlements):>6} rows")
    print(f"    Snapshots:   {len(snapshots):>6} rows")

    analyze_settlements(settlements)
    analyze_signals(signals)
    analyze_fills(fills)
    analyze_model_accuracy(signals, settlements)

    if args.sweep:
        parameter_sweep(signals, settlements)

    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)

    if settlements.empty and signals.empty:
        print("  No data yet. Run the strategy and come back.")
    elif settlements.empty:
        print("  Signals are being logged but no settlements yet.")
        print("  Wait for contracts to expire with positions to see outcomes.")
    else:
        wr = settlements["model_correct"].mean() * 100
        avg_pnl = settlements["pnl_cents"].mean()
        if wr > 55:
            print(f"  Model accuracy {wr:.0f}% is promising.")
            if avg_pnl < 0:
                print("  But avg P&L is negative — likely fees eating edge.")
                print("  Consider: increase edge threshold or reduce fees.")
        elif wr > 45:
            print(f"  Model accuracy {wr:.0f}% is borderline.")
            print("  Consider: tighter entry criteria or model improvements.")
        else:
            print(f"  Model accuracy {wr:.0f}% is below breakeven.")
            print("  Consider: pausing live trading and revisiting the model.")

    print()


if __name__ == "__main__":
    main()
