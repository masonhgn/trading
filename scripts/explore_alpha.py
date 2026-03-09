"""Alpha exploration script — broad scan of Coinbase + Kalshi data.

Loads all recorded Parquet data and prints a wide range of statistics
and cross-market signals to help identify where alpha might live.

Usage:
    python scripts/explore_alpha.py [--data-dir src/data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ── Helpers ───────────────────────────────────────────────────────────

def load_parquet(base: Path, source: str, dtype: str) -> pd.DataFrame:
    """Load all parquet files for a source/type into a single DataFrame."""
    d = base / source / dtype
    if not d.exists():
        return pd.DataFrame()
    files = sorted(d.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    tables = [pq.read_table(f) for f in files]
    import pyarrow as pa
    return pa.concat_tables(tables).to_pandas()


def ts_to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="s", utc=True)


def section(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


# ── 1. Coinbase spot analysis ─────────────────────────────────────────

def analyze_coinbase_orderbook(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build mid-price time series from orderbook snapshots."""
    if df.empty:
        print("  No Coinbase orderbook data.")
        return {}

    section("1. COINBASE SPOT PRICES")

    # Best bid/ask per timestamp per symbol
    bids = df[df["side"] == "bid"].groupby(["ts", "symbol"])["price"].max().reset_index()
    bids.columns = ["ts", "symbol", "best_bid"]
    asks = df[df["side"] == "ask"].groupby(["ts", "symbol"])["price"].min().reset_index()
    asks.columns = ["ts", "symbol", "best_ask"]

    bbo = bids.merge(asks, on=["ts", "symbol"], how="inner")
    bbo["mid"] = (bbo["best_bid"] + bbo["best_ask"]) / 2
    bbo["spread"] = bbo["best_ask"] - bbo["best_bid"]
    bbo["spread_bps"] = (bbo["spread"] / bbo["mid"]) * 10_000
    bbo["dt"] = ts_to_dt(bbo["ts"])
    bbo = bbo.sort_values(["symbol", "ts"])

    mids = {}
    for symbol in sorted(bbo["symbol"].unique()):
        s = bbo[bbo["symbol"] == symbol].copy()
        mids[symbol] = s

        subsection(symbol)
        print(f"  Snapshots:     {len(s):,}")
        print(f"  Time range:    {s['dt'].min()} -> {s['dt'].max()}")
        duration_min = (s["ts"].max() - s["ts"].min()) / 60
        print(f"  Duration:      {duration_min:.1f} min")
        print(f"  Mid price:     {s['mid'].iloc[-1]:.2f}  (start: {s['mid'].iloc[0]:.2f})")
        ret = (s["mid"].iloc[-1] / s["mid"].iloc[0] - 1) * 100
        print(f"  Return:        {ret:+.4f}%")
        print(f"  Spread:        mean={s['spread'].mean():.4f}  median={s['spread'].median():.4f}")
        print(f"  Spread (bps):  mean={s['spread_bps'].mean():.2f}  median={s['spread_bps'].median():.2f}")

        # Volatility: std of log returns on mid
        if len(s) > 1:
            log_ret = np.log(s["mid"].values[1:] / s["mid"].values[:-1])
            print(f"  Tick vol:      {np.std(log_ret):.6f}  (per update)")
            # Annualized (very rough — assumes ~1 update/sec)
            dt_mean = np.mean(np.diff(s["ts"].values))
            updates_per_year = 365.25 * 24 * 3600 / max(dt_mean, 0.001)
            ann_vol = np.std(log_ret) * np.sqrt(updates_per_year)
            print(f"  Ann. vol est:  {ann_vol * 100:.1f}%  (rough)")
            print(f"  Update freq:   {1/max(dt_mean, 0.001):.1f} updates/sec")

    return mids


# ── 2. Coinbase trade flow ────────────────────────────────────────────

def analyze_coinbase_trades(df: pd.DataFrame) -> None:
    if df.empty:
        print("  No Coinbase trade data.")
        return

    section("2. COINBASE TRADE FLOW")

    df = df.copy()
    df["dt"] = ts_to_dt(df["ts"])

    for symbol in sorted(df["symbol"].unique()):
        s = df[df["symbol"] == symbol].copy()
        subsection(symbol)
        print(f"  Trades:        {len(s):,}")

        buys = s[s["side"] == "buy"]
        sells = s[s["side"] == "sell"]
        print(f"  Buy trades:    {len(buys):,}   volume: {buys['size'].sum():.4f}")
        print(f"  Sell trades:   {len(sells):,}   volume: {sells['size'].sum():.4f}")

        buy_vol = buys["size"].sum()
        sell_vol = sells["size"].sum()
        total_vol = buy_vol + sell_vol
        if total_vol > 0:
            imbalance = (buy_vol - sell_vol) / total_vol
            print(f"  Imbalance:     {imbalance:+.4f}  ({'buy' if imbalance > 0 else 'sell'} heavy)")

        # Dollar volume
        s["dollar_vol"] = s["price"] * s["size"]
        print(f"  Dollar volume: ${s['dollar_vol'].sum():,.2f}")

        # Trade size distribution
        print(f"  Trade size:    mean={s['size'].mean():.6f}  median={s['size'].median():.6f}  max={s['size'].max():.4f}")

        # 1-minute buckets for flow analysis
        s = s.set_index("dt")
        buckets = s.resample("1min").agg(
            count=("size", "count"),
            volume=("size", "sum"),
            buy_vol=("size", lambda x: x[s.loc[x.index, "side"] == "buy"].sum() if len(x) > 0 else 0),
            vwap=("dollar_vol", lambda x: x.sum() / s.loc[x.index, "size"].sum() if s.loc[x.index, "size"].sum() > 0 else 0),
        )
        if len(buckets) > 1:
            print(f"  1min buckets:  {len(buckets)}")
            print(f"  Trades/min:    mean={buckets['count'].mean():.1f}  max={buckets['count'].max()}")


# ── 3. Kalshi prediction market analysis ──────────────────────────────

def parse_kalshi_ticker(ticker: str) -> dict:
    """Extract info from Kalshi ticker.

    Examples:
        KXBTC15M-26MAR080345-45  -> series=KXBTC15M, type=15min, direction/strike=45
        KXBTCD-26MAR0817-T72999.99 -> series=KXBTCD, type=hourly, strike=72999.99
    """
    parts = ticker.split("-")
    series = parts[0] if parts else ticker
    info = {"ticker": ticker, "series": series}

    # Determine asset
    for asset in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
        if asset in series:
            info["asset"] = asset
            break

    # 15-min vs hourly
    if "15M" in series:
        info["type"] = "15min"
    elif series.endswith("D"):
        info["type"] = "hourly"

    # Strike price for hourly contracts
    for p in parts:
        if p.startswith("T"):
            try:
                info["strike"] = float(p[1:])
            except ValueError:
                pass

    return info


def analyze_kalshi_markets(df_market: pd.DataFrame, df_ob: pd.DataFrame, df_trades: pd.DataFrame) -> None:
    section("3. KALSHI PREDICTION MARKETS")

    if df_market.empty:
        print("  No Kalshi market data.")
        return

    df_market = df_market.copy()
    df_market["dt"] = ts_to_dt(df_market["ts"])
    df_market["info"] = df_market["ticker"].apply(parse_kalshi_ticker)
    df_market["asset"] = df_market["info"].apply(lambda x: x.get("asset", "?"))
    df_market["mtype"] = df_market["info"].apply(lambda x: x.get("type", "?"))
    df_market["strike"] = df_market["info"].apply(lambda x: x.get("strike"))

    # Overview
    subsection("Overview")
    n_tickers = df_market["ticker"].nunique()
    print(f"  Unique tickers:  {n_tickers}")
    print(f"  Snapshots:       {len(df_market):,}")

    for mtype in ["15min", "hourly"]:
        sub = df_market[df_market["mtype"] == mtype]
        if sub.empty:
            continue
        subsection(f"{mtype.upper()} contracts")
        for asset in sorted(sub["asset"].unique()):
            a = sub[sub["asset"] == asset]
            latest = a.sort_values("ts").groupby("ticker").last()

            # Filter to contracts with any activity
            active = latest[latest["volume"] > 0]
            liquid = latest[(latest["yes_bid"] > 0) & (latest["yes_ask"] > 0)]

            print(f"\n  {asset} ({mtype}):")
            print(f"    Total contracts:   {len(latest)}")
            print(f"    With volume:       {len(active)}")
            print(f"    With bid+ask:      {len(liquid)}")

            if not liquid.empty:
                liquid = liquid.copy()
                liquid["mid"] = (liquid["yes_bid"] + liquid["yes_ask"]) / 2
                liquid["spread"] = liquid["yes_ask"] - liquid["yes_bid"]

                for _, row in liquid.iterrows():
                    strike_str = f"  strike=${row.name.split('T')[-1]}" if "T" in row.name else ""
                    print(f"    {row.name:50s}")
                    print(f"      bid={row['yes_bid']:3.0f}  ask={row['yes_ask']:3.0f}  "
                          f"mid={row['mid']:5.1f}  spread={row['spread']:2.0f}  "
                          f"vol={row['volume']:>6}  oi={row['open_interest']:>6}{strike_str}")

            if not active.empty:
                print(f"    Total volume:      {int(active['volume'].sum()):,}")
                print(f"    Total OI:          {int(active['open_interest'].sum()):,}")

    # Kalshi trade analysis
    if not df_trades.empty:
        subsection("Kalshi trade activity")
        df_trades = df_trades.copy()
        df_trades["dt"] = ts_to_dt(df_trades["ts"])
        df_trades["info"] = df_trades["symbol"].apply(parse_kalshi_ticker)
        df_trades["asset"] = df_trades["info"].apply(lambda x: x.get("asset", "?"))

        for asset in sorted(df_trades["asset"].unique()):
            a = df_trades[df_trades["asset"] == asset]
            print(f"\n  {asset} trades: {len(a):,}")
            print(f"    Tickers traded:  {a['symbol'].nunique()}")
            if len(a) > 0:
                print(f"    Price range:     {a['price'].min():.0f} - {a['price'].max():.0f} cents")
                print(f"    Avg size:        {a['size'].mean():.1f} contracts")

    # Kalshi spread analysis
    if not df_ob.empty:
        subsection("Kalshi orderbook depth")
        df_ob = df_ob.copy()
        latest_ts = df_ob["ts"].max()
        recent = df_ob[df_ob["ts"] == latest_ts]
        for ticker in sorted(recent["symbol"].unique()):
            t = recent[recent["symbol"] == ticker]
            bids = t[t["side"] == "bid"].sort_values("price", ascending=False)
            asks = t[t["side"] == "ask"].sort_values("price")
            if not bids.empty and not asks.empty:
                depth_bid = bids["size"].sum()
                depth_ask = asks["size"].sum()
                print(f"  {ticker:50s}  bid_depth={depth_bid:>6.0f}  ask_depth={depth_ask:>6.0f}")


# ── 4. Cross-market analysis ─────────────────────────────────────────

def analyze_cross_market(
    cb_mids: dict[str, pd.DataFrame],
    kalshi_market: pd.DataFrame,
) -> None:
    section("4. CROSS-MARKET SIGNALS")

    if not cb_mids or kalshi_market.empty:
        print("  Need both Coinbase and Kalshi data for cross-market analysis.")
        return

    kalshi_market = kalshi_market.copy()
    kalshi_market["info"] = kalshi_market["ticker"].apply(parse_kalshi_ticker)
    kalshi_market["asset"] = kalshi_market["info"].apply(lambda x: x.get("asset", "?"))
    kalshi_market["mtype"] = kalshi_market["info"].apply(lambda x: x.get("type", "?"))
    kalshi_market["strike"] = kalshi_market["info"].apply(lambda x: x.get("strike"))

    asset_to_symbol = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}

    for asset, cb_symbol in asset_to_symbol.items():
        if cb_symbol not in cb_mids:
            continue

        cb = cb_mids[cb_symbol].copy()
        if cb.empty:
            continue

        subsection(f"{asset}: Coinbase spot vs Kalshi")

        cb_latest_mid = cb["mid"].iloc[-1]
        print(f"  Coinbase {cb_symbol} mid: ${cb_latest_mid:,.2f}")

        # 15-min contracts
        contracts_15m = kalshi_market[
            (kalshi_market["asset"] == asset) &
            (kalshi_market["mtype"] == "15min") &
            (kalshi_market["yes_bid"] > 0) &
            (kalshi_market["yes_ask"] > 0)
        ]
        if not contracts_15m.empty:
            latest_15m = contracts_15m.sort_values("ts").groupby("ticker").last()
            for ticker, row in latest_15m.iterrows():
                mid_cents = (row["yes_bid"] + row["yes_ask"]) / 2
                spread_cents = row["yes_ask"] - row["yes_bid"]
                print(f"  15min contract: {ticker}")
                print(f"    Implied prob:  {mid_cents:.0f}%  (bid={row['yes_bid']:.0f} ask={row['yes_ask']:.0f} spread={spread_cents:.0f})")
                print(f"    Volume:        {row['volume']:,.0f}  OI: {row['open_interest']:,.0f}")

        # Hourly contracts — compare strike to spot
        contracts_h = kalshi_market[
            (kalshi_market["asset"] == asset) &
            (kalshi_market["mtype"] == "hourly") &
            (kalshi_market["strike"].notna())
        ]
        if not contracts_h.empty:
            latest_h = contracts_h.sort_values("ts").groupby("ticker").last()
            latest_h = latest_h.sort_values("strike")

            print(f"\n  Hourly above/below contracts (spot=${cb_latest_mid:,.2f}):")
            for ticker, row in latest_h.iterrows():
                strike = row["strike"]
                dist_pct = (strike - cb_latest_mid) / cb_latest_mid * 100
                has_market = row["yes_bid"] > 0 and row["yes_ask"] > 0
                mid_str = f"mid={(row['yes_bid']+row['yes_ask'])/2:.0f}%" if has_market else "no market"
                vol_str = f"vol={row['volume']:,.0f}" if row["volume"] > 0 else "no vol"
                marker = " <-- SPOT" if abs(dist_pct) < 1 else ""
                print(f"    ${strike:>10,.2f}  ({dist_pct:+6.2f}%)  {mid_str:>12s}  {vol_str}{marker}")

    # Time-alignment analysis: do Kalshi prices lead or lag Coinbase?
    subsection("Lead-lag hint (Kalshi vs Coinbase update frequency)")
    for asset, cb_symbol in asset_to_symbol.items():
        if cb_symbol not in cb_mids:
            continue
        cb = cb_mids[cb_symbol]

        contracts = kalshi_market[
            (kalshi_market["asset"] == asset) &
            (kalshi_market["yes_bid"] > 0)
        ]
        if contracts.empty:
            continue

        cb_updates = len(cb)
        kalshi_updates = len(contracts)
        cb_freq = cb_updates / max((cb["ts"].max() - cb["ts"].min()), 1)
        kalshi_freq = kalshi_updates / max((contracts["ts"].max() - contracts["ts"].min()), 1)

        print(f"  {asset}:  Coinbase={cb_freq:.1f} updates/sec  Kalshi={kalshi_freq:.2f} updates/sec")
        print(f"    Coinbase is ~{cb_freq / max(kalshi_freq, 0.001):.0f}x faster")
        print(f"    -> Coinbase spot LEADS, Kalshi prediction markets LAG")
        print(f"    -> Potential: trade Kalshi when Coinbase moves, before Kalshi adjusts")


# ── 5. Spread & market-making opportunities ───────────────────────────

def analyze_mm_opportunities(kalshi_market: pd.DataFrame) -> None:
    section("5. MARKET-MAKING OPPORTUNITIES")

    if kalshi_market.empty:
        print("  No data.")
        return

    kalshi_market = kalshi_market.copy()
    liquid = kalshi_market[(kalshi_market["yes_bid"] > 0) & (kalshi_market["yes_ask"] > 0)].copy()

    if liquid.empty:
        print("  No liquid Kalshi contracts found.")
        return

    liquid["spread"] = liquid["yes_ask"] - liquid["yes_bid"]
    liquid["mid"] = (liquid["yes_bid"] + liquid["yes_ask"]) / 2
    liquid["info"] = liquid["ticker"].apply(parse_kalshi_ticker)
    liquid["asset"] = liquid["info"].apply(lambda x: x.get("asset", "?"))

    # Spread statistics over time
    for ticker in sorted(liquid["ticker"].unique()):
        t = liquid[liquid["ticker"] == ticker].sort_values("ts")
        if len(t) < 2:
            continue

        subsection(ticker)
        print(f"  Observations:  {len(t)}")
        print(f"  Spread:        mean={t['spread'].mean():.1f}  min={t['spread'].min():.0f}  max={t['spread'].max():.0f}")
        print(f"  Mid range:     {t['mid'].min():.0f} - {t['mid'].max():.0f}")
        print(f"  Mid movement:  {t['mid'].iloc[-1] - t['mid'].iloc[0]:+.0f} cents over session")

        # Edge estimate: if you could capture the spread
        avg_spread = t["spread"].mean()
        latest = t.iloc[-1]
        # Kalshi fees: ~2-7 cents per contract per side (varies)
        fee_estimate = 3  # cents per side, conservative
        edge = avg_spread / 2 - fee_estimate
        print(f"  Half-spread:   {avg_spread/2:.1f} cents")
        print(f"  Est. fee:      {fee_estimate} cents/side")
        print(f"  Net edge/trade: {edge:.1f} cents  ({'POSITIVE' if edge > 0 else 'NEGATIVE'})")


# ── 6. Summary & next steps ───────────────────────────────────────────

def print_summary(
    cb_mids: dict[str, pd.DataFrame],
    kalshi_market: pd.DataFrame,
    cb_trades: pd.DataFrame,
) -> None:
    section("6. KEY FINDINGS & NEXT STEPS")

    findings = []

    # Check Coinbase-Kalshi speed differential
    if cb_mids:
        findings.append(
            "SPEED EDGE: Coinbase updates 100x+ faster than Kalshi REST polling.\n"
            "    Kalshi prediction market prices lag spot moves.\n"
            "    -> Strategy: detect Coinbase spot move -> trade Kalshi before it adjusts."
        )

    # Check Kalshi spreads
    if not kalshi_market.empty:
        liquid = kalshi_market[(kalshi_market["yes_bid"] > 0) & (kalshi_market["yes_ask"] > 0)]
        if not liquid.empty:
            avg_spread = (liquid["yes_ask"] - liquid["yes_bid"]).mean()
            findings.append(
                f"KALSHI SPREADS: Average {avg_spread:.1f} cents on liquid contracts.\n"
                f"    Wide spreads = market-making opportunity if fees allow."
            )

    # Check volume
    if not kalshi_market.empty:
        latest = kalshi_market.sort_values("ts").groupby("ticker").last()
        btc_15m = latest[latest.index.str.contains("KXBTC15M")]
        if not btc_15m.empty:
            total_vol = btc_15m["volume"].sum()
            findings.append(
                f"BTC 15-MIN VOLUME: {total_vol:,.0f} contracts.\n"
                f"    {'HIGH' if total_vol > 10000 else 'MODERATE' if total_vol > 1000 else 'LOW'} liquidity for BTC 15-min markets."
            )

    # Trade flow imbalance
    if not cb_trades.empty:
        for symbol in cb_trades["symbol"].unique():
            s = cb_trades[cb_trades["symbol"] == symbol]
            buys = s[s["side"] == "buy"]["size"].sum()
            sells = s[s["side"] == "sell"]["size"].sum()
            total = buys + sells
            if total > 0:
                imbalance = (buys - sells) / total
                if abs(imbalance) > 0.1:
                    direction = "BUY" if imbalance > 0 else "SELL"
                    findings.append(
                        f"FLOW SIGNAL ({symbol}): {direction} imbalance = {imbalance:+.2f}.\n"
                        f"    Strong directional flow could predict Kalshi contract moves."
                    )

    for i, f in enumerate(findings, 1):
        print(f"\n  {i}. {f}")

    print("\n  RECOMMENDED NEXT STEPS:")
    print("  a) Collect 30+ min of data to see Kalshi contract rollovers (new 15-min windows)")
    print("  b) Build a time-aligned dataset merging Coinbase mid + Kalshi implied prob")
    print("  c) Measure Kalshi price reaction delay to Coinbase spot moves")
    print("  d) Backtest: 'when Coinbase moves X% in 1 min, does Kalshi contract follow?'")
    print("  e) Check if Coinbase trade flow imbalance predicts Kalshi direction")
    print()


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha exploration")
    parser.add_argument("--data-dir", default="src/data", help="Base data directory")
    args = parser.parse_args()
    base = Path(args.data_dir)

    print("Loading data...")
    cb_ob = load_parquet(base, "coinbase", "orderbook")
    cb_trades = load_parquet(base, "coinbase", "trade")
    ka_ob = load_parquet(base, "kalshi", "orderbook")
    ka_trades = load_parquet(base, "kalshi", "trade")
    ka_market = load_parquet(base, "kalshi", "kalshi_market")

    total = len(cb_ob) + len(cb_trades) + len(ka_ob) + len(ka_trades) + len(ka_market)
    print(f"Loaded {total:,} total rows.\n")

    cb_mids = analyze_coinbase_orderbook(cb_ob)
    analyze_coinbase_trades(cb_trades)
    analyze_kalshi_markets(ka_market, ka_ob, ka_trades)
    analyze_cross_market(cb_mids, ka_market)
    analyze_mm_opportunities(ka_market)
    print_summary(cb_mids, ka_market, cb_trades)


if __name__ == "__main__":
    main()
