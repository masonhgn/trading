#!/usr/bin/env python3
"""Live strategy monitor — compact tile-based dashboard.

Polls Kalshi API for balance, positions, and resting orders.
Reads strategy Parquet logs for FV, edge, and signal data.
Shows compact per-contract tiles focused on what matters:
fair value, Kalshi price, indicators, and P&L.

Usage:
    python scripts/monitor.py                  # default 3s refresh
    python scripts/monitor.py --interval 5     # 5s refresh
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gateway.kalshi.auth import load_private_key
from gateway.kalshi.client import KalshiRestClient

LOGS_DIR = Path(__file__).parent.parent / "logs"
SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M"]

console = Console()
START_TIME = time.monotonic()


def _fmt_usd(cents: float) -> str:
    return f"${cents / 100:,.2f}"


def _uptime() -> str:
    elapsed = int(time.monotonic() - START_TIME)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load_recent_signals(limit: int = 50):
    import pandas as pd
    sig_dir = LOGS_DIR / "signals"
    if not sig_dir.exists():
        return pd.DataFrame()
    files = sorted(sig_dir.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files[-5:]]
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values("ts", ascending=False).head(limit)


def _load_latest_snapshots():
    import pandas as pd
    snap_dir = LOGS_DIR / "snapshots"
    if not snap_dir.exists():
        return pd.DataFrame()
    files = sorted(snap_dir.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files[-3:]]
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values("ts").drop_duplicates("ticker", keep="last")


def _load_recent_fills(limit: int = 10):
    import pandas as pd
    fill_dir = LOGS_DIR / "fills"
    if not fill_dir.exists():
        return pd.DataFrame()
    files = sorted(fill_dir.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files[-3:]]
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values("ts", ascending=False).head(limit)


# ---------------------------------------------------------------------------
# API data fetching (returns raw dicts, not panels)
# ---------------------------------------------------------------------------

def _fetch_api_data(client: KalshiRestClient) -> dict:
    """Fetch all API data as raw dicts."""
    data = {"balance": {}, "positions": [], "orders": [], "markets": {}}

    try:
        data["balance"] = client.get_balance()
    except Exception as e:
        data["balance_error"] = str(e)[:60]

    try:
        pos_resp = client.get_positions(settlement_status="unsettled", limit=100)
        data["positions"] = [
            p for p in pos_resp.get("market_positions", [])
            if any(p.get("ticker", "").startswith(s) for s in SERIES)
        ]
    except Exception:
        pass

    try:
        orders_resp = client.get_orders(status="resting", limit=50)
        data["orders"] = orders_resp.get("orders", [])
    except Exception:
        pass

    # Fetch market data for active contracts
    for series in SERIES:
        try:
            resp = client.get_markets(series_ticker=series, status="open", limit=10)
            for m in resp.get("markets", []):
                ticker = m.get("ticker", "")
                data["markets"][ticker] = m
        except Exception:
            pass

    return data


# ---------------------------------------------------------------------------
# Header bar
# ---------------------------------------------------------------------------

def _make_header(api_data: dict) -> Panel:
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    bal = api_data.get("balance", {})
    balance = bal.get("balance", 0)
    portfolio = bal.get("portfolio_value", 0)
    total = balance + portfolio

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    grid.add_column(justify="center", ratio=2)
    grid.add_column(justify="right", ratio=1)

    bal_text = Text.assemble(
        ("Cash ", "dim"),
        (_fmt_usd(balance), "green"),
        ("  Portfolio ", "dim"),
        (_fmt_usd(portfolio), "green"),
        ("  Total ", "dim"),
        (_fmt_usd(total), "bold green"),
    )

    grid.add_row(
        Text(f"MONITOR  {now_str}", style="bold cyan"),
        bal_text,
        Text(f"up {_uptime()}", style="dim"),
    )

    n_orders = len(api_data.get("orders", []))
    n_positions = len(api_data.get("positions", []))
    if api_data.get("balance_error"):
        status = Text(f"API error: {api_data['balance_error']}", style="red")
    else:
        status = Text.assemble(
            (f"{n_positions} positions  ", ""),
            (f"{n_orders} resting orders", "dim"),
        )
    grid2 = Table.grid(expand=True, padding=(0, 1))
    grid2.add_column(ratio=1)
    grid2.add_row(status)

    outer = Table.grid(expand=True)
    outer.add_column()
    outer.add_row(grid)
    outer.add_row(grid2)

    return Panel(outer, border_style="cyan", padding=(0, 1))


# ---------------------------------------------------------------------------
# Contract tiles
# ---------------------------------------------------------------------------

def _make_contract_tile(
    ticker: str,
    market_data: dict,
    position_data: dict | None,
    signal_data: dict | None,
    snapshot_data: dict | None,
    resting_orders: list[dict],
) -> Panel:
    """Build a compact tile for a single contract."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", style="dim", min_width=6)
    grid.add_column(justify="left", min_width=14)

    # Kalshi market prices
    yb = market_data.get("yes_bid", 0)
    ya = market_data.get("yes_ask", 0)
    mid = (yb + ya) / 2 if yb and ya else 0
    spread = ya - yb if yb and ya else 0

    bid_ask = Text.assemble(
        (f"{yb}", "green"), ("/", "dim"), (f"{ya}", "red"),
        (f"  mid {mid:.0f}", ""),
    )
    grid.add_row("Kalshi", bid_ask)

    # Fair value + edge from latest signal
    if signal_data:
        fv = signal_data.get("fair_value", 0)
        edge = signal_data.get("edge", 0)
        edge_style = "bold green" if edge > 0 else "bold red" if edge < 0 else ""
        fv_text = Text.assemble(
            (f"{fv:.1f}c", "bold"),
            ("  edge ", "dim"),
            (f"{edge:+.1f}c", edge_style),
        )
        grid.add_row("FV", fv_text)

        # Indicators
        tr = signal_data.get("time_remaining", 0)
        vol = signal_data.get("vol_15m", 0)
        sr = signal_data.get("spot_return_pct", 0)
        spot = signal_data.get("spot", 0)

        tr_min = tr / 60
        tr_style = "green" if tr_min > 5 else "yellow" if tr_min > 2 else "red"
        ind_text = Text.assemble(
            (f"{tr_min:.1f}m", tr_style),
            ("  vol ", "dim"),
            (f"{vol:.4f}", ""),
            ("  sr ", "dim"),
            (f"{sr:.2f}%", ""),
        )
        grid.add_row("T/Vol", ind_text)

        if spot:
            grid.add_row("Spot", Text(f"${spot:,.2f}", style=""))
    else:
        grid.add_row("FV", Text("--", style="dim"))

    # Position + P&L
    # Kalshi REST API fields:
    #   position_fp    = contract count (positive=YES, negative=NO)
    #   realized_pnl_dollars = realized P&L in fixed-point dollars (divide by 100 for $)
    #   fees_paid_dollars    = fees in fixed-point dollars
    #   market_exposure_dollars = dollar exposure (NOT contract count!)
    if position_data:
        pos = int(float(position_data.get("position_fp", 0) or 0))
        rpnl_dollars = float(position_data.get("realized_pnl_dollars", 0) or 0)
        fees_dollars = float(position_data.get("fees_paid_dollars", 0) or 0)
        rpnl_cents = rpnl_dollars * 100
        fees_cents = fees_dollars * 100
        pnl_style = "green" if rpnl_cents >= 0 else "red"
        if pos:
            pos_text = Text.assemble(
                (f"{pos}", "bold"),
                ("  rpnl ", "dim"),
                (f"{rpnl_cents:+.0f}c", pnl_style),
                ("  fees ", "dim"),
                (f"{fees_cents:.0f}c", ""),
            )
            grid.add_row("Pos", pos_text)
        elif rpnl_cents != 0:
            # No position but has realized P&L (already exited)
            pos_text = Text.assemble(
                ("flat", "dim"),
                ("  rpnl ", "dim"),
                (f"{rpnl_cents:+.0f}c", pnl_style),
            )
            grid.add_row("Pos", pos_text)
    elif snapshot_data:
        pos = int(snapshot_data.get("position", 0))
        pnl = snapshot_data.get("total_pnl_cents", 0)
        pnl_style = "green" if pnl >= 0 else "red"
        if pos:
            pos_text = Text.assemble(
                (f"{pos}", "bold"),
                ("  pnl ", "dim"),
                (f"{pnl:+.0f}c", pnl_style),
            )
            grid.add_row("Pos", pos_text)

    # Resting orders
    if resting_orders:
        parts = []
        for o in resting_orders:
            side = o.get("side", "")
            action = o.get("action", "")
            price = o.get("yes_price", o.get("no_price", 0))
            qty = o.get("remaining_count", 0)
            style = "green" if action == "buy" else "red"
            if parts:
                parts.append(("  ", ""))
            parts.append((f"{action[0].upper()}{side[0].upper()} {price}c x{qty}", style))
        grid.add_row("Orders", Text.assemble(*parts))

    # Volume + OI
    vol_traded = market_data.get("volume", 0)
    oi = market_data.get("open_interest", 0)
    if vol_traded or oi:
        grid.add_row("Vol/OI", Text(f"{vol_traded}/{oi}", style="dim"))

    # Short ticker for title (strip series prefix)
    short = ticker
    for s in SERIES:
        if ticker.startswith(s + "-"):
            short = ticker[len(s) + 1:]
            break

    # Border color based on position/edge
    has_pos = (position_data and position_data.get("market_exposure", 0)) or \
              (snapshot_data and snapshot_data.get("position", 0))
    border = "yellow" if has_pos else "blue"

    return Panel(grid, title=short, border_style=border, width=42, padding=(0, 1))


# ---------------------------------------------------------------------------
# P&L summary tile
# ---------------------------------------------------------------------------

def _make_pnl_tile(api_data: dict, signals, snapshots) -> Panel:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", style="dim", min_width=10)
    grid.add_column(justify="left", min_width=20)

    # Aggregate P&L from Kalshi positions API (correct fields)
    total_rpnl_cents = 0.0
    total_fees_cents = 0.0
    total_contracts = 0
    for p in api_data.get("positions", []):
        pos = int(float(p.get("position_fp", 0) or 0))
        rpnl = float(p.get("realized_pnl_dollars", 0) or 0) * 100  # dollars -> cents
        fees = float(p.get("fees_paid_dollars", 0) or 0) * 100
        total_rpnl_cents += rpnl
        total_fees_cents += fees
        total_contracts += abs(pos)

    rpnl_style = "bold green" if total_rpnl_cents >= 0 else "bold red"
    grid.add_row("Realized", Text(f"{total_rpnl_cents:+.0f}c ({_fmt_usd(total_rpnl_cents)})", style=rpnl_style))
    grid.add_row("Fees", Text(f"{total_fees_cents:.0f}c ({_fmt_usd(total_fees_cents)})", style=""))
    net = total_rpnl_cents - total_fees_cents
    net_style = "bold green" if net >= 0 else "bold red"
    grid.add_row("Net", Text(f"{net:+.0f}c ({_fmt_usd(net)})", style=net_style))
    grid.add_row("Contracts", Text(f"{total_contracts}", style=""))

    # Unrealized from strategy snapshots (already computed correctly)
    if snapshots is not None and not snapshots.empty:
        total_snap_pnl = 0.0
        for _, s in snapshots.iterrows():
            total_snap_pnl += s.get("total_pnl_cents", 0)
        if total_snap_pnl != 0:
            snap_style = "green" if total_snap_pnl >= 0 else "red"
            grid.add_row("Strat P&L", Text(f"{total_snap_pnl:+.0f}c ({_fmt_usd(total_snap_pnl)})", style=snap_style))

    # Signal stats from logs
    if signals is not None and not signals.empty:
        n = len(signals)
        avg_edge = signals["edge"].abs().mean()
        buy_yes = int((signals["side"] == "buy_yes").sum())
        buy_no = int((signals["side"] == "buy_no").sum())
        grid.add_row("Signals", Text(f"{n} ({buy_yes}Y/{buy_no}N)  avg|edge|={avg_edge:.1f}c"))

    return Panel(grid, title="P&L", border_style="green", width=42, padding=(0, 1))


# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------

def build_dashboard(api_data: dict) -> Group:
    header = _make_header(api_data)

    # Load log data
    signals = _load_recent_signals(200)
    snapshots = _load_latest_snapshots()

    # Build per-signal lookup (latest signal per ticker)
    signal_by_ticker = {}
    if signals is not None and not signals.empty:
        for _, row in signals.iterrows():
            t = row.get("ticker", "")
            if t not in signal_by_ticker:
                signal_by_ticker[t] = row.to_dict()

    # Build per-ticker snapshot lookup
    snap_by_ticker = {}
    if snapshots is not None and not snapshots.empty:
        for _, row in snapshots.iterrows():
            snap_by_ticker[row.get("ticker", "")] = row.to_dict()

    # Build position lookup
    pos_by_ticker = {}
    for p in api_data.get("positions", []):
        pos_by_ticker[p.get("ticker", "")] = p

    # Build resting orders lookup
    orders_by_ticker: dict[str, list] = {}
    for o in api_data.get("orders", []):
        t = o.get("ticker", "")
        orders_by_ticker.setdefault(t, []).append(o)

    # Collect all known tickers (union of markets, positions, signals)
    all_tickers = set(api_data.get("markets", {}).keys())
    all_tickers |= set(pos_by_ticker.keys())
    all_tickers |= set(signal_by_ticker.keys())

    # Filter to our series only
    all_tickers = sorted(
        t for t in all_tickers
        if any(t.startswith(s) for s in SERIES)
    )

    # Build tiles
    tiles = []
    for ticker in all_tickers:
        mkt = api_data.get("markets", {}).get(ticker, {})
        # Skip contracts with no market data and no position
        if not mkt and ticker not in pos_by_ticker and ticker not in snap_by_ticker:
            continue

        tile = _make_contract_tile(
            ticker=ticker,
            market_data=mkt,
            position_data=pos_by_ticker.get(ticker),
            signal_data=signal_by_ticker.get(ticker),
            snapshot_data=snap_by_ticker.get(ticker),
            resting_orders=orders_by_ticker.get(ticker, []),
        )
        tiles.append(tile)

    # P&L summary tile
    pnl_tile = _make_pnl_tile(api_data, signals, snapshots)
    tiles.append(pnl_tile)

    if not tiles:
        tiles = [Panel(Text("No active contracts", style="dim"), border_style="dim")]

    # Arrange tiles in columns (auto-wrap)
    tile_row = Columns(tiles, equal=False, expand=False)

    return Group(header, tile_row)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    ka_cfg = cfg.get("kalshi", {})
    ka_key_id = ka_cfg.get("api_key", "") or os.environ.get("KALSHI_API_KEY", "")
    ka_key_file = ka_cfg.get("private_key_path", "") or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")

    if not ka_key_id or not ka_key_file:
        console.print("[red]Error:[/] KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH must be set")
        console.print("  Set them in .env or config.yaml")
        return

    pk = load_private_key(ka_key_file)
    client = KalshiRestClient(ka_key_id, pk)

    interval = 3.0

    console.print("[cyan]Starting monitor...[/] Press Ctrl+C to stop.\n")

    try:
        api_data = _fetch_api_data(client)
        last_api_fetch = time.monotonic()

        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                now = time.monotonic()
                if now - last_api_fetch >= interval:
                    try:
                        api_data = _fetch_api_data(client)
                    except Exception:
                        pass
                    last_api_fetch = now

                try:
                    live.update(build_dashboard(api_data))
                except Exception as e:
                    live.update(Panel(Text(f"Error: {e}", style="red"), title="Error"))
                time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/]")


if __name__ == "__main__":
    main()
