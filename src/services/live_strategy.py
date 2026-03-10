"""Live strategy service for Kalshi 15-min crypto prediction contracts.

Wires together:
- CoinbaseDataGateway  (spot prices via WebSocket)
- KalshiDataGateway    (Kalshi orderbook via WebSocket)
- KalshiOrderGateway   (order placement via REST)
- Fair value model     (same as backtest)
- ML-derived filters   (spot return, vol, time remaining)
- Risk management      (position limits, loss limits)

Runs the same logic as BacktestEngine._evaluate_signals() but
against live data with real order submission.

Usage:
    python -m services.live_strategy --paper     # paper trading (log only)
    python -m services.live_strategy --live      # real orders
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from gateway.base import BookUpdate, FillUpdate, OrderRequest, OrderUpdate, PositionUpdate, TradeUpdate
from gateway.coinbase.data import CoinbaseDataGateway
from gateway.kalshi import (
    KalshiDataGateway,
    KalshiOrderGateway,
    KalshiRestClient,
    load_private_key,
)
from risk.order_manager import OrderManager
from risk.position_manager import PositionManager
from risk.risk_manager import RiskManager, PositionState
from services.trade_logger import TradeLogger
from strategy.fair_value import (
    ContractInfo,
    VolEstimator,
    compute_fair_value,
    parse_contract_ticker,
)
from strategy.models import MarketState, TheoModel, get_model
from strategy.features import FlowTracker, BookImbalanceTracker, MomentumTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_strategy")


@dataclass
class StrategyConfig:
    edge_threshold_cents: float = 15.0
    fee_per_side_cents: float = 7.0
    max_position_per_contract: int = 5
    cooldown_sec: float = 30.0
    min_time_remaining_sec: float = 120.0
    max_time_remaining_sec: float = 840.0
    vol_lookback_sec: float = 600.0
    max_spot_return_pct: float = 0.05
    max_vol_15m: float = 0.0
    prefer_early_entry: bool = True
    max_loss_cents: float = 5000.0
    max_model_disagreement: float = 30.0  # skip if |FV - market_mid| > this (model probably wrong)
    order_size: int = 1
    take_profit_pct: float = 20.0   # sell when unrealized gain >= X% of entry cost
    stop_loss_pct: float = 35.0    # sell when unrealized loss >= X% of entry cost
    exit_warmup_sec: float = 30.0  # don't check TP/SL within N seconds of fill
    model: str = "gbm"              # "gbm" (analytical) or "ml" (LightGBM)
    series: list[str] | None = None
    assets: list[str] | None = None


class LiveStrategy:
    """Live trading engine for Kalshi 15-min contracts."""

    def __init__(
        self,
        config: StrategyConfig,
        cb_data: CoinbaseDataGateway,
        ka_data: KalshiDataGateway,
        ka_orders: KalshiOrderGateway,
        paper: bool = True,
    ) -> None:
        self.cfg = config
        self.cb_data = cb_data
        self.ka_data = ka_data
        self.ka_orders = ka_orders
        self.paper = paper

        self.assets = config.assets or ["BTC", "ETH", "SOL"]
        self.series = config.series or [f"KX{a}15M" for a in self.assets]

        # FIX #1: Store event loop reference for thread-safe callbacks
        self._loop: asyncio.AbstractEventLoop | None = None

        # Pluggable model
        self._model: TheoModel = get_model(config.model)
        logger.info("Using model: %s", self._model.name)

        # Per-asset state
        self._spot: dict[str, float] = {}
        self._vol: dict[str, VolEstimator] = {
            a: VolEstimator(config.vol_lookback_sec, asset=a) for a in self.assets
        }

        # Feature trackers (feed into MarketState for models that use them)
        self._flow: dict[str, FlowTracker] = {
            a: FlowTracker(lookback_sec=120.0) for a in self.assets
        }
        self._book_imb: dict[str, BookImbalanceTracker] = {
            a: BookImbalanceTracker() for a in self.assets
        }
        self._momentum: dict[str, MomentumTracker] = {
            a: MomentumTracker(lookback_sec=60.0) for a in self.assets
        }

        # Per-contract state
        self._contracts: dict[str, ContractInfo] = {}
        self._spot_at_open: dict[str, float] = {}
        self._positions: dict[str, PositionManager] = {}
        self._order_mgrs: dict[str, OrderManager] = {}
        self._last_trade_ts: dict[str, float] = {}

        # Global risk
        self._risk = RiskManager(
            max_position=config.max_position_per_contract,
            max_loss_cents=config.max_loss_cents,
        )

        # Trade logger
        self._trade_log = TradeLogger(base_dir="logs")

        # Circuit breaker: stop new entries when total P&L hits this
        self._max_total_loss_cents = config.max_loss_cents  # -5000c = -$50
        self._circuit_breaker_tripped = False

        # Safety: pending order tracking to prevent position overshoot
        self._pending_entry_qty: dict[str, int] = {}   # ticker -> pending buy contracts
        self._pending_exits: set[str] = set()           # tickers with outstanding exit orders
        self._pending_exit_ids: dict[str, str] = {}     # ticker -> exit order_id (track specific order)
        self._last_exit_ts: dict[str, float] = {}       # ticker -> last exit attempt time
        self._exit_cooldown_after_fill: dict[str, float] = {}  # ticker -> ts when position went flat
        self._last_entry_fill_ts: dict[str, float] = {}         # ticker -> ts of last buy fill

        # Track contracts we traded on for settlement validation
        # {ticker: {"side": "yes"/"no", "entry_price": float, "fv": float, "size": int}}
        self._traded_contracts: dict[str, dict] = {}

        # Safety: per-ticker evaluation locks to prevent concurrent signal evaluation
        self._eval_locks: dict[str, asyncio.Lock] = {}

        # Stats
        self._signals_generated = 0
        self._orders_sent = 0
        self._orders_rejected_risk = 0
        self._start_time = time.time()

    async def run(self) -> None:
        # FIX #1: Capture event loop for thread-safe callbacks
        self._loop = asyncio.get_running_loop()

        mode = "PAPER" if self.paper else "LIVE"
        logger.info("Starting %s strategy", mode)
        logger.info("Config: edge=%.0fc fee=%.0fc maxpos=%d cool=%.0fs spotret=%.2f%% early=%s",
                     self.cfg.edge_threshold_cents, self.cfg.fee_per_side_cents,
                     self.cfg.max_position_per_contract, self.cfg.cooldown_sec,
                     self.cfg.max_spot_return_pct, self.cfg.prefer_early_entry)
        logger.info("Exit rules: TP=%.0f%% SL=%.0f%% warmup=%.0fs",
                     self.cfg.take_profit_pct, self.cfg.stop_loss_pct,
                     self.cfg.exit_warmup_sec)

        # Register callbacks
        self.cb_data.on_book_update(self._on_coinbase_book)
        self.cb_data.on_trade(self._on_coinbase_trade)
        self.ka_data.on_book_update(self._on_kalshi_book)
        self.ka_data.on_fill(self._on_fill)
        self.ka_data.on_order_update(self._on_order_update)
        self.ka_data.on_position_update(self._on_position_update)

        # Discover and subscribe to Kalshi markets
        tickers = await self.ka_data.discover_markets(self.series)
        logger.info("Discovered %d active Kalshi markets", len(tickers))

        for ticker in tickers:
            self._register_contract(ticker)

        await self.ka_data.subscribe(tickers)

        # Configure Coinbase symbols
        cb_symbols = [f"{a}-USD" for a in self.assets]
        self.cb_data._symbols = cb_symbols

        # Seed initial positions from exchange (one-time REST call at startup)
        await self._seed_positions()

        # Run all feeds concurrently
        self._tasks = [
            asyncio.create_task(self.cb_data.run()),
            asyncio.create_task(self.ka_data.run()),
            asyncio.create_task(self._market_discovery_loop()),
            asyncio.create_task(self._contract_cleanup_loop()),
            asyncio.create_task(self._status_loop()),
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    def _register_contract(self, ticker: str) -> None:
        info = parse_contract_ticker(ticker)
        if info is None:
            return
        if info.asset not in self.assets:
            return

        self._contracts[ticker] = info
        self._last_trade_ts[ticker] = 0

        if ticker not in self._positions:
            self._positions[ticker] = PositionManager(ticker)
        if ticker not in self._order_mgrs:
            self._order_mgrs[ticker] = OrderManager(
                gateway=self.ka_orders,
                symbol=ticker,
                default_size=self.cfg.order_size,
            )

        # FIX #8: Set spot-at-open for contracts whose window already started
        if ticker not in self._spot_at_open:
            asset = info.asset
            if asset in self._spot and time.time() >= info.window_start:
                self._spot_at_open[ticker] = self._spot[asset]
                logger.info("Window already open: %s spot=%.2f (retroactive)",
                            ticker, self._spot[asset])

    async def _market_discovery_loop(self) -> None:
        """Periodically discover new contracts."""
        while True:
            await asyncio.sleep(60)
            try:
                tickers = await self.ka_data.discover_markets(self.series)
                new = [t for t in tickers if t not in self._contracts]
                if new:
                    logger.info("Discovered %d new contracts", len(new))
                    for ticker in new:
                        self._register_contract(ticker)
                    # FIX #13: Only subscribe to new tickers
                    await self.ka_data.subscribe(new)
            except Exception:
                logger.exception("Market discovery error")

    async def _contract_cleanup_loop(self) -> None:
        """Periodically check expired contracts for settlement and clean up."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            expired = [
                (t, info) for t, info in self._contracts.items()
                if info.window_end < now - 60  # 1 min grace period
            ]
            for ticker, info in expired:
                # Check if we had a position and record settlement
                await self._record_settlement(ticker, info)
                self._contracts.pop(ticker, None)
                self._spot_at_open.pop(ticker, None)
                self._last_trade_ts.pop(ticker, None)
            if expired:
                logger.info("Cleaned up %d expired contracts", len(expired))

    async def _seed_positions(self) -> None:
        """One-time REST query to seed positions on startup.

        After this, all position/order state is maintained via WS callbacks.
        """
        if self.paper:
            return
        try:
            positions = await self.ka_orders.get_positions()
            for pi in positions:
                pm = self._positions.get(pi.symbol)
                if pm is not None:
                    pm.initialize(pi.position, 0.0)
                    logger.info("Seeded position: %s pos=%d", pi.symbol, pi.position)
        except Exception:
            logger.exception("Startup position seed failed")

    async def _record_settlement(self, ticker: str, info: ContractInfo) -> None:
        """Query Kalshi for settlement result and log it.

        Records settlement even if we already exited the position (via TP/SL),
        so we can validate model accuracy.
        """
        pm = self._positions.get(ticker)
        tc = self._traded_contracts.get(ticker)

        # Skip if we never traded this contract
        has_position = pm is not None and pm.position != 0
        has_trade_history = tc is not None
        if not has_position and not has_trade_history:
            return

        try:
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None, lambda: self.ka_orders._client.get_market(ticker),
            )
            market = resp.get("market", resp)
            result = market.get("result", "").lower().strip()
            if result not in ("yes", "no"):
                return

            # Use live position data if still holding, otherwise use trade history
            if has_position:
                entry_price = pm.avg_entry_price
                entry_side = pm.entry_side
                entry_size = abs(pm.position)
                fees = pm._total_fees
            else:
                entry_price = tc["entry_price"]
                entry_side = tc["side"]
                entry_size = tc["size"]
                fees = 0  # already accounted for in TP/SL exit

            # Calculate payout (only meaningful if still holding)
            if has_position:
                payout = entry_size * 100 if entry_side == result else 0
                cost = entry_price * entry_size
                pnl = payout - cost - fees
            else:
                # Already exited — P&L was realized at exit, not settlement
                payout = 0
                pnl = 0

            model_correct = (entry_side == result)

            # Get fair value at entry from signal logs
            fv_at_entry = 0.0
            try:
                import pandas as pd
                sig_dir = Path("logs/signals")
                if sig_dir.exists():
                    files = sorted(sig_dir.rglob("*.parquet"))
                    if files:
                        df = pd.read_parquet(files[-1])
                        ticker_sigs = df[df["ticker"] == ticker]
                        if not ticker_sigs.empty:
                            fv_at_entry = ticker_sigs.iloc[-1]["fair_value"]
            except Exception:
                pass

            self._trade_log.log_settlement(
                ts=time.time(),
                ticker=ticker,
                asset=info.asset,
                window_end=info.window_end,
                entry_side=entry_side,
                entry_price=entry_price,
                entry_size=entry_size,
                settlement_result=result,
                payout_cents=payout,
                pnl_cents=pnl,
                fees_cents=fees,
                fair_value_at_entry=fv_at_entry,
                model_correct=model_correct,
            )
            self._trade_log.flush_all()

            # Clean up trade history
            self._traded_contracts.pop(ticker, None)

            status = "HELD" if has_position else "EXITED EARLY"
            logger.info(
                "SETTLED [%s]: %s result=%s model_side=%s correct=%s entry=%.0fc size=%d pnl=%+.0fc",
                status, ticker, result, entry_side, model_correct,
                entry_price, entry_size, pnl,
            )
        except Exception as e:
            logger.error("Settlement check failed for %s: %s", ticker, e)

    # -- Callbacks -----------------------------------------------------------

    def _on_coinbase_book(self, update: BookUpdate) -> None:
        """Handle Coinbase spot price update.

        NOTE: This runs from the Coinbase WS thread via call_soon_threadsafe.
        We must not call async code directly — schedule via run_coroutine_threadsafe.
        """
        asset = update.symbol.split("-")[0]
        self._spot[asset] = update.mid
        self._vol[asset].update(update.ts, update.mid)

        # Update feature trackers
        if asset in self._momentum:
            self._momentum[asset].update(update.ts, update.mid)
        if asset in self._book_imb:
            self._book_imb[asset].update(update.bid_size, update.ask_size)

        # Set window-open price for active contracts
        for ticker, info in self._contracts.items():
            if info.asset == asset and ticker not in self._spot_at_open:
                if update.ts >= info.window_start:
                    self._spot_at_open[ticker] = update.mid
                    logger.info("Window open: %s spot=%.2f", ticker, update.mid)

        # FIX #1: Thread-safe coroutine scheduling
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._evaluate_signals(update.ts, asset),
                self._loop,
            )

    def _on_coinbase_trade(self, update: TradeUpdate) -> None:
        """Handle Coinbase trade event — feeds the FlowTracker."""
        asset = update.symbol.split("-")[0]
        if asset in self._flow:
            self._flow[asset].update(update.ts, update.size, update.side)

    def _on_kalshi_book(self, update: BookUpdate) -> None:
        """Handle Kalshi book update — also triggers evaluation.

        NOTE: Kalshi WS runs in the same asyncio loop (not a separate thread),
        so we can schedule directly.
        """
        info = self._contracts.get(update.symbol)
        if info:
            if self._loop is not None and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._evaluate_signals(time.time(), info.asset),
                    self._loop,
                )

    def _on_fill(self, fill: FillUpdate) -> None:
        """Handle a fill on one of our orders."""
        # Reject fills with invalid price
        if fill.price <= 0:
            logger.error("Ignoring fill with zero/negative price: %s %s",
                         fill.symbol, fill.order_id)
            return

        pm = self._positions.get(fill.symbol)
        om = self._order_mgrs.get(fill.symbol)
        if pm:
            pm.process_fill(fill)

        # Check if this fill is from an exit order (bypasses OrderManager)
        exit_oid = self._pending_exit_ids.get(fill.symbol)
        if exit_oid == fill.order_id:
            matched_side = "exit"
            logger.info("Exit fill matched: %s %s %dc x%.0f",
                        fill.symbol, fill.side, fill.price, fill.size)
        else:
            matched_side = ""
            if om:
                matched_side = om.process_fill(fill) or ""

        # Clear pending tracking on fills
        if fill.action == "buy":
            self._last_entry_fill_ts[fill.symbol] = time.time()
            qty = self._pending_entry_qty.get(fill.symbol, 0)
            self._pending_entry_qty[fill.symbol] = max(0, qty - int(fill.size))
            # Track this contract for settlement validation
            if fill.symbol not in self._traded_contracts:
                self._traded_contracts[fill.symbol] = {
                    "side": fill.side,
                    "entry_price": fill.price,
                    "size": int(fill.size),
                }
            else:
                tc = self._traded_contracts[fill.symbol]
                tc["size"] += int(fill.size)
        elif fill.action == "sell":
            # Only clear pending exit when position is flat (all fills received)
            pm = self._positions.get(fill.symbol)
            if pm and pm.position <= 0:
                self._pending_exits.discard(fill.symbol)
                self._pending_exit_ids.pop(fill.symbol, None)
                # Set a cooldown to prevent immediate re-entry → re-exit churn
                self._exit_cooldown_after_fill[fill.symbol] = time.time()

        logger.info("FILL: %s %s %s %dc x%.0f fee=%.1fc",
                     fill.symbol, fill.action, fill.side,
                     fill.price, fill.size, fill.fee)

        # Log fill to Parquet
        info = self._contracts.get(fill.symbol)
        asset = info.asset if info else fill.symbol.split("-")[0] if "-" in fill.symbol else ""
        self._trade_log.log_fill(
            ts=time.time(), ticker=fill.symbol, asset=asset,
            side=fill.side, action=fill.action,
            price_cents=fill.price, size=fill.size,
            fee_cents=fill.fee, order_id=fill.order_id,
            matched_side=matched_side,
        )

    def _on_order_update(self, update: OrderUpdate) -> None:
        """Handle real-time order status from Kalshi WS.

        Authoritatively clears _active orders on executed/canceled,
        fixing the fill-before-order race condition.
        """
        om = self._order_mgrs.get(update.symbol)
        if om:
            side = om.process_order_update(update)
            if side and update.status == "canceled":
                # If exchange cancelled our order, clear pending entry tracking
                self._pending_entry_qty.pop(update.symbol, None)

        # Clear pending exit tracking if the exit order was cancelled or fully executed
        if update.status in ("canceled", "executed"):
            exit_oid = self._pending_exit_ids.get(update.symbol)
            if exit_oid == update.order_id:
                self._pending_exits.discard(update.symbol)
                self._pending_exit_ids.pop(update.symbol, None)
                if update.status == "canceled":
                    logger.warning("Exit order cancelled by exchange: %s %s",
                                   update.symbol, update.order_id)

    def _on_position_update(self, update: PositionUpdate) -> None:
        """Handle real-time position update from Kalshi WS.

        This is the authoritative source of truth — replaces REST reconciliation.
        """
        pm = self._positions.get(update.symbol)
        if pm is None:
            # Position for a contract we don't track — could be manual trade
            return

        old_pos = pm.position
        pm.process_ws_position(update)

        # Clear stale pending state when position goes flat
        if update.position == 0 and old_pos != 0:
            self._pending_entry_qty.pop(update.symbol, None)
            self._pending_exits.discard(update.symbol)
            self._pending_exit_ids.pop(update.symbol, None)
            self._exit_cooldown_after_fill[update.symbol] = time.time()
            self._last_entry_fill_ts.pop(update.symbol, None)

    # -- Signal evaluation ---------------------------------------------------

    async def _evaluate_signals(self, ts: float, asset: str) -> None:
        """Check exits first, then new entry opportunities."""
        # Per-asset lock prevents concurrent evaluations from stacking orders
        lock = self._eval_locks.setdefault(asset, asyncio.Lock())
        if lock.locked():
            return  # another evaluation is already running for this asset
        async with lock:
            await self._evaluate_signals_locked(ts, asset)

    async def _evaluate_signals_locked(self, ts: float, asset: str) -> None:
        """Actual signal evaluation (called under lock)."""
        # Check take-profit / stop-loss before looking for new entries
        await self._check_exits(ts, asset)

        spot = self._spot.get(asset)
        if spot is None:
            return

        vol_est = self._vol.get(asset)
        if vol_est is None:
            return
        vol = vol_est.vol_15m()

        cfg = self.cfg

        # Circuit breaker: compute total P&L across all contracts
        if self._circuit_breaker_tripped:
            return
        total_pnl = 0.0
        for t, ci in self._contracts.items():
            pm = self._positions.get(t)
            if pm is None:
                continue
            mid_price = None
            book = self.ka_data.get_book(t)
            if book:
                _, _, mid_price = book.top()
            total_pnl += pm.get_pnl(mid_price).get("total_pnl_cents", 0)
        if total_pnl <= -self._max_total_loss_cents:
            if not self._circuit_breaker_tripped:
                logger.warning(
                    "CIRCUIT BREAKER: total P&L %.0fc ($%.2f) hit -$%.0f limit. "
                    "No new entries. Existing positions will be held.",
                    total_pnl, total_pnl / 100, self._max_total_loss_cents / 100,
                )
                self._circuit_breaker_tripped = True
            return

        for ticker, info in list(self._contracts.items()):
            if info.asset != asset:
                continue

            # Skip expired contracts
            time_remaining = info.window_end - ts
            if time_remaining <= 0:
                continue

            spot_open = self._spot_at_open.get(ticker)
            if spot_open is None:
                continue

            if time_remaining < cfg.min_time_remaining_sec:
                continue
            if time_remaining > cfg.max_time_remaining_sec:
                continue

            # Get Kalshi book
            book = self.ka_data.get_book(ticker)
            if book is None:
                continue

            yes_bid, yes_ask, mid = book.top()
            if not yes_bid or not yes_ask:
                continue
            bid, ask = yes_bid[0], yes_ask[0]
            if ask <= bid:
                continue

            # Cooldown
            if ts - self._last_trade_ts.get(ticker, 0) < cfg.cooldown_sec:
                continue

            # Position limit (includes pending unfilled orders)
            pm = self._positions.get(ticker)
            pos = pm.position if pm else 0
            pending = self._pending_entry_qty.get(ticker, 0)
            effective_pos = abs(pos) + pending
            if effective_pos >= cfg.max_position_per_contract:
                continue

            # ML-derived filters
            spot_return = abs(spot / spot_open - 1)  # raw fraction (0.05 = 5%)
            if cfg.max_spot_return_pct > 0 and spot_return > cfg.max_spot_return_pct:
                continue
            if cfg.max_vol_15m > 0 and vol > cfg.max_vol_15m:
                continue
            # FIX #2: prefer_early_entry means ONLY trade when time_remaining >= 540s
            # (skip late entries, keep early ones)
            if cfg.prefer_early_entry and time_remaining < 540:
                continue

            # Build market state and compute fair value via pluggable model
            flow_t = self._flow.get(info.asset)
            book_t = self._book_imb.get(info.asset)
            mom_t = self._momentum.get(info.asset)
            state = MarketState(
                spot=spot,
                spot_at_open=spot_open,
                vol_15m=vol,
                time_remaining_sec=time_remaining,
                asset=info.asset,
                kalshi_bid=bid,
                kalshi_ask=ask,
                flow_imbalance=flow_t.imbalance if flow_t else None,
                book_imbalance=book_t.imbalance if book_t else None,
                momentum_1m=mom_t.momentum if mom_t else None,
            )
            fv = self._model.fair_value(state)
            fv_cents = fv * 100
            kalshi_mid = (bid + ask) / 2
            edge = fv_cents - kalshi_mid
            # Edge threshold is the minimum model edge required to enter.
            # Fees are already accounted for in P&L at settlement — don't
            # double-count them as an additional entry hurdle.
            min_edge = cfg.edge_threshold_cents

            # Skip if model disagrees with market by too much — we're probably wrong
            if cfg.max_model_disagreement > 0 and abs(edge) > cfg.max_model_disagreement:
                continue

            # Log every evaluation so the monitor always has data
            spot_return_pct = abs(spot / spot_open - 1) * 100 if spot_open else 0
            eval_side = "buy_yes" if edge > 0 else "buy_no"
            self._trade_log.log_signal(
                ts=ts, ticker=ticker, asset=info.asset, side=eval_side,
                price_cents=ask if edge > 0 else 100 - bid,
                fair_value=fv_cents, edge=edge,
                spot=spot, spot_open=spot_open, spot_return_pct=spot_return_pct,
                vol_15m=vol, time_remaining=time_remaining,
                kalshi_bid=bid, kalshi_ask=ask,
                position=pos, paper=self.paper,
            )

            if edge > min_edge and effective_pos < cfg.max_position_per_contract:
                await self._send_signal(
                    ts, ticker, info, "buy_yes", ask, fv_cents, edge,
                    spot, spot_open, time_remaining, vol, bid, ask,
                )
            elif edge < -min_edge and effective_pos < cfg.max_position_per_contract:
                await self._send_signal(
                    ts, ticker, info, "buy_no", 100 - bid, fv_cents, edge,
                    spot, spot_open, time_remaining, vol, bid, ask,
                )

    async def _send_signal(
        self, ts: float, ticker: str, info: ContractInfo,
        side: str, price_cents: float, fv_cents: float, edge: float,
        spot: float, spot_open: float, time_remaining: float,
        vol_15m: float, kalshi_bid: float, kalshi_ask: float,
    ) -> None:
        self._signals_generated += 1
        self._last_trade_ts[ticker] = ts

        t_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")
        logger.info(
            "SIGNAL: %s %s %s @ %dc  FV=%.1fc  edge=%+.1fc  spot=%.2f  tr=%.0fs",
            t_str, ticker, side, price_cents, fv_cents, edge, spot, time_remaining,
        )

        pm = self._positions.get(ticker)
        pos = pm.position if pm else 0
        spot_return_pct = abs(spot / spot_open - 1) * 100 if spot_open else 0

        # Log signal to Parquet
        self._trade_log.log_signal(
            ts=ts, ticker=ticker, asset=info.asset, side=side,
            price_cents=price_cents, fair_value=fv_cents, edge=edge,
            spot=spot, spot_open=spot_open, spot_return_pct=spot_return_pct,
            vol_15m=vol_15m, time_remaining=time_remaining,
            kalshi_bid=kalshi_bid, kalshi_ask=kalshi_ask,
            position=pos, paper=self.paper,
        )

        if self.paper:
            logger.info("PAPER: would %s %s @ %dc", side, ticker, price_cents)
            # Log paper order too
            self._trade_log.log_order(
                ts=ts, ticker=ticker, asset=info.asset,
                side="bid" if side == "buy_yes" else "ask",
                price_cents=price_cents, size=self.cfg.order_size,
                order_id="paper", success=True, paper=True,
            )
            return

        # Place real order
        om = self._order_mgrs.get(ticker)
        if om is None:
            return

        # Track pending entry to prevent position overshoot
        order_size = self.cfg.order_size
        self._pending_entry_qty[ticker] = self._pending_entry_qty.get(ticker, 0) + order_size

        order_side = "bid" if side == "buy_yes" else "ask"
        order = await om.place_order(order_side, price_cents, order_size)

        if order:
            self._orders_sent += 1
            logger.info("ORDER: %s %s %dc id=%s", side, ticker, price_cents, order.order_id)
            self._trade_log.log_order(
                ts=ts, ticker=ticker, asset=info.asset,
                side=order_side, price_cents=price_cents,
                size=order_size, order_id=order.order_id,
                success=True, paper=False,
            )
        else:
            # Order failed — release the pending reservation
            self._pending_entry_qty[ticker] = max(0, self._pending_entry_qty.get(ticker, 0) - order_size)
            self._orders_rejected_risk += 1
            logger.warning("ORDER FAILED: %s %s %dc", side, ticker, price_cents)
            self._trade_log.log_order(
                ts=ts, ticker=ticker, asset=info.asset,
                side=order_side, price_cents=price_cents,
                size=order_size, order_id="",
                success=False, error="placement_failed", paper=False,
            )

    # -- Position exit logic ---------------------------------------------------

    async def _check_exits(self, ts: float, asset: str) -> None:
        """Check open positions for take-profit or stop-loss exits."""
        cfg = self.cfg
        if cfg.take_profit_pct <= 0 and cfg.stop_loss_pct <= 0:
            return

        for ticker, info in list(self._contracts.items()):
            if info.asset != asset:
                continue

            # Skip if an exit order is already pending for this ticker
            if ticker in self._pending_exits:
                continue

            # Cooldown: don't retry exits more than once per 5 seconds
            if ts - self._last_exit_ts.get(ticker, 0) < 5.0:
                continue

            pm = self._positions.get(ticker)
            if pm is None or pm.position == 0 or pm.avg_entry_price <= 0:
                continue

            # Warmup: don't check TP/SL within N seconds of entry fill
            last_entry = self._last_entry_fill_ts.get(ticker, 0)
            if ts - last_entry < cfg.exit_warmup_sec:
                continue

            # Get current market price
            book = self.ka_data.get_book(ticker)
            if book is None:
                continue
            yes_bid, yes_ask, mid = book.top()
            if not yes_bid or not yes_ask:
                continue

            entry = pm.avg_entry_price
            pos = pm.position
            entry_side = pm.entry_side  # "yes" or "no"

            # Use mid price for valuation to avoid spread-induced false triggers
            if entry_side == "yes":
                current_val = (yes_bid[0] + yes_ask[0]) / 2
            else:
                current_val = 100 - (yes_bid[0] + yes_ask[0]) / 2

            pnl_per_contract = current_val - entry
            pnl_pct = (pnl_per_contract / entry) * 100 if entry > 0 else 0

            # Check take-profit
            if cfg.take_profit_pct > 0 and pnl_pct >= cfg.take_profit_pct:
                logger.info(
                    "TAKE PROFIT: %s %s pos=%d entry=%.0fc current=%.0fc pnl=%+.1f%%",
                    ticker, entry_side, pos, entry, current_val, pnl_pct,
                )
                await self._exit_position(ts, ticker, info, pm, "take_profit")
                continue

            # Check stop-loss
            if cfg.stop_loss_pct > 0 and pnl_pct <= -cfg.stop_loss_pct:
                logger.info(
                    "STOP LOSS: %s %s pos=%d entry=%.0fc current=%.0fc pnl=%+.1f%%",
                    ticker, entry_side, pos, entry, current_val, pnl_pct,
                )
                await self._exit_position(ts, ticker, info, pm, "stop_loss")
                continue

    async def _exit_position(
        self, ts: float, ticker: str, info: ContractInfo,
        pm: PositionManager, reason: str,
    ) -> None:
        """Sell entire position at market (hit the bid)."""
        pos = abs(pm.position)
        entry_side = pm.entry_side  # "yes" or "no"

        # Mark as pending and record timestamp
        self._pending_exits.add(ticker)
        self._last_exit_ts[ticker] = ts

        if self.paper:
            book = self.ka_data.get_book(ticker)
            if book:
                yes_bid, yes_ask, _ = book.top()
                if entry_side == "yes" and yes_bid:
                    sell_price = yes_bid[0]
                elif entry_side == "no" and yes_ask:
                    sell_price = 100 - yes_ask[0]
                else:
                    sell_price = 0
            else:
                sell_price = 0
            logger.info(
                "PAPER EXIT (%s): would sell %d %s %s @ %dc",
                reason, pos, entry_side, ticker, sell_price,
            )
            self._trade_log.log_order(
                ts=ts, ticker=ticker, asset=info.asset,
                side=f"sell_{entry_side}",
                price_cents=sell_price, size=pos,
                order_id="paper", success=True, paper=True,
            )
            # Simulate the fill: clear position and pending state
            pm._position = 0
            pm._avg_entry_price = 0.0
            pm._entry_side = ""
            if sell_price > 0:
                pm._cash_cents += sell_price * pos
            self._pending_exits.discard(ticker)
            self._pending_exit_ids.pop(ticker, None)
            self._last_entry_fill_ts.pop(ticker, None)
            return

        # Place sell order via gateway directly (OrderManager only handles buys)
        book = self.ka_data.get_book(ticker)
        if not book:
            return
        yes_bid, yes_ask, _ = book.top()

        if entry_side == "yes" and yes_bid:
            # Sell YES at yes_bid
            sell_price = yes_bid[0]
            req = OrderRequest(
                symbol=ticker, side="yes", action="sell",
                size=pos, order_type="limit", price=sell_price,
            )
        elif entry_side == "no" and yes_ask:
            # Sell NO at no_bid (= 100 - yes_ask)
            sell_price = 100 - yes_ask[0]
            req = OrderRequest(
                symbol=ticker, side="no", action="sell",
                size=pos, order_type="limit", price=sell_price,
            )
        else:
            logger.warning("EXIT: no valid price to sell %s", ticker)
            return

        resp = await self.ka_orders.submit_order(req)
        if resp.success:
            self._pending_exit_ids[ticker] = resp.order_id
            logger.info(
                "EXIT ORDER (%s): sell %d %s %s @ %dc id=%s",
                reason, pos, entry_side, ticker, sell_price, resp.order_id,
            )
            self._trade_log.log_order(
                ts=ts, ticker=ticker, asset=info.asset,
                side=f"sell_{entry_side}",
                price_cents=sell_price, size=pos,
                order_id=resp.order_id, success=True, paper=False,
            )
        else:
            # Exit order failed — allow retry after cooldown
            self._pending_exits.discard(ticker)
            logger.error("EXIT FAILED (%s): %s %s - %s", reason, ticker, entry_side, resp.error)

    # -- Status & shutdown ---------------------------------------------------

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            uptime = time.time() - self._start_time
            mode = "PAPER" if self.paper else "LIVE"
            total_pos = sum(pm.position for pm in self._positions.values())

            # FIX #11: Pass actual mid prices for mark-to-market
            total_pnl = 0.0
            for ticker, pm in self._positions.items():
                info = self._contracts.get(ticker)
                mid = None
                if info:
                    book = self.ka_data.get_book(ticker)
                    if book:
                        _, _, mid = book.top()
                total_pnl += pm.get_pnl(mid)["total_pnl_cents"]

            active = sum(
                1 for t, info in self._contracts.items()
                if info.window_end > time.time()
            )

            # Log per-contract snapshots
            now = time.time()
            for ticker, pm in self._positions.items():
                info = self._contracts.get(ticker)
                if info is None:
                    continue
                # Get per-ticker mid for accurate P&L
                ka_mid = 0.0
                ticker_mid = None
                book = self.ka_data.get_book(ticker)
                if book:
                    _, _, ka_mid_val = book.top()
                    ka_mid = ka_mid_val or 0.0
                    ticker_mid = ka_mid_val
                pnl = pm.get_pnl(ticker_mid)
                self._trade_log.log_snapshot(
                    ts=now, ticker=ticker, asset=info.asset,
                    position=pm.position,
                    cash_cents=pnl.get("cash_cents", 0),
                    mtm_cents=pnl.get("mark_to_market", 0),
                    total_pnl_cents=pnl.get("total_pnl_cents", 0),
                    fill_count=pnl.get("fill_count", 0),
                    fees_paid=pnl.get("total_fees_paid", 0),
                    kalshi_mid=ka_mid,
                    spot=self._spot.get(info.asset, 0),
                    signals_total=self._signals_generated,
                    orders_total=self._orders_sent,
                    orders_rejected=self._orders_rejected_risk,
                )

            # Flush trade logs every status cycle to avoid data loss
            self._trade_log.flush_all()

            cb_tag = " CB!" if self._circuit_breaker_tripped else ""
            spots = " ".join(f"{a}=${p:.0f}" for a, p in self._spot.items())
            print(
                f"\n[{mode}{cb_tag}] {uptime/60:.0f}m | "
                f"{active}mkts {self._signals_generated}sig {self._orders_sent}ord | "
                f"pos={total_pos} pnl={total_pnl:+.0f}c (${total_pnl/100:+.2f}) | "
                f"{spots}",
                flush=True,
            )

            # Compact per-contract line: ticker FV bid/ask edge pos
            now = time.time()
            lines = []
            for ticker, info in list(self._contracts.items()):
                tr = info.window_end - now
                if tr <= 0:
                    continue
                spot = self._spot.get(info.asset, 0)
                spot_open = self._spot_at_open.get(ticker, 0)
                vol_est = self._vol.get(info.asset)
                vol = vol_est.vol_15m() if vol_est else 0
                book = self.ka_data.get_book(ticker)
                if book:
                    yb, ya, _ = book.top()
                    kb = yb[0] if yb else 0
                    ka = ya[0] if ya else 0
                else:
                    kb, ka = 0, 0
                fv = 0.0
                edge = 0.0
                if spot_open and vol > 0 and tr > 0:
                    flow_t = self._flow.get(info.asset)
                    book_t = self._book_imb.get(info.asset)
                    mom_t = self._momentum.get(info.asset)
                    st = MarketState(
                        spot=spot, spot_at_open=spot_open, vol_15m=vol,
                        time_remaining_sec=tr, asset=info.asset,
                        kalshi_bid=kb if kb else None,
                        kalshi_ask=ka if ka else None,
                        flow_imbalance=flow_t.imbalance if flow_t else None,
                        book_imbalance=book_t.imbalance if book_t else None,
                        momentum_1m=mom_t.momentum if mom_t else None,
                    )
                    fv = self._model.fair_value(st) * 100
                    edge = fv - (kb + ka) / 2 if kb and ka else 0
                pm = self._positions.get(ticker)
                pos = pm.position if pm else 0
                # Short ticker
                short = ticker
                for s in self.series:
                    if ticker.startswith(s + "-"):
                        short = ticker[len(s) + 1:]
                pos_tag = f" pos={pos}" if pos else ""
                lines.append(
                    f"  {short} {tr/60:.0f}m FV={fv:.0f} {kb}/{ka} e={edge:+.1f}{pos_tag}"
                )
            if lines:
                print(" | ".join(lines), flush=True)

    async def _shutdown(self) -> None:
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True
        logger.info("Shutting down strategy...")

        # Cancel all active orders first (needs live gateway)
        if not self.paper:
            for om in self._order_mgrs.values():
                try:
                    await asyncio.wait_for(om.cancel_all(), timeout=3.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.error("Error cancelling orders: %s", e)

        # Disconnect gateways BEFORE cancelling tasks — this closes the
        # WS connections cleanly so the tasks exit on their own
        try:
            await asyncio.wait_for(self.ka_data.disconnect(), timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            pass
        try:
            await asyncio.wait_for(self.cb_data.disconnect(), timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            pass

        # Now cancel any remaining tasks
        tasks = getattr(self, "_tasks", [])
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=3.0,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Flush trade logs
        self._trade_log.flush_all()
        logger.info("Trade logs flushed to disk.")

        # Final stats
        logger.info("Final stats: signals=%d orders=%d rejected=%d",
                     self._signals_generated, self._orders_sent,
                     self._orders_rejected_risk)


def _load_config(path: str = "config.yaml") -> dict:
    """Load config from YAML file, falling back to defaults."""
    import yaml
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", config_path)
        return cfg
    logger.warning("No config.yaml found, using defaults")
    return {}


def main() -> None:
    cfg = _load_config()

    strat = cfg.get("strategy", {})
    filters = cfg.get("filters", {})
    risk = cfg.get("risk", {})
    cb_cfg = cfg.get("coinbase", {})
    ka_cfg = cfg.get("kalshi", {})

    paper = cfg.get("mode", "paper") != "live"

    # Coinbase data gateway
    cb_data = CoinbaseDataGateway(key_file=cb_cfg.get("key_file", "cdp_api_key.json"))

    # Kalshi gateways
    ka_key_id = ka_cfg.get("api_key", "") or os.environ.get("KALSHI_API_KEY", "")
    ka_key_path = ka_cfg.get("private_key_path", "") or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
    ka_private_key = load_private_key(ka_key_path) if ka_key_path else None

    ka_client = KalshiRestClient(ka_key_id, ka_private_key)
    ka_data = KalshiDataGateway(ka_key_id, ka_private_key, client=ka_client)
    ka_orders = KalshiOrderGateway(ka_client)

    config = StrategyConfig(
        edge_threshold_cents=strat.get("edge_threshold_cents", 15.0),
        fee_per_side_cents=strat.get("fee_per_side_cents", 7.0),
        max_position_per_contract=strat.get("max_position_per_contract", 5),
        cooldown_sec=strat.get("cooldown_sec", 30.0),
        min_time_remaining_sec=filters.get("min_time_remaining_sec", 120.0),
        max_time_remaining_sec=filters.get("max_time_remaining_sec", 840.0),
        max_spot_return_pct=filters.get("max_spot_return_pct", 0.05),
        max_vol_15m=filters.get("max_vol_15m", 0.0),
        prefer_early_entry=strat.get("prefer_early_entry", True),
        max_loss_cents=risk.get("max_loss_cents", 5000.0),
        max_model_disagreement=strat.get("max_model_disagreement", 30.0),
        take_profit_pct=risk.get("take_profit_pct", 20.0),
        stop_loss_pct=risk.get("stop_loss_pct", 35.0),
        exit_warmup_sec=risk.get("exit_warmup_sec", 30.0),
        model=strat.get("model", "gbm"),
        assets=cfg.get("assets", ["BTC", "ETH", "SOL"]),
        order_size=strat.get("order_size", 1),
    )

    strategy = LiveStrategy(
        config=config,
        cb_data=cb_data,
        ka_data=ka_data,
        ka_orders=ka_orders,
        paper=paper,
    )

    # Single-instance lock — prevent multiple strategies from running
    lock_path = Path("logs/live_strategy.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = None
    try:
        lock_file = open(lock_path, "w")
        try:
            import msvcrt
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        except (ImportError, OSError):
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (ImportError, OSError):
                logger.error("Another strategy instance is already running! Exiting.")
                lock_file.close()
                sys.exit(1)
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        logger.info("Acquired strategy lock (PID %d)", os.getpid())
    except OSError:
        logger.error("Another strategy instance is already running! Exiting.")
        if lock_file:
            lock_file.close()
        sys.exit(1)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(strategy.run())
    except KeyboardInterrupt:
        logger.info("Ctrl+C received, shutting down gracefully...")
        try:
            loop.run_until_complete(strategy._shutdown())
        except KeyboardInterrupt:
            logger.warning("Second Ctrl+C, forcing exit...")
    finally:
        # Release lock
        if lock_file:
            lock_file.close()
            try:
                lock_path.unlink()
            except OSError:
                pass
        logger.info("Strategy stopped.")
        # Force exit — Windows ProactorEventLoop hangs on close() with
        # pending IOCP handles, and Coinbase WSClient spawns non-daemon
        # threads that prevent clean process exit.
        os._exit(0)


if __name__ == "__main__":
    main()
