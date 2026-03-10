"""Tests for LiveStrategy — the main trading engine with all safety mechanisms.

Covers bug regressions, safety guards, signal evaluation, exit logic,
fill handling, and position reconciliation.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from conftest import make_fill
from gateway.base import (
    BookUpdate,
    FillUpdate,
    OrderGateway,
    OrderRequest,
    OrderResponse,
    OrderUpdate,
    PositionInfo,
    PositionUpdate,
)
from risk.order_manager import OrderManager
from risk.position_manager import PositionManager
from strategy.fair_value import ContractInfo
from services.live_strategy import LiveStrategy, StrategyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A contract ticker in the far future so time_remaining is comfortable
TICKER = "KXBTC15M-26MAR080345-45"
ASSET = "BTC"

def _make_contract_info(
    ticker: str = TICKER,
    asset: str = ASSET,
    window_start: float | None = None,
    window_end: float | None = None,
) -> ContractInfo:
    now = time.time()
    return ContractInfo(
        ticker=ticker,
        asset=asset,
        cb_symbol=f"{asset}-USD",
        series=f"KX{asset}15M",
        window_start=window_start or (now - 300),
        window_end=window_end or (now + 600),  # 10 min from now
    )


def _mock_book(yes_bid=50, yes_ask=55, mid=52.5):
    """Return a mock book whose .top() returns the given values.

    top() returns (yes_bid_tuple, yes_ask_tuple, mid) matching
    the KalshiOrderBook.top() signature.
    """
    book = MagicMock()
    bid_tuple = (yes_bid, 10) if yes_bid is not None else None
    ask_tuple = (yes_ask, 10) if yes_ask is not None else None
    book.top.return_value = (bid_tuple, ask_tuple, mid)
    return book


def _build_strategy(
    paper: bool = False,
    max_pos: int = 5,
    edge: float = 5.0,
    fee: float = 2.0,
    cooldown: float = 0.0,
    take_profit_pct: float = 20.0,
    stop_loss_pct: float = 35.0,
    max_loss: float = 5000.0,
) -> tuple[LiveStrategy, AsyncMock, AsyncMock, AsyncMock]:
    """Build a LiveStrategy with mocked gateways, ready for testing."""
    cfg = StrategyConfig(
        edge_threshold_cents=edge,
        fee_per_side_cents=fee,
        max_position_per_contract=max_pos,
        cooldown_sec=cooldown,
        max_loss_cents=max_loss,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        order_size=1,
        assets=[ASSET],
        max_spot_return_pct=0.0,   # disable spot return filter
        max_vol_15m=0.0,           # disable vol filter
        prefer_early_entry=False,  # disable early entry filter
        max_model_disagreement=0.0,  # disable disagreement filter in tests
    )

    cb_data = MagicMock()
    ka_data = MagicMock()
    ka_orders = AsyncMock(spec=OrderGateway)
    type(ka_orders).exchange_name = PropertyMock(return_value="kalshi")
    ka_orders.submit_order.return_value = OrderResponse(
        success=True, order_id="ord-001"
    )

    strategy = LiveStrategy(
        config=cfg,
        cb_data=cb_data,
        ka_data=ka_data,
        ka_orders=ka_orders,
        paper=paper,
    )

    return strategy, cb_data, ka_data, ka_orders


def _setup_strategy_state(
    strategy: LiveStrategy,
    ka_data: MagicMock,
    ka_orders: AsyncMock,
    loop: asyncio.AbstractEventLoop,
    ticker: str = TICKER,
    position: int = 0,
    avg_entry: float = 0.0,
    entry_side: str = "",
    yes_bid: float = 50,
    yes_ask: float = 55,
    mid: float = 52.5,
) -> None:
    """Populate internal state so the strategy is ready for evaluation."""
    strategy._loop = loop

    info = _make_contract_info(ticker)
    strategy._contracts[ticker] = info
    strategy._spot[ASSET] = 90000.0
    strategy._spot_at_open[ticker] = 90000.0
    strategy._last_trade_ts[ticker] = 0

    # Position manager
    pm = PositionManager(ticker)
    pm._position = position
    pm._avg_entry_price = avg_entry
    pm._entry_side = entry_side
    strategy._positions[ticker] = pm

    # Order manager with mocked gateway
    om = OrderManager(gateway=ka_orders, symbol=ticker, default_size=1)
    strategy._order_mgrs[ticker] = om

    # Book mock
    book = _mock_book(yes_bid, yes_ask, mid)
    ka_data.get_book.return_value = book

    # Patch the trade logger to avoid file I/O
    strategy._trade_log = MagicMock()


# ===========================================================================
# Bug regression tests
# ===========================================================================


class TestPendingEntryPreventsOvershoot:
    """test_pending_entry_prevents_position_overshoot"""

    async def test_pending_entry_prevents_position_overshoot(self):
        """Place order (pending=1), check effective_pos blocks next entry at limit."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy(max_pos=2)
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop, position=1)

        # Simulate one pending entry already reserved
        strategy._pending_entry_qty[TICKER] = 1

        # effective_pos = abs(1) + 1 = 2 >= max_pos(2) => blocked
        # Use a high FV so edge is large enough
        strategy._model.fair_value = MagicMock(return_value=0.90)
        await strategy._evaluate_signals(time.time(), ASSET)

        ka_orders.submit_order.assert_not_called()


class TestPendingEntryReleasedOnFill:
    """test_pending_entry_released_on_fill"""

    async def test_pending_entry_released_on_fill(self):
        """Buy fill decrements _pending_entry_qty."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop)

        strategy._pending_entry_qty[TICKER] = 2

        fill = make_fill(
            symbol=TICKER, side="yes", action="buy",
            price=50.0, size=1.0, order_id="ord-001",
        )
        strategy._on_fill(fill)

        assert strategy._pending_entry_qty[TICKER] == 1


class TestPendingEntryReleasedOnOrderFailure:
    """test_pending_entry_released_on_order_failure"""

    async def test_pending_entry_released_on_order_failure(self):
        """Failed order releases reservation."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop)

        # Make order fail
        ka_orders.submit_order.return_value = OrderResponse(
            success=False, error="insufficient_balance"
        )

        info = strategy._contracts[TICKER]
        await strategy._send_signal(
            ts=time.time(), ticker=TICKER, info=info,
            side="buy_yes", price_cents=55, fv_cents=70, edge=15,
            spot=90000, spot_open=90000, time_remaining=600,
            vol_15m=0.001, kalshi_bid=50, kalshi_ask=55,
        )

        # Pending should be released back to 0
        assert strategy._pending_entry_qty.get(TICKER, 0) == 0


class TestExitDeduplication:
    """test_exit_deduplication_pending_set"""

    async def test_exit_deduplication_pending_set(self):
        """Exit adds to _pending_exits, next _check_exits skips."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(
            strategy, ka_data, ka_orders, loop,
            position=5, avg_entry=40, entry_side="yes",
            yes_bid=60, yes_ask=65, mid=62.5,  # big profit
        )
        # Past warmup
        strategy._last_entry_fill_ts[TICKER] = time.time() - 60

        # First call should trigger exit and add to pending
        await strategy._check_exits(time.time(), ASSET)
        assert TICKER in strategy._pending_exits

        # Reset the mock to track new calls
        ka_orders.submit_order.reset_mock()

        # Second call should skip because ticker is in _pending_exits
        await strategy._check_exits(time.time() + 10, ASSET)
        ka_orders.submit_order.assert_not_called()


class TestExitCooldown:
    """test_exit_cooldown"""

    async def test_exit_cooldown(self):
        """_last_exit_ts prevents retry within 5s."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(
            strategy, ka_data, ka_orders, loop,
            position=5, avg_entry=40, entry_side="yes",
            yes_bid=60, yes_ask=65, mid=62.5,
        )

        now = time.time()
        # Past warmup
        strategy._last_entry_fill_ts[TICKER] = now - 60
        # Set last exit timestamp to recent past (within 5s)
        strategy._last_exit_ts[TICKER] = now - 2.0

        await strategy._check_exits(now, ASSET)

        # Should be blocked by cooldown
        ka_orders.submit_order.assert_not_called()


class TestExitPendingClearedOnSellFill:
    """test_exit_pending_cleared_on_sell_fill"""

    async def test_exit_pending_cleared_on_sell_fill(self):
        """Sell fill removes from _pending_exits."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop, position=5)

        strategy._pending_exits.add(TICKER)

        fill = make_fill(
            symbol=TICKER, side="yes", action="sell",
            price=60.0, size=5.0, order_id="exit-001",
        )
        strategy._on_fill(fill)

        assert TICKER not in strategy._pending_exits


class TestDoubleShutdownGuard:
    """test_double_shutdown_guard"""

    async def test_double_shutdown_guard(self):
        """Second _shutdown() call is a no-op."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        strategy._loop = loop
        strategy._trade_log = MagicMock()

        # First shutdown
        await strategy._shutdown()
        assert strategy._shutdown_called is True

        # Track calls after first shutdown
        cb_data.disconnect = AsyncMock()
        ka_data.disconnect = AsyncMock()
        strategy._trade_log.reset_mock()

        # Second shutdown should be no-op
        await strategy._shutdown()
        cb_data.disconnect.assert_not_called()
        ka_data.disconnect.assert_not_called()


# ===========================================================================
# Safety mechanism tests
# ===========================================================================


class TestEvalLockPreventsConcurrent:
    """test_eval_lock_prevents_concurrent_evaluation"""

    async def test_eval_lock_prevents_concurrent_evaluation(self):
        """Two concurrent _evaluate_signals for same asset: second skips."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop)

        call_count = 0
        original_locked = strategy._evaluate_signals_locked

        async def slow_evaluate(ts, asset):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)

        strategy._evaluate_signals_locked = slow_evaluate

        # Launch two concurrent evaluations
        t1 = asyncio.create_task(strategy._evaluate_signals(time.time(), ASSET))
        # Small delay so the first acquires the lock first
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(strategy._evaluate_signals(time.time(), ASSET))

        await asyncio.gather(t1, t2)

        # Only one should have run because the second sees lock.locked() == True
        assert call_count == 1


class TestCircuitBreakerTrips:
    """test_circuit_breaker_trips_on_loss"""

    async def test_circuit_breaker_trips_on_loss(self):
        """Total P&L < -max_loss => _circuit_breaker_tripped = True, no entries."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy(max_loss=100.0)
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop)

        # Set up position with massive loss
        pm = strategy._positions[TICKER]
        pm._cash_cents = -200.0  # realized loss
        pm._position = 0         # no position, so mtm = 0, total = -200

        strategy._model.fair_value = MagicMock(return_value=0.90)
        await strategy._evaluate_signals(time.time(), ASSET)

        assert strategy._circuit_breaker_tripped is True
        ka_orders.submit_order.assert_not_called()


class TestCircuitBreakerPersists:
    """test_circuit_breaker_persists"""

    async def test_circuit_breaker_persists(self):
        """Once tripped, stays tripped even if P&L recovers."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy(max_loss=100.0)
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop)

        # Trip the breaker
        strategy._circuit_breaker_tripped = True

        # Even with positive P&L, breaker stays tripped
        pm = strategy._positions[TICKER]
        pm._cash_cents = 500.0

        strategy._model.fair_value = MagicMock(return_value=0.90)
        await strategy._evaluate_signals(time.time(), ASSET)

        assert strategy._circuit_breaker_tripped is True
        ka_orders.submit_order.assert_not_called()


# ===========================================================================
# Signal evaluation tests
# ===========================================================================


class TestEntrySignalBuyYes:
    """test_entry_signal_buy_yes"""

    async def test_entry_signal_buy_yes(self):
        """edge > min_edge => order placed."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy(
            edge=5.0, fee=2.0,
        )
        loop = asyncio.get_running_loop()
        _setup_strategy_state(
            strategy, ka_data, ka_orders, loop,
            yes_bid=40, yes_ask=45, mid=42.5,
        )

        # FV = 0.60 => 60c.  kalshi_mid = 42.5c.  edge = 60 - 42.5 = 17.5c
        # min_edge = 5 + 2 = 7c.  17.5 > 7 => buy_yes at ask=45c
        strategy._model.fair_value = MagicMock(return_value=0.60)
        await strategy._evaluate_signals(time.time(), ASSET)

        ka_orders.submit_order.assert_called_once()


class TestNoSignalInsufficientEdge:
    """test_no_signal_insufficient_edge"""

    async def test_no_signal_insufficient_edge(self):
        """abs(edge) < min_edge => no order."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy(
            edge=15.0, fee=7.0,
        )
        loop = asyncio.get_running_loop()
        _setup_strategy_state(
            strategy, ka_data, ka_orders, loop,
            yes_bid=50, yes_ask=55, mid=52.5,
        )

        # FV = 0.53 => 53c.  kalshi_mid = 52.5c.  edge = 0.5c
        # min_edge = 15 + 7 = 22c.  0.5 < 22 => no order
        strategy._model.fair_value = MagicMock(return_value=0.53)
        await strategy._evaluate_signals(time.time(), ASSET)

        ka_orders.submit_order.assert_not_called()


class TestCooldownBlocksSignal:
    """test_cooldown_blocks_signal"""

    async def test_cooldown_blocks_signal(self):
        """Within cooldown => no order."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy(
            cooldown=60.0,
        )
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop)

        # Set last trade to recent past
        strategy._last_trade_ts[TICKER] = time.time() - 10  # 10s ago, cooldown=60s

        strategy._model.fair_value = MagicMock(return_value=0.90)
        await strategy._evaluate_signals(time.time(), ASSET)

        ka_orders.submit_order.assert_not_called()


class TestTimeRemainingFilters:
    """test_time_remaining_filters"""

    async def test_too_little_time_remaining(self):
        """too low time remaining => skipped."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop)

        # Set window_end so time_remaining < min_time_remaining_sec (120s)
        now = time.time()
        strategy._contracts[TICKER] = _make_contract_info(
            window_end=now + 60,  # only 60s remaining
        )

        strategy._model.fair_value = MagicMock(return_value=0.90)
        await strategy._evaluate_signals(now, ASSET)

        ka_orders.submit_order.assert_not_called()

    async def test_too_much_time_remaining(self):
        """too high time remaining => skipped."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop)

        # Set window_end so time_remaining > max_time_remaining_sec (840s)
        now = time.time()
        strategy._contracts[TICKER] = _make_contract_info(
            window_end=now + 1000,  # 1000s remaining
        )

        strategy._model.fair_value = MagicMock(return_value=0.90)
        await strategy._evaluate_signals(now, ASSET)

        ka_orders.submit_order.assert_not_called()


class TestPaperModeNoRealOrders:
    """test_paper_mode_no_real_orders"""

    async def test_paper_mode_no_real_orders(self):
        """Paper mode logs but no gateway calls."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy(paper=True)
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop)

        strategy._model.fair_value = MagicMock(return_value=0.90)
        await strategy._evaluate_signals(time.time(), ASSET)

        # No real orders submitted
        ka_orders.submit_order.assert_not_called()
        # But trade_log should have been called (paper order logged)
        strategy._trade_log.log_order.assert_called()


# ===========================================================================
# Exit logic tests
# ===========================================================================


class TestTakeProfitTriggersExit:
    """test_take_profit_triggers_exit"""

    async def test_take_profit_triggers_exit(self):
        """pnl_pct >= TP% => exit (after warmup)."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy(
            take_profit_pct=20.0,
        )
        loop = asyncio.get_running_loop()
        # Entry at 40c, mid=(55+60)/2=57.5c => pnl_pct = (57.5-40)/40*100 = 43.75%
        _setup_strategy_state(
            strategy, ka_data, ka_orders, loop,
            position=3, avg_entry=40, entry_side="yes",
            yes_bid=55, yes_ask=60, mid=57.5,
        )
        # Set entry fill time in the past (past warmup)
        strategy._last_entry_fill_ts[TICKER] = time.time() - 60

        await strategy._check_exits(time.time(), ASSET)

        # Should have submitted a sell order
        ka_orders.submit_order.assert_called_once()
        req = ka_orders.submit_order.call_args[0][0]
        assert req.action == "sell"
        assert req.side == "yes"
        assert req.size == 3

    async def test_take_profit_blocked_during_warmup(self):
        """TP is not checked within warmup period."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy(
            take_profit_pct=20.0,
        )
        loop = asyncio.get_running_loop()
        _setup_strategy_state(
            strategy, ka_data, ka_orders, loop,
            position=3, avg_entry=40, entry_side="yes",
            yes_bid=55, yes_ask=60, mid=57.5,
        )
        # Entry fill just happened (within warmup)
        strategy._last_entry_fill_ts[TICKER] = time.time() - 5

        await strategy._check_exits(time.time(), ASSET)

        ka_orders.submit_order.assert_not_called()


class TestStopLossTriggersExit:
    """test_stop_loss_triggers_exit"""

    async def test_stop_loss_triggers_exit(self):
        """pnl_pct <= -SL% => exit (using mid price, after warmup)."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy(
            stop_loss_pct=35.0,
        )
        loop = asyncio.get_running_loop()
        # Entry at 50c, mid=(25+30)/2=27.5c => pnl_pct = (27.5-50)/50*100 = -45%
        _setup_strategy_state(
            strategy, ka_data, ka_orders, loop,
            position=2, avg_entry=50, entry_side="yes",
            yes_bid=25, yes_ask=30, mid=27.5,
        )
        # Past warmup
        strategy._last_entry_fill_ts[TICKER] = time.time() - 60

        await strategy._check_exits(time.time(), ASSET)

        ka_orders.submit_order.assert_called_once()
        req = ka_orders.submit_order.call_args[0][0]
        assert req.action == "sell"
        assert req.side == "yes"
        assert req.size == 2

    async def test_stop_loss_not_triggered_by_spread(self):
        """Spread alone should NOT trigger stop-loss (uses mid, not bid)."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy(
            stop_loss_pct=35.0,
        )
        loop = asyncio.get_running_loop()
        # Entry at 45c, bid=35 ask=45 mid=40 => pnl=(40-45)/45=-11.1%, under 35% SL
        _setup_strategy_state(
            strategy, ka_data, ka_orders, loop,
            position=1, avg_entry=45, entry_side="yes",
            yes_bid=35, yes_ask=45, mid=40,
        )
        strategy._last_entry_fill_ts[TICKER] = time.time() - 60

        await strategy._check_exits(time.time(), ASSET)

        ka_orders.submit_order.assert_not_called()


class TestExitNoPositionSkipped:
    """test_exit_no_position_skipped"""

    async def test_exit_no_position_skipped(self):
        """position=0 => no exit."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(
            strategy, ka_data, ka_orders, loop,
            position=0,
        )

        await strategy._check_exits(time.time(), ASSET)

        ka_orders.submit_order.assert_not_called()


# ===========================================================================
# Fill handling tests
# ===========================================================================


class TestOnFillUpdatesPosition:
    """test_on_fill_updates_position"""

    async def test_on_fill_updates_position(self):
        """Fill callback updates PositionManager."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop, position=0)

        fill = make_fill(
            symbol=TICKER, side="yes", action="buy",
            price=50.0, size=3.0, order_id="ord-fill-1",
        )
        strategy._on_fill(fill)

        pm = strategy._positions[TICKER]
        assert pm.position == 3
        assert pm.avg_entry_price == 50.0
        assert pm.entry_side == "yes"


class TestOnFillRejectsZeroPrice:
    """test_on_fill_rejects_zero_price"""

    async def test_on_fill_rejects_zero_price(self):
        """Fill with price=0 is ignored."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop, position=0)

        fill = make_fill(
            symbol=TICKER, side="yes", action="buy",
            price=0.0, size=1.0, order_id="ord-bad",
        )
        strategy._on_fill(fill)

        pm = strategy._positions[TICKER]
        assert pm.position == 0  # unchanged


# ===========================================================================
# WS position update tests (replaces REST reconciliation)
# ===========================================================================


class TestPositionUpdateFromWS:
    """WS position updates authoritatively set local position."""

    async def test_position_update_corrects_local(self):
        """Exchange WS says position=5, local had 3 => corrected to 5."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop, position=3)

        update = PositionUpdate(
            exchange="kalshi", symbol=TICKER,
            position=5, realized_pnl=0, fees_paid=0, ts=time.time(),
        )
        strategy._on_position_update(update)

        pm = strategy._positions[TICKER]
        assert pm.position == 5

    async def test_position_update_zeros_flat(self):
        """Position going flat clears pending state."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop, position=3)
        strategy._pending_entry_qty[TICKER] = 1
        strategy._pending_exits.add(TICKER)

        update = PositionUpdate(
            exchange="kalshi", symbol=TICKER,
            position=0, realized_pnl=0, fees_paid=0, ts=time.time(),
        )
        strategy._on_position_update(update)

        pm = strategy._positions[TICKER]
        assert pm.position == 0
        assert TICKER not in strategy._pending_entry_qty
        assert TICKER not in strategy._pending_exits


# ===========================================================================
# WS order update tests
# ===========================================================================


class TestOrderUpdateFromWS:
    """WS order updates clear stale _active orders."""

    async def test_order_executed_clears_active(self):
        """Order executed via WS clears _active even if fill arrived first."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop)

        om = strategy._order_mgrs[TICKER]
        # Simulate a resting order that fill couldn't match (race condition)
        from risk.order_manager import ActiveOrder
        om._active["ask"] = ActiveOrder(
            order_id="test-123", side="ask", price=50, size=1, placed_at=time.time(),
        )
        assert om.has_active("ask")

        # WS says order is executed
        update = OrderUpdate(
            exchange="kalshi", order_id="test-123", symbol=TICKER,
            side="no", price=50, initial_count=1, remaining_count=0,
            fill_count=1, status="executed", ts=time.time(),
        )
        strategy._on_order_update(update)

        assert not om.has_active("ask")

    async def test_order_canceled_clears_pending(self):
        """Order canceled via WS clears _active and pending entry tracking."""
        strategy, cb_data, ka_data, ka_orders = _build_strategy()
        loop = asyncio.get_running_loop()
        _setup_strategy_state(strategy, ka_data, ka_orders, loop)

        om = strategy._order_mgrs[TICKER]
        from risk.order_manager import ActiveOrder
        om._active["ask"] = ActiveOrder(
            order_id="test-456", side="ask", price=50, size=1, placed_at=time.time(),
        )
        strategy._pending_entry_qty[TICKER] = 1

        update = OrderUpdate(
            exchange="kalshi", order_id="test-456", symbol=TICKER,
            side="no", price=50, initial_count=1, remaining_count=1,
            fill_count=0, status="canceled", ts=time.time(),
        )
        strategy._on_order_update(update)

        assert not om.has_active("ask")
        assert TICKER not in strategy._pending_entry_qty
