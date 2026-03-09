"""Event-driven backtest engine for Kalshi 15-min strategy.

Processes a time-ordered stream of Coinbase mid-price updates and
Kalshi market snapshots, computes fair values, and simulates trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from strategy.fair_value import ContractInfo, VolEstimator, compute_fair_value


@dataclass(slots=True)
class TradeRecord:
    ts: float
    ticker: str
    asset: str
    side: str           # "buy_yes" or "buy_no"
    price_cents: float  # what we paid
    qty: int
    fair_value: float   # our model's fair value at trade time (cents)
    edge_cents: float   # fair_value - market_mid (for buy_yes), negative for buy_no
    spot: float         # Coinbase spot at trade time
    spot_at_open: float
    time_remaining: float


@dataclass(slots=True)
class ContractResult:
    ticker: str
    asset: str
    window_start: float
    window_end: float
    spot_at_open: float
    spot_at_close: float
    settled_yes: bool
    trades: list[TradeRecord]
    pnl_cents: float        # total P&L in cents
    position_at_settle: int  # net YES position at settlement
    max_position: int


@dataclass
class BacktestConfig:
    edge_threshold_cents: float = 5.0
    fee_per_side_cents: float = 5.0
    max_position_per_contract: int = 5
    cooldown_sec: float = 10.0
    min_time_remaining_sec: float = 120.0
    max_time_remaining_sec: float = 840.0  # 14 min (skip first minute)
    vol_lookback_sec: float = 600.0
    min_spot_move_pct: float = 0.001  # skip evaluation if spot hasn't moved

    # ML-derived filters
    max_spot_return_pct: float = 0.1    # skip when |spot return from open| > this %
    max_vol_15m: float = 0.0            # skip when vol too high (0 = disabled)
    prefer_early_entry: bool = False     # require time_remaining > 9 min (540s)


class BacktestEngine:
    """Runs an event-driven backtest over historical data."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

        # Per-asset state
        self._spot: dict[str, float] = {}           # asset -> current spot
        self._vol: dict[str, VolEstimator] = {}      # asset -> vol estimator
        self._last_eval_spot: dict[str, float] = {}  # asset -> spot at last eval

        # Per-contract state
        self._contracts: dict[str, ContractInfo] = {}
        self._spot_at_open: dict[str, float] = {}    # ticker -> spot at window open
        self._kalshi_bid: dict[str, float] = {}      # ticker -> yes_bid (cents)
        self._kalshi_ask: dict[str, float] = {}      # ticker -> yes_ask (cents)
        self._position: dict[str, int] = {}           # ticker -> net YES position
        self._last_trade_ts: dict[str, float] = {}   # ticker -> last trade timestamp
        self._trades: dict[str, list[TradeRecord]] = {}
        self._max_pos: dict[str, int] = {}

        # Results
        self.results: list[ContractResult] = []

    def register_contract(self, info: ContractInfo) -> None:
        self._contracts[info.ticker] = info
        self._position[info.ticker] = 0
        self._last_trade_ts[info.ticker] = 0
        self._trades[info.ticker] = []
        self._max_pos[info.ticker] = 0

        # Initialize vol estimator for this asset if needed
        if info.asset not in self._vol:
            self._vol[info.asset] = VolEstimator(self.config.vol_lookback_sec)

    def on_spot_update(self, ts: float, asset: str, mid: float) -> None:
        """Process a Coinbase mid-price update."""
        self._spot[asset] = mid
        self._vol[asset].update(ts, mid)

        # Set window-open price for any contracts starting now
        for ticker, info in self._contracts.items():
            if ticker not in self._spot_at_open:
                if ts >= info.window_start:
                    self._spot_at_open[ticker] = mid

        # Skip evaluation if spot hasn't moved enough
        last = self._last_eval_spot.get(asset, 0)
        if last > 0 and abs(mid / last - 1) < self.config.min_spot_move_pct / 100:
            return
        self._last_eval_spot[asset] = mid

        # Evaluate trading signals for active contracts
        self._evaluate_signals(ts, asset)

    def on_kalshi_update(
        self, ts: float, ticker: str, yes_bid: float, yes_ask: float
    ) -> None:
        """Process a Kalshi market snapshot."""
        self._kalshi_bid[ticker] = yes_bid
        self._kalshi_ask[ticker] = yes_ask

        # Also evaluate on Kalshi updates (new quote might create opportunity)
        info = self._contracts.get(ticker)
        if info:
            self._evaluate_signals(ts, info.asset)

    def settle_contract(self, ticker: str, spot_at_close: float) -> ContractResult | None:
        """Settle a contract and compute P&L."""
        info = self._contracts.get(ticker)
        if not info:
            return None

        spot_at_open = self._spot_at_open.get(ticker)
        if spot_at_open is None:
            return None

        settled_yes = spot_at_close > spot_at_open
        position = self._position.get(ticker, 0)
        trades = self._trades.get(ticker, [])

        # P&L calculation
        # Each YES contract pays 100 cents if YES, 0 if NO
        # Each NO contract pays 100 cents if NO, 0 if YES
        pnl = 0.0
        for trade in trades:
            if trade.side == "buy_yes":
                # Paid trade.price_cents, receives 100 if yes, 0 if no
                payout = 100.0 if settled_yes else 0.0
                pnl += (payout - trade.price_cents - self.config.fee_per_side_cents) * trade.qty
            else:  # buy_no
                # Paid trade.price_cents for NO, receives 100 if no, 0 if yes
                payout = 100.0 if not settled_yes else 0.0
                pnl += (payout - trade.price_cents - self.config.fee_per_side_cents) * trade.qty

        result = ContractResult(
            ticker=ticker,
            asset=info.asset,
            window_start=info.window_start,
            window_end=info.window_end,
            spot_at_open=spot_at_open,
            spot_at_close=spot_at_close,
            settled_yes=settled_yes,
            trades=trades,
            pnl_cents=pnl,
            position_at_settle=position,
            max_position=self._max_pos.get(ticker, 0),
        )
        self.results.append(result)

        # Cleanup
        for key in [self._contracts, self._spot_at_open, self._kalshi_bid,
                     self._kalshi_ask, self._position, self._last_trade_ts,
                     self._trades, self._max_pos]:
            key.pop(ticker, None)

        return result

    def _evaluate_signals(self, ts: float, asset: str) -> None:
        """Check all active contracts for this asset for trading opportunities."""
        spot = self._spot.get(asset)
        if spot is None:
            return

        vol_est = self._vol.get(asset)
        if vol_est is None:
            return
        vol = vol_est.vol_15m()

        cfg = self.config

        for ticker, info in list(self._contracts.items()):
            if info.asset != asset:
                continue

            spot_open = self._spot_at_open.get(ticker)
            if spot_open is None:
                continue

            time_remaining = info.window_end - ts
            if time_remaining < cfg.min_time_remaining_sec:
                continue
            if time_remaining > cfg.max_time_remaining_sec:
                continue

            # Need Kalshi quotes
            bid = self._kalshi_bid.get(ticker)
            ask = self._kalshi_ask.get(ticker)
            if bid is None or ask is None or bid <= 0 or ask <= 0:
                continue
            if ask <= bid:
                continue

            # Cooldown
            if ts - self._last_trade_ts.get(ticker, 0) < cfg.cooldown_sec:
                continue

            # Position limit
            pos = self._position.get(ticker, 0)
            if abs(pos) >= cfg.max_position_per_contract:
                continue

            # ML-derived filters
            spot_return_pct = abs(spot / spot_open - 1) * 100
            if cfg.max_spot_return_pct > 0 and spot_return_pct > cfg.max_spot_return_pct:
                continue

            if cfg.max_vol_15m > 0 and vol > cfg.max_vol_15m:
                continue

            if cfg.prefer_early_entry and time_remaining < 540:
                continue

            # Fair value
            fv = compute_fair_value(spot, spot_open, vol, time_remaining)
            fv_cents = fv * 100

            kalshi_mid = (bid + ask) / 2
            edge = fv_cents - kalshi_mid

            min_edge = cfg.edge_threshold_cents + cfg.fee_per_side_cents

            if edge > min_edge and pos < cfg.max_position_per_contract:
                # Buy YES at the ask
                self._execute_trade(ts, ticker, info, "buy_yes", ask, fv_cents,
                                    edge, spot, spot_open, time_remaining)
            elif edge < -min_edge and pos > -cfg.max_position_per_contract:
                # Buy NO at (100 - bid)
                no_price = 100 - bid
                self._execute_trade(ts, ticker, info, "buy_no", no_price, fv_cents,
                                    edge, spot, spot_open, time_remaining)

    def _execute_trade(
        self, ts: float, ticker: str, info: ContractInfo,
        side: str, price_cents: float, fv_cents: float, edge: float,
        spot: float, spot_open: float, time_remaining: float,
    ) -> None:
        qty = 1  # trade 1 contract at a time

        trade = TradeRecord(
            ts=ts, ticker=ticker, asset=info.asset,
            side=side, price_cents=price_cents, qty=qty,
            fair_value=fv_cents, edge_cents=abs(edge),
            spot=spot, spot_at_open=spot_open,
            time_remaining=time_remaining,
        )
        self._trades[ticker].append(trade)
        self._last_trade_ts[ticker] = ts

        if side == "buy_yes":
            self._position[ticker] = self._position.get(ticker, 0) + qty
        else:
            self._position[ticker] = self._position.get(ticker, 0) - qty

        abs_pos = abs(self._position.get(ticker, 0))
        self._max_pos[ticker] = max(self._max_pos.get(ticker, 0), abs_pos)
