"""GBM (Black-Scholes) analytical model for 15-min binary options.

Wraps the existing compute_fair_value() function from fair_value.py
behind the TheoModel interface.
"""

from __future__ import annotations

from strategy.fair_value import compute_fair_value
from strategy.models.base import MarketState


class GBMModel:
    """Analytical GBM model: P(up) = Φ(current_return / (σ√T))."""

    name = "gbm"

    def fair_value(self, state: MarketState) -> float:
        return compute_fair_value(
            state.spot,
            state.spot_at_open,
            state.vol_15m,
            state.time_remaining_sec,
        )
