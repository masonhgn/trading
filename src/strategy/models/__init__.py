"""Pluggable theoretical value models.

Usage:
    from strategy.models import get_model, MarketState

    model = get_model("gbm")        # or "logistic", "lgbm", etc.
    state = MarketState(spot=68000, spot_at_open=67500, vol_15m=0.002,
                        time_remaining_sec=600, asset="BTC")
    prob = model.fair_value(state)   # P(up) in [0, 1]
"""

from strategy.models.base import MarketState, TheoModel
from strategy.models.gbm import GBMModel
from strategy.models.logistic import LogisticModel

# -- Registry ----------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "gbm": GBMModel,
    "logistic": LogisticModel,
}


def register_model(name: str, cls: type) -> None:
    """Register a new model class by name."""
    _REGISTRY[name] = cls


def get_model(name: str, **kwargs) -> TheoModel:
    """Instantiate a model by name.

    Raises KeyError if the name is not registered.
    """
    cls = _REGISTRY[name]
    return cls(**kwargs)


def available_models() -> list[str]:
    return list(_REGISTRY.keys())


__all__ = [
    "MarketState",
    "TheoModel",
    "GBMModel",
    "LogisticModel",
    "get_model",
    "register_model",
    "available_models",
]
