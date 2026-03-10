"""Logistic regression model for 15-min binary prediction.

Estimates P(spot finishes above open) using microstructure features
fit via MLE (sklearn LogisticRegression).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from strategy.models.base import MarketState

logger = logging.getLogger(__name__)

# Feature names in the order the model expects them.
FEATURE_NAMES = [
    "current_return",   # log(spot / spot_open)
    "vol_15m",          # realized vol scaled to 15 min
    "time_frac",        # time_remaining / 900
    "flow_imbalance",   # (buy - sell) / total volume, rolling
    "book_imbalance",   # (bid_depth - ask_depth) / total depth
    "momentum_1m",      # log return over last 60s
]

DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent.parent / "models" / "fv_logistic_v1.pkl"


class LogisticModel:
    """Logistic regression fair-value model.

    Features are standardized internally (the scaler is saved alongside
    the model coefficients in the pickle).
    """

    name = "logistic"

    def __init__(self, model_path: Path | str | None = None) -> None:
        self._model = None
        self._scaler = None
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._load_attempted = False

    # -- Persistence ----------------------------------------------------------

    def _try_load(self) -> bool:
        """Lazy-load model from disk on first call."""
        if self._load_attempted:
            return self._model is not None
        self._load_attempted = True

        if not self._model_path.exists():
            logger.warning("Logistic model not found at %s", self._model_path)
            return False

        try:
            import pickle
            with open(self._model_path, "rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._scaler = data.get("scaler")
            logger.info("Loaded logistic model from %s", self._model_path)
            return True
        except Exception as e:
            logger.error("Failed to load logistic model: %s", e)
            return False

    def save(self, path: Path | str | None = None) -> None:
        """Save model + scaler to disk."""
        import pickle
        save_path = Path(path) if path else self._model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler}, f)
        logger.info("Saved logistic model to %s", save_path)

    # -- Training -------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Fit the model on labeled data.

        Args:
            X: Feature matrix, shape (n_samples, len(FEATURE_NAMES)).
            y: Binary labels, 1 = spot finished above open.

        Returns:
            Dict with training metrics.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
        )
        self._model.fit(X_scaled, y)

        # Cross-validated metrics
        cv_scores = cross_val_score(
            self._model, X_scaled, y, cv=5, scoring="accuracy",
        )
        cv_logloss = -cross_val_score(
            self._model, X_scaled, y, cv=5, scoring="neg_log_loss",
        )

        coefs = dict(zip(FEATURE_NAMES, self._model.coef_[0]))

        metrics = {
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "cv_logloss_mean": float(cv_logloss.mean()),
            "cv_logloss_std": float(cv_logloss.std()),
            "intercept": float(self._model.intercept_[0]),
            "coefficients": coefs,
        }
        return metrics

    # -- Inference ------------------------------------------------------------

    @staticmethod
    def extract_features(state: MarketState) -> np.ndarray:
        """Build feature vector from MarketState.

        Missing optional features are filled with 0 (neutral).
        """
        return np.array([
            state.current_return,
            state.vol_15m,
            state.time_frac,
            state.flow_imbalance if state.flow_imbalance is not None else 0.0,
            state.book_imbalance if state.book_imbalance is not None else 0.0,
            state.momentum_1m if state.momentum_1m is not None else 0.0,
        ], dtype=np.float64)

    def fair_value(self, state: MarketState) -> float:
        """Return P(spot finishes above open) in [0, 1]."""
        if state.time_remaining_sec <= 0:
            return 1.0 if state.spot > state.spot_at_open else 0.0

        if not self._try_load():
            # No model available — fall back to 0.5 (no opinion)
            return 0.5

        x = self.extract_features(state).reshape(1, -1)
        if self._scaler is not None:
            x = self._scaler.transform(x)

        prob = self._model.predict_proba(x)[0, 1]
        return float(np.clip(prob, 0.001, 0.999))
