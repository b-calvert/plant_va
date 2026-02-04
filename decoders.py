from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


@dataclass
class RidgeEnvDecoder:
    """
    Simple multi-target environment decoder: X -> [temperature, humidity].
    Uses one Ridge model per target for clarity.
    """
    alpha: float = 1.0
    models: Dict[str, Ridge] | None = None
    targets: List[str] | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, targets: List[str]) -> "RidgeEnvDecoder":
        assert Y.ndim == 2, "Y must be [N, n_targets]"
        assert Y.shape[1] == len(targets)

        self.targets = list(targets)
        self.models = {}

        for j, t in enumerate(self.targets):
            m = Ridge(alpha=self.alpha)
            m.fit(X, Y[:, j])
            self.models[t] = m

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.models is not None and self.targets is not None
        preds = []
        for t in self.targets:
            preds.append(self.models[t].predict(X))
        return np.column_stack(preds)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    y_true, y_pred: [N] vectors
    Returns R2, RMSE, MAE.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"r2": float(r2), "rmse": rmse, "mae": mae}


def summarise_multi_target(Y_true: np.ndarray, Y_pred: np.ndarray, targets: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Returns dict target -> metrics dict.
    """
    out = {}
    for j, t in enumerate(targets):
        out[t] = regression_metrics(Y_true[:, j], Y_pred[:, j])
    return out
