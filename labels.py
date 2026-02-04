import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# =============================================================================
# HELPERS (used across label creation, dataset building, and evaluation)
# =============================================================================

def robust_z(s: pd.Series) -> pd.Series:
    """
    Robust normalisation for noisy / heavy-tailed signals.

    Returns a robust z-score using:
      (x - median) / IQR
    and falls back to std if IQR is zero or invalid.

    Why: environmental signals (light, temp, humidity) can have outliers and
    non-Gaussian distributions; robust scaling stabilises downstream label rules.
    """
    s = s.astype(float)
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if (iqr == 0) or (not np.isfinite(iqr)):
        sd = s.std()
        return (s - med) / (sd if sd else 1.0)
    return (s - med) / iqr


def quadrant_from_sign(v_sign: np.ndarray, a_sign: np.ndarray) -> np.ndarray:
    """
    Combine binary valence sign and arousal sign into a single 4-class label.

    Convention:
      v_sign in {0,1} -> 0=V-, 1=V+
      a_sign in {0,1} -> 0=A-, 1=A+

    Quadrant encoding:
      q = 2*v + a  -> {0,1,2,3}
        0: V-, A-
        1: V-, A+
        2: V+, A-
        3: V+, A+
    """
    return (2 * v_sign + a_sign).astype(int)

# =============================================================================
# 1) ENVIRONMENT-DEFINED LABELS (ground truth proxy)
# =============================================================================

def compute_env_quadrant(df: pd.DataFrame, margin: float = 0.2, smooth_minutes: int = 5) -> pd.DataFrame:
    """
    Compute environment-defined valence/arousal labels and quadrants.

    This is your "definition of emotion" grounded in environment, NOT internal plant signals.
    You later train models to predict these labels using internal signals only.

    Steps:
      1) Compute robust z-scored environmental factors:
         - L: light (log IR)
         - T: temperature
         - H: humidity
      2) Residualise light against {T,H} via linear regression:
         - removes confounding where IR changes are partially explained by T/H
      3) Define valence and arousal scores:
         V_raw = +L_resid -T +H
         A_raw = +L_resid +T -H
      4) Smooth V/A by rolling median (reduces label jitter)
      5) Apply a dead-zone (margin) so small fluctuations near 0 are dropped
      6) Convert signs to binary labels and 4-class quadrant

    Returns:
      DataFrame filtered to "confident" windows + columns:
        V_raw, A_raw, V_s, A_s, v_label, a_label, quadrant
    """
    d = df.copy()

    # Minimal sanity filter to avoid log(0) and extreme noise in low IR
    d = d[d["ir"] > 1].copy()
    d["ir_log"] = np.log1p(d["ir"].astype(float))

    if not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError("compute_env_quadrant expects a DatetimeIndex.")

    # Robustly normalise environmental drivers
    L = robust_z(d["ir_log"])
    T = robust_z(d["temperature"])
    H = robust_z(d["humidity"])

    # Residualise light to reduce spurious correlations with T/H
    X_th = np.vstack([T.to_numpy(), H.to_numpy()]).T
    lr = LinearRegression().fit(X_th, L.to_numpy())
    L_resid = L.to_numpy() - lr.predict(X_th)

    # Define environment-based "emotion axes"
    d["V_raw"] = (+L_resid) + (-T.to_numpy()) + (+H.to_numpy())
    d["A_raw"] = (+L_resid) + (+T.to_numpy()) + (-H.to_numpy())

    # Smooth to reduce label flicker (rolling median is robust)
    roll = f"{int(smooth_minutes)}min"
    d["V_s"] = d["V_raw"].rolling(roll, min_periods=1).median()
    d["A_s"] = d["A_raw"].rolling(roll, min_periods=1).median()

    # Dead-zone: remove ambiguous points near 0 in either axis
    keep = (np.abs(d["V_s"]) > margin) & (np.abs(d["A_s"]) > margin)
    d = d.loc[keep].copy()

    # Binary labels
    d["v_label"] = (d["V_s"] > 0).astype(int)
    d["a_label"] = (d["A_s"] > 0).astype(int)

    # 4-class quadrant
    d["quadrant"] = quadrant_from_sign(
        d["v_label"].to_numpy(),
        d["a_label"].to_numpy()
    )

    return d
