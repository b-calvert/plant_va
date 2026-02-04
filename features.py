import numpy as np
import pandas as pd

# =============================================================================
# 2) INTERNAL FEATURES (plant signals only)
# =============================================================================

def build_internal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the *internal-state* feature matrix (plant signals only).

    Includes:
      - TVOC, eCO2 (gas sensors)  -> log1p transform
      - engineered analog features if present:
          analog_raw_mean/std/rms/ptp
          analog_voltage_mean/std/rms/ptp

    Important:
      - This assumes analog_* are already aggregated features, not raw 50 Hz streams.
        If you have raw 50 Hz, you should first compute window features upstream.

    Also applies:
      - Daily de-meaning (group by day) to remove slow regime shifts / diurnal baseline.
        This helps prevent the model from learning "time of day" rather than responses.
    """
    base_cols = []

    # Gas sensors (optional)
    for c in ["TVOC", "eCO2"]:
        if c in df.columns:
            base_cols.append(c)

    # Analog engineered features (optional)
    analog_feature_cols = [
        "analog_raw_mean", "analog_raw_std", "analog_raw_rms", "analog_raw_ptp",
        "analog_voltage_mean", "analog_voltage_std", "analog_voltage_rms", "analog_voltage_ptp",
    ]
    for c in analog_feature_cols:
        if c in df.columns:
            base_cols.append(c)

    if not base_cols:
        raise ValueError("No internal feature columns found (TVOC/eCO2 and/or analog_* features).")

    X = df[base_cols].copy()

    # Log transform gases (stabilises variance, handles heavy tails)
    for c in ["TVOC", "eCO2"]:
        if c in X.columns:
            X[c] = np.log1p(X[c].clip(lower=0))

    # Remove slow regime leakage: subtract daily mean per feature
    day_index = X.index.floor("D")
    for col in X.columns:
        X[col] = X[col] - X.groupby(day_index)[col].transform("mean")

    return X