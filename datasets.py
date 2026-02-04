import numpy as np
import pandas as pd

from .features import build_internal_features

# =============================================================================
# DATASET HELPERS (sequence-level vs window-level)
# =============================================================================

def make_sequence_dataset(
    d: pd.DataFrame,
    feature_cols: list[str] | None = None,
    label_col: str = "v_label",
):
    """
    Create a *sequence-level* dataset aligned at each timestamp.

    This is used for ESN experiments (reservoir takes a time sequence):
      U: [T, D] internal features (unscaled here; scale later fold-by-fold)
      y: [T] binary labels aligned to the same timestamps
      idx: timestamps used

    Pipeline context:
      - compute_env_quadrant() creates v_label/a_label from environment only
      - build_internal_features() extracts internal signals (TVOC/eCO2/analog)
      - this function aligns internal features and labels by timestamp
    """
    X_int = build_internal_features(d)

    if feature_cols is not None:
        X_int = X_int[feature_cols].copy()

    y = d[label_col].astype(int)

    # Align and drop any rows with missing values in either features or labels
    Z = pd.concat([X_int, y.rename("y")], axis=1).dropna()

    U = Z[X_int.columns].to_numpy().astype(float)
    y = Z["y"].to_numpy().astype(int)
    idx = Z.index

    return U, y, idx, list(X_int.columns)

# =============================================================================
# 3) WINDOW DATASET (turn time series into supervised samples)
# =============================================================================

def make_window_dataset(
    X_int: pd.DataFrame,
    y_quad: pd.Series,
    window_minutes: int = 15,
    stride_minutes: int = 1,
    add_trend: bool = True,
):
    """
    Create a *window-level* dataset:
      - each sample summarises internal features over a trailing window
      - target is the quadrant at (or before) the window endpoint

    Use case:
      - LogisticRegression baseline on aggregated internal features
      - (and other static classifiers)

    Inputs:
      X_int: internal feature time series (DatetimeIndex)
      y_quad: quadrant labels time series (DatetimeIndex)
    Output:
      Xw: numpy array [N, F]
      yw: numpy array [N] with quadrant 0..3
      idx: window end timestamps (DatetimeIndex)
    """
    if not isinstance(X_int.index, pd.DatetimeIndex):
        raise ValueError("X_int must have a DatetimeIndex.")
    if not isinstance(y_quad.index, pd.DatetimeIndex):
        raise ValueError("y_quad must have a DatetimeIndex.")

    # Align internal features and labels; drop missing
    Z = pd.concat([X_int, y_quad.rename("quadrant")], axis=1).dropna()
    X_int = Z[X_int.columns]
    y_quad = Z["quadrant"].astype(int)

    # Define window endpoints at stride resolution
    idx_end = pd.date_range(
        start=X_int.index.min() + pd.Timedelta(minutes=window_minutes),
        end=X_int.index.max(),
        freq=f"{int(stride_minutes)}min",
    )

    feats, ys, ends = [], [], []
    for t_end in idx_end:
        t_start = t_end - pd.Timedelta(minutes=window_minutes)
        chunk = X_int.loc[t_start:t_end]

        # Skip if the window has too few samples (e.g., missing data)
        if len(chunk) < 5:
            continue

        # Basic window summary statistics
        f = pd.concat(
            [
                chunk.mean().add_suffix("_mean"),
                chunk.std(ddof=0).add_suffix("_std"),
                chunk.min().add_suffix("_min"),
                chunk.max().add_suffix("_max"),
            ]
        )

        # Optional "trend" feature: last - first value in window
        if add_trend:
            trend = (chunk.iloc[-1] - chunk.iloc[0]).add_suffix("_trend")
            f = pd.concat([f, trend])

        # Target is the most recent label at or before t_end
        q = y_quad.loc[:t_end].iloc[-1]

        feats.append(f)
        ys.append(q)
        ends.append(t_end)

    Xw = pd.DataFrame(feats, index=pd.DatetimeIndex(ends))
    yw = pd.Series(ys, index=pd.DatetimeIndex(ends), name="quadrant").astype(int)

    # Final dropna to ensure complete rows
    Z2 = pd.concat([Xw, yw], axis=1).dropna()
    Xw = Z2.drop(columns=["quadrant"])
    yw = Z2["quadrant"].astype(int)

    return Xw.to_numpy(), yw.to_numpy(), yw.index


def apply_lag_to_targets_window_level(X: np.ndarray, y: np.ndarray, idx: pd.DatetimeIndex, lag_minutes: int):
    """
    Apply a prediction horizon at the window level:
      X(t_end) predicts y(t_end + lag)

    Returns:
      X_new, y_new, idx_new  where idx_new are the *original* X timestamps kept
      after alignment/dropna (i.e., the timestamps of X rows still present).
    """
    if lag_minutes == 0:
        return X, y, idx

    y_s = pd.Series(y, index=idx)
    y_s.index = y_s.index + pd.Timedelta(minutes=int(lag_minutes))

    X_df = pd.DataFrame(X, index=idx)
    Z = pd.concat([X_df, y_s.rename("quadrant")], axis=1).dropna()

    X_new = Z.iloc[:, :-1].to_numpy()
    y_new = Z["quadrant"].astype(int).to_numpy()
    idx_new = Z.index  # IMPORTANT: aligns with X_new rows

    return X_new, y_new, idx_new


# =============================================================================
# BINARY WINDOW DATASET + BINARY BASELINE RUNNERS
# =============================================================================

def make_window_dataset_binary(
    X_int: pd.DataFrame,
    y_bin: pd.Series,
    window_minutes: int = 15,
    stride_minutes: int = 15,
    add_trend: bool = True,
):
    """
    Window-level dataset builder for *binary* targets (valence or arousal).

    Identical to make_window_dataset(), but:
      - target y is binary (0/1)
      - adds a few hand-crafted window features (iqr, mean abs diff, max ptp)
        that can help detect variability / volatility
    """
    Z = pd.concat([X_int, y_bin.rename("y")], axis=1).dropna()
    X_int = Z[X_int.columns]
    y_bin = Z["y"].astype(int)

    idx_end = pd.date_range(
        start=X_int.index.min() + pd.Timedelta(minutes=window_minutes),
        end=X_int.index.max(),
        freq=f"{int(stride_minutes)}min",
    )

    feats, ys, ends = [], [], []

    for t_end in idx_end:
        t_start = t_end - pd.Timedelta(minutes=window_minutes)
        chunk = X_int.loc[t_start:t_end]
        if len(chunk) < 5:
            continue

        f = pd.concat(
            [
                chunk.mean().add_suffix("_mean"),
                chunk.std(ddof=0).add_suffix("_std"),
                chunk.min().add_suffix("_min"),
                chunk.max().add_suffix("_max"),
            ]
        )

        if add_trend:
            trend = (chunk.iloc[-1] - chunk.iloc[0]).add_suffix("_trend")
            f = pd.concat([f, trend])

        # Hand-crafted features (optional)
        if "TVOC" in chunk.columns:
            f["TVOC_iqr"] = chunk["TVOC"].quantile(0.75) - chunk["TVOC"].quantile(0.25)

        if "analog_raw_rms" in chunk.columns:
            f["analog_rms_change"] = chunk["analog_raw_rms"].diff().abs().mean()

        if "analog_raw_ptp" in chunk.columns:
            f["analog_ptp_max"] = chunk["analog_raw_ptp"].max()

        # Binary target at or before t_end
        y = y_bin.loc[:t_end].iloc[-1]

        feats.append(f)
        ys.append(y)
        ends.append(t_end)

    Xw = pd.DataFrame(feats, index=pd.DatetimeIndex(ends))
    yw = pd.Series(ys, index=pd.DatetimeIndex(ends), name="y").astype(int)

    Z2 = pd.concat([Xw, yw], axis=1).dropna()
    return Z2.drop(columns=["y"]).to_numpy(), Z2["y"].to_numpy(), Z2.index

import numpy as np
import pandas as pd
from .features import build_internal_features


# DATASET CURATED FOR THE CNN #

def make_cnn_sequence_windows(
    d: pd.DataFrame,
    label_col: str,
    window_minutes: int,
    stride_minutes: int,
    sample_seconds: int = 5,
):
    """
    Build CNN windows:
      X: [N, C, T] float32
      y: [N] int64
      idx_end: DatetimeIndex (window endpoints)
      channel_names: list[str]
    """
    X_int = build_internal_features(d)                 # DataFrame [time, C]
    y_s = d[label_col].astype(int)                     # Series [time]

    Z = pd.concat([X_int, y_s.rename("y")], axis=1).dropna()
    X_int = Z[X_int.columns]
    y_s = Z["y"].astype(int)

    T = int(round(window_minutes * 60 / sample_seconds))
    step = int(round(stride_minutes * 60 / sample_seconds))
    if T < 4:
        raise ValueError("Window too short for CNN.")
    if step < 1:
        step = 1

    X_np = X_int.to_numpy().astype(np.float32)         # [M, C]
    y_np = y_s.to_numpy().astype(np.int64)             # [M]
    idx = X_int.index

    Xw, yw, ends = [], [], []
    for end_i in range(T - 1, len(X_np), step):
        start_i = end_i - (T - 1)
        x_win = X_np[start_i:end_i + 1]                # [T, C]
        # transpose for Conv1d: [C, T]
        Xw.append(x_win.T)
        yw.append(y_np[end_i])                         # label at window end
        ends.append(idx[end_i])

    Xw = np.stack(Xw, axis=0)                          # [N, C, T]
    yw = np.asarray(yw, dtype=np.int64)                # [N]
    return Xw, yw, pd.DatetimeIndex(ends), list(X_int.columns)
