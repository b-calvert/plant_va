from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from plant_va.data import load_sensor_df_chunked
from plant_va.labels import compute_env_quadrant
from plant_va.features import build_internal_features
from plant_va.datasets import make_window_dataset_binary
from plant_va.decoders import RidgeEnvDecoder, summarise_multi_target

from plant_va.plant_va_config.presets import (
    DEFAULT_DATA,
    DEFAULT_LABELS,
    VALENCE_WINDOW,
    AROUSAL_WINDOWS,
    DEFAULT_CV,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _align_env_targets(
    df_full: pd.DataFrame,
    idx_base: pd.DatetimeIndex,
    targets: list[str],
    delta_minutes: int,
) -> np.ndarray:
    """
    For each time in idx_base, returns env targets at time + delta (ffill).
    Output: Y [N, len(targets)]
    """
    idx_tgt = idx_base + pd.Timedelta(minutes=int(delta_minutes))
    env = df_full[targets].copy()
    Y = env.reindex(idx_tgt, method="ffill").to_numpy(dtype=float)
    return Y


def _drop_nan_rows(*arrays):
    """
    Drop rows where ANY array has NaNs.
    arrays can be 1D or 2D, but must have same first dimension.
    Returns (filtered_arrays_list, mask_ok).
    """
    n = arrays[0].shape[0]
    mask = np.ones(n, dtype=bool)

    for a in arrays:
        if a.ndim == 1:
            mask &= np.isfinite(a)
        else:
            mask &= np.all(np.isfinite(a), axis=1)

    out = []
    for a in arrays:
        out.append(a[mask] if a.ndim == 1 else a[mask, :])

    return out, mask


def _oof_state_probabilities(
    XV_all: np.ndarray,
    v_true: np.ndarray,
    XA_on_V: np.ndarray,
    a_true_on_V: np.ndarray,
    n_splits: int,
    seed: int = 0,
):
    """
    Compute out-of-fold probabilities p(V+), p(A+) aligned to the valence timeline.

    Returns:
      pV_hat, pA_hat: arrays length N (N = len(v_true))
    """
    N = len(v_true)
    pV_hat = np.full(N, np.nan, dtype=float)
    pA_hat = np.full(N, np.nan, dtype=float)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold, (tr, te) in enumerate(tscv.split(XV_all), 1):
        # ---- Valence model on XV ----
        scV = StandardScaler()
        XVtr = scV.fit_transform(XV_all[tr])
        XVte = scV.transform(XV_all[te])

        baseV = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=seed)
        clfV = CalibratedClassifierCV(baseV, method="sigmoid", cv=3)
        clfV.fit(XVtr, v_true[tr])
        pV_hat[te] = clfV.predict_proba(XVte)[:, 1]

        # ---- Arousal model on XA_on_V ----
        scA = StandardScaler()
        XAtr = scA.fit_transform(XA_on_V[tr])
        XAte = scA.transform(XA_on_V[te])

        baseA = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=seed + 1)
        clfA = CalibratedClassifierCV(baseA, method="sigmoid", cv=3)
        clfA.fit(XAtr, a_true_on_V[tr])
        pA_hat[te] = clfA.predict_proba(XAte)[:, 1]

        print(f"Fold {fold} | state OOF probs filled: te={len(te)}")

    return pV_hat, pA_hat


def _cv_decode_env(
    X: np.ndarray,
    Y: np.ndarray,
    targets: list[str],
    n_splits: int,
    ridge_alpha: float = 1.0,
    scale_X: bool = False,
):
    """
    TimeSeriesSplit CV decoding:
      Fit ridge decoder on train, predict on test, accumulate OOF preds.

    If scale_X=True, we standardise X fold-by-fold (recommended for direct baseline on XV).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = np.zeros_like(Y, dtype=float)

    for fold, (tr, te) in enumerate(tscv.split(Y), 1):
        Xtr = X[tr]
        Xte = X[te]

        if scale_X:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

        dec = RidgeEnvDecoder(alpha=ridge_alpha).fit(Xtr, Y[tr], targets=targets)
        preds[te] = dec.predict(Xte)

    return preds


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def main(
    data=DEFAULT_DATA,
    labels=DEFAULT_LABELS,
    v_window=VALENCE_WINDOW,
    a_window=AROUSAL_WINDOWS,
    cv=DEFAULT_CV,
    env_targets=("temperature", "humidity"),
    horizons_minutes=(0, 15, 60),
    ridge_alpha=1.0,
):
    """
    Internal -> estimated state -> env forecast.

    Three routes per horizon Δ:
      (1) Oracle state: [v_true, a_true] -> env(t+Δ)
      (2) Estimated state: [pV_hat, pA_hat] -> env(t+Δ)
      (3) Direct baseline: internal features (XV) -> env(t+Δ)

    Valence and arousal can use different windows, aligned onto valence timeline (idxV).
    """

    # 1) Load df in chunks (safer for Influx)
    # Adjust start/stop to desired coverage.
    df_full = load_sensor_df_chunked(
        start="2025-12-18T00:00:00Z",
        stop="2026-01-20T03:30:00Z",
        rule=data.rule,
        analog_mode=data.analog_mode,
        chunk="3h",
    )

    # 2) Compute environment-defined labels (filters rows by IR threshold + margin)
    d = compute_env_quadrant(df_full, margin=labels.margin, smooth_minutes=labels.smooth_minutes)

    # 3) Internal features (plant signals only)
    X_int = build_internal_features(d)

    # 4) Build window datasets
    # Valence windows define the MASTER timeline idxV
    XV, v_true, idxV = make_window_dataset_binary(
        X_int,
        d["v_label"],
        window_minutes=v_window.window_minutes,
        stride_minutes=v_window.stride_minutes,
        add_trend=v_window.add_trend,
    )

    XA, a_true, idxA = make_window_dataset_binary(
        X_int,
        d["a_label"],
        window_minutes=a_window.window_minutes,
        stride_minutes=a_window.stride_minutes,
        add_trend=a_window.add_trend,
    )

    # Align arousal labels/features onto valence timeline
    a_true_on_V = pd.Series(a_true, index=idxA).reindex(idxV, method="ffill").to_numpy().astype(int)
    v_true_on_V = v_true.astype(int)

    XA_on_V = pd.DataFrame(XA.astype(float), index=idxA).reindex(idxV, method="ffill").to_numpy()
    XV_all = XV.astype(float)

    print("GLOBAL STATE DISTRIBUTIONS (aligned on valence windows)")
    print(f"V (0/1): {np.bincount(v_true_on_V, minlength=2)} | N={len(v_true_on_V)}")
    print(f"A (0/1): {np.bincount(a_true_on_V, minlength=2)} | N={len(a_true_on_V)}")
    print(f"Valence windows: N={len(idxV)} stride={v_window.stride_minutes}min window={v_window.window_minutes}min")
    print(f"Arousal  windows: N={len(idxA)} stride={a_window.stride_minutes}min window={a_window.window_minutes}min")
    print("Env targets:", list(env_targets))
    print("Horizons (minutes):", list(horizons_minutes))

    # 5) Out-of-fold state probability estimates (aligned to idxV)
    pV_hat, pA_hat = _oof_state_probabilities(
        XV_all=XV_all,
        v_true=v_true_on_V,
        XA_on_V=XA_on_V,
        a_true_on_V=a_true_on_V,
        n_splits=cv.n_splits,
        seed=0,
    )

    # 6) Build state matrices (oracle vs estimated)
    Z_oracle = np.column_stack([v_true_on_V.astype(float), a_true_on_V.astype(float)])  # [N,2]
    Z_hat = np.column_stack([pV_hat, pA_hat])  # [N,2]

    # 7) Drop rows with NaNs in any required array (common early on due to ffill gaps)
    (out_arrays, mask_ok) = _drop_nan_rows(Z_oracle, Z_hat, XV_all)
    Z_oracle_ok, Z_hat_ok, XV_ok = out_arrays
    idx_ok = idxV[mask_ok]

    assert len(idx_ok) == Z_oracle_ok.shape[0] == Z_hat_ok.shape[0] == XV_ok.shape[0]

    # 8) For each horizon, build env targets on idx_ok and evaluate 3 decoders
    targets = list(env_targets)

    for delta in horizons_minutes:
        Y = _align_env_targets(df_full=df_full, idx_base=idx_ok, targets=targets, delta_minutes=int(delta))

        # Also drop any NaNs in Y (data gaps)
        (out_arrays2, mask2) = _drop_nan_rows(Z_oracle_ok, Z_hat_ok, XV_ok, Y)
        Z_oracle2, Z_hat2, XV2, Y2 = out_arrays2

        print("\n" + "=" * 72)
        print(f"ENV FORECAST: Δ={delta} minutes | N={len(Y2)} | targets={targets}")

        # (1) Oracle state -> env
        preds_oracle = _cv_decode_env(
            X=Z_oracle2, Y=Y2, targets=targets, n_splits=cv.n_splits, ridge_alpha=ridge_alpha, scale_X=False
        )

        # (2) Estimated state -> env
        preds_hat = _cv_decode_env(
            X=Z_hat2, Y=Y2, targets=targets, n_splits=cv.n_splits, ridge_alpha=ridge_alpha, scale_X=False
        )

        # (3) Direct baseline (internal XV) -> env (scale fold-by-fold)
        preds_direct = _cv_decode_env(
            X=XV2, Y=Y2, targets=targets, n_splits=cv.n_splits, ridge_alpha=ridge_alpha, scale_X=True
        )

        # Summaries per target
        m_oracle = summarise_multi_target(Y2, preds_oracle, targets=targets)
        m_hat = summarise_multi_target(Y2, preds_hat, targets=targets)
        m_direct = summarise_multi_target(Y2, preds_direct, targets=targets)

        def _print_block(title, metrics_dict):
            print(f"\n{title}")
            for t in targets:
                mm = metrics_dict[t]
                print(f"  {t:>12s}: R2={mm['r2']:+.3f} | RMSE={mm['rmse']:.3f} | MAE={mm['mae']:.3f}")

        _print_block("Oracle state (true V/A) -> env", m_oracle)
        _print_block("Estimated state (pV_hat,pA_hat) -> env", m_hat)
        _print_block("Direct baseline (XV internal) -> env", m_direct)


if __name__ == "__main__":
    main()
