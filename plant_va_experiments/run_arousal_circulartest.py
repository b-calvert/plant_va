import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from plant_va.data import load_sensor_df
from plant_va.labels import compute_env_quadrant
from plant_va.features import build_internal_features
from plant_va.datasets import make_window_dataset_binary
from plant_va.plant_va_config.presets import (
    DEFAULT_DATA, DEFAULT_LABELS, AROUSAL_WINDOWS, DEFAULT_CV
)

def eval_cv_binary_logreg(Xw, y, n_splits: int, C: float = 1.0):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    accs, baccs, f1s = [], [], []
    cm = np.zeros((2, 2), dtype=int)

    for tr, te in tscv.split(Xw):
        X_tr, X_te = Xw[tr], Xw[te]
        y_tr, y_te = y[tr], y[te]

        # --- guard: LR can't train on a single class ---
        if len(np.unique(y_tr)) < 2:
            # return None to signal "invalid split for this y"
            return None

        sc = StandardScaler()
        Xtr = sc.fit_transform(X_tr)
        Xte = sc.transform(X_te)

        clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=C)
        clf.fit(Xtr, y_tr)

        y_pred = clf.predict(Xte)

        accs.append(accuracy_score(y_te, y_pred))

        # balanced_accuracy_score can warn if y_te has 1 class; handle explicitly
        if len(np.unique(y_te)) < 2:
            # skip this fold for bacc/f1 (or set to np.nan). Skipping is safer.
            continue

        baccs.append(balanced_accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="binary", zero_division=0))
        cm += confusion_matrix(y_te, y_pred, labels=[0, 1])

    # If all bacc folds got skipped (rare), return None
    if len(baccs) == 0:
        return None

    return float(np.mean(accs)), float(np.mean(baccs)), float(np.mean(f1s)), cm

def time_shift_null(
    Xw,
    y,
    n_splits: int,
    stride_minutes: int,
    min_shift_minutes: int,
    n_shifts: int = 200,
    C: float = 1.0,
    seed: int = 0,
    max_tries_factor: int = 50,
):
    rng = np.random.default_rng(seed)
    N = len(y)

    min_shift = int(np.ceil(min_shift_minutes / max(stride_minutes, 1)))
    if 2 * min_shift >= N:
        raise ValueError(f"min_shift too large for dataset: min_shift={min_shift} windows, N={N}")

    null_bacc = []
    tries = 0
    max_tries = n_shifts * max_tries_factor

    while len(null_bacc) < n_shifts and tries < max_tries:
        tries += 1
        k = int(rng.integers(min_shift, N - min_shift))
        y_shift = np.roll(y, k)

        out = eval_cv_binary_logreg(Xw, y_shift, n_splits=n_splits, C=C)
        if out is None:
            continue

        _, bacc_k, _, _ = out
        null_bacc.append(bacc_k)

    if len(null_bacc) < n_shifts:
        raise RuntimeError(
            f"Could only obtain {len(null_bacc)}/{n_shifts} valid null runs. "
            f"Try reducing n_splits or min_shift_minutes, or increase max_tries_factor."
        )

    return np.array(null_bacc, dtype=float), min_shift


def main(data=DEFAULT_DATA, labels=DEFAULT_LABELS, window=AROUSAL_WINDOWS, cv=DEFAULT_CV):
    df = load_sensor_df(hours=data.hours, rule=data.rule, analog_mode=data.analog_mode)
    d  = compute_env_quadrant(df, margin=labels.margin, smooth_minutes=labels.smooth_minutes)

    X_int = build_internal_features(d)
    Xw, y, idx = make_window_dataset_binary(
        X_int,
        d["a_label"],  # <-- arousal target
        window_minutes=window.window_minutes,
        stride_minutes=window.stride_minutes,
        add_trend=window.add_trend,
    )

    counts = np.bincount(y, minlength=2)
    maj = counts.max() / counts.sum()
    print("GLOBAL LABEL DISTRIBUTIONS")
    print(f"Arousal dist (A-, A+): {counts}")
    print(f"Arousal majority baseline acc: {maj:.3f}")
    print(f"Samples: {len(y)}")

    tscv = TimeSeriesSplit(n_splits=cv.n_splits)

    accs, baccs, f1s = [], [], []
    cm = np.zeros((2, 2), dtype=int)

    for fold, (tr, te) in enumerate(tscv.split(Xw), 1):
        X_tr, X_te = Xw[tr], Xw[te]
        y_tr, y_te = y[tr], y[te]

        sc = StandardScaler()
        Xtr = sc.fit_transform(X_tr)
        Xte = sc.transform(X_te)

        clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, y_tr)

        y_pred = clf.predict(Xte)

        acc  = accuracy_score(y_te, y_pred)
        bacc = balanced_accuracy_score(y_te, y_pred)
        mf1  = f1_score(y_te, y_pred, average="binary", zero_division=0)

        accs.append(acc); baccs.append(bacc); f1s.append(mf1)
        cm += confusion_matrix(y_te, y_pred, labels=[0, 1])

        print(f"Fold {fold} | A acc={acc:.3f} bacc={bacc:.3f} F1={mf1:.3f}")

    real_acc  = float(np.mean(accs))
    real_bacc = float(np.mean(baccs))
    real_f1   = float(np.mean(f1s))
    real_cm   = cm

    print("\nREAL (unshifted) CV Summary:")
    print(f"Arousal: acc={real_acc:.3f}±{np.std(accs):.3f} | "
        f"bacc={real_bacc:.3f}±{np.std(baccs):.3f} | "
        f"F1={real_f1:.3f}±{np.std(f1s):.3f}")
    print("\nConfusion (overall):")
    print(real_cm)

    # --- TIME-SHIFT NULL TESTS ---
    for min_shift_minutes, seed in [(60, 0), (360, 1)]:
        null_bacc, min_shift_windows = time_shift_null(
            Xw=Xw,
            y=y,
            n_splits=cv.n_splits,
            stride_minutes=window.stride_minutes,
            min_shift_minutes=min_shift_minutes,
            n_shifts=200,
            C=1.0,
            seed=seed,
        )

        p = (np.sum(null_bacc >= real_bacc) + 1) / (len(null_bacc) + 1)

        print(f"\nTIME-SHIFT NULL (>= {min_shift_minutes} min):")
        print(f"min_shift_windows={min_shift_windows} (stride={window.stride_minutes} min)")
        print(f"Null bacc: mean={null_bacc.mean():.3f} ± {null_bacc.std():.3f}")
        print(f"Real bacc: {real_bacc:.3f}")
        print(f"p-value (null >= real): {p:.4f}")


if __name__ == "__main__":
    main()
