import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from plant_va.data import load_sensor_df
from plant_va.labels import compute_env_quadrant
from plant_va.features import build_internal_features
from plant_va.datasets import make_window_dataset_binary
from plant_va.plant_va_config.presets import DEFAULT_DATA, DEFAULT_LABELS, VALENCE_WINDOW, DEFAULT_CV


def main(data=DEFAULT_DATA, labels=DEFAULT_LABELS, window=VALENCE_WINDOW, cv=DEFAULT_CV):
    df = load_sensor_df(hours=data.hours, rule=data.rule, analog_mode=data.analog_mode)

    print("Raw df rows:", len(df))
    print("Time span:", df.index.min(), "->", df.index.max())
    print("Columns:", df.columns.tolist())
    print("IR non-null:", df["ir"].notna().sum() if "ir" in df.columns else "NO IR COL")
    print("IR > 2:", (df["ir"] > 2).sum() if "ir" in df.columns else "NO IR COL")

    d  = compute_env_quadrant(df, margin=labels.margin, smooth_minutes=labels.smooth_minutes)
    

    X_int = build_internal_features(d)

    Xw, y, idx = make_window_dataset_binary(
        X_int,
        d["v_label"],
        window_minutes=window.window_minutes,
        stride_minutes=window.stride_minutes,
        add_trend=window.add_trend,
    )

    counts = np.bincount(y, minlength=2)
    maj = counts.max() / counts.sum()
    print("GLOBAL LABEL DISTRIBUTIONS")
    print(f"Valence dist (V-, V+): {counts}")
    print(f"Valence majority baseline acc: {maj:.3f}")
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

        acc = accuracy_score(y_te, y_pred)
        bacc = balanced_accuracy_score(y_te, y_pred)
        mf1 = f1_score(y_te, y_pred, average="binary", zero_division=0)

        accs.append(acc); baccs.append(bacc); f1s.append(mf1)
        cm += confusion_matrix(y_te, y_pred, labels=[0, 1])

        print(f"Fold {fold} | V acc={acc:.3f} bacc={bacc:.3f} F1={mf1:.3f}")

    print("\nSummary:")
    print(
        f"Valence: acc={np.mean(accs):.3f}±{np.std(accs):.3f} | "
        f"bacc={np.mean(baccs):.3f}±{np.std(baccs):.3f} | "
        f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}"
    )
    print("\nConfusion (overall):")
    print(cm)


if __name__ == "__main__":
    main()
