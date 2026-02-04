import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from plant_va.data import load_sensor_df
from plant_va.labels import compute_env_quadrant
from plant_va.features import build_internal_features
from plant_va.datasets import make_window_dataset
from plant_va.plant_va_config.presets import DEFAULT_DATA, DEFAULT_LABELS, VALENCE_WINDOW, DEFAULT_CV


def main(data=DEFAULT_DATA, labels=DEFAULT_LABELS, window=VALENCE_WINDOW, cv=DEFAULT_CV):
    df = load_sensor_df(hours=data.hours, rule=data.rule, analog_mode=data.analog_mode)
    d = compute_env_quadrant(df, margin=labels.margin, smooth_minutes=labels.smooth_minutes)

    X_int = build_internal_features(d)

    # Windowed 4-class dataset
    Xw, yq, idx = make_window_dataset(
        X_int, d["quadrant"],
        window_minutes=window.window_minutes,
        stride_minutes=window.stride_minutes,
        add_trend=window.add_trend,
    )

    print("GLOBAL LABEL DISTRIBUTIONS")
    print("Quadrant dist:", np.bincount(yq, minlength=4))
    print("Samples:", len(yq))

    tscv = TimeSeriesSplit(n_splits=cv.n_splits)

    cm_q = np.zeros((4, 4), dtype=int)
    accs, baccs, f1s = [], [], []

    for fold, (tr, te) in enumerate(tscv.split(Xw), 1):
        X_tr, X_te = Xw[tr], Xw[te]
        y_tr, y_te = yq[tr], yq[te]

        sc = StandardScaler()
        Xtr = sc.fit_transform(X_tr)
        Xte = sc.transform(X_te)

        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=1.0,
            multi_class="auto",
        )
        clf.fit(Xtr, y_tr)

        y_pred = clf.predict(Xte)

        acc = accuracy_score(y_te, y_pred)
        bacc = balanced_accuracy_score(y_te, y_pred)
        mf1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

        accs.append(acc); baccs.append(bacc); f1s.append(mf1)
        cm_q += confusion_matrix(y_te, y_pred, labels=[0, 1, 2, 3])

        print(f"Fold {fold} | Q acc={acc:.3f} bacc={bacc:.3f} F1={mf1:.3f}")

    print("\nSummary:")
    print(
        f"Quadrant: acc={np.mean(accs):.3f}±{np.std(accs):.3f} | "
        f"bacc={np.mean(baccs):.3f}±{np.std(baccs):.3f} | "
        f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}"
    )
    print("\nConfusion (overall):")
    print(cm_q)


if __name__ == "__main__":
    main()
