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
    # 1) load + label from environment
    df = load_sensor_df(hours=data.hours, rule=data.rule, analog_mode=data.analog_mode)
    d = compute_env_quadrant(df, margin=labels.margin, smooth_minutes=labels.smooth_minutes)

    # 2) internal-only signals
    X_int = build_internal_features(d)

    # 3) ONE windowed feature matrix used for both targets
    Xw, v, idx = make_window_dataset_binary(
        X_int, d["v_label"],
        window_minutes=window.window_minutes,
        stride_minutes=window.stride_minutes,
        add_trend=window.add_trend,
    )
    a = d["a_label"].reindex(idx, method="ffill").to_numpy().astype(int)

    q_true = (2 * v + a).astype(int)

    print("GLOBAL LABEL DISTRIBUTIONS")
    print("Quadrant dist:", np.bincount(q_true, minlength=4))
    print("Valence dist (V-, V+):", np.bincount(v, minlength=2))
    print("Arousal dist (A-, A+):", np.bincount(a, minlength=2))
    print("Samples:", len(q_true))

    tscv = TimeSeriesSplit(n_splits=cv.n_splits)

    cm_q = np.zeros((4, 4), dtype=int)
    accs, baccs, f1s = [], [], []

    for fold, (tr, te) in enumerate(tscv.split(Xw), 1):
        X_tr, X_te = Xw[tr], Xw[te]
        v_tr, v_te = v[tr], v[te]
        a_tr, a_te = a[tr], a[te]
        q_te = q_true[te]

        sc = StandardScaler()
        Xtr = sc.fit_transform(X_tr)
        Xte = sc.transform(X_te)

        clfV = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
        clfA = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)

        clfV.fit(Xtr, v_tr)
        clfA.fit(Xtr, a_tr)

        v_pred = clfV.predict(Xte)
        a_pred = clfA.predict(Xte)

        q_pred = (2 * v_pred + a_pred).astype(int)

        acc = accuracy_score(q_te, q_pred)
        bacc = balanced_accuracy_score(q_te, q_pred)
        mf1 = f1_score(q_te, q_pred, average="macro", zero_division=0)

        accs.append(acc); baccs.append(bacc); f1s.append(mf1)
        cm_q += confusion_matrix(q_te, q_pred, labels=[0, 1, 2, 3])

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
