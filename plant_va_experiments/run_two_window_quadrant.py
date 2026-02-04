import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from plant_va.data import load_sensor_df
from plant_va.labels import compute_env_quadrant
from plant_va.features import build_internal_features
from plant_va.datasets import make_window_dataset_binary
from plant_va.plant_va_config.presets import DEFAULT_DATA, DEFAULT_LABELS, VALENCE_WINDOW, AROUSAL_WINDOW_FAST, DEFAULT_CV


def main(
    data=DEFAULT_DATA,
    labels=DEFAULT_LABELS,
    windowV=VALENCE_WINDOW,
    windowA=AROUSAL_WINDOW_FAST,
    cv=DEFAULT_CV
):
    df = load_sensor_df(hours=data.hours, rule=data.rule, analog_mode=data.analog_mode)

    # Note: both labels come from the same definition, but you can smooth differently if you want later.
    dV = compute_env_quadrant(df, margin=labels.margin, smooth_minutes=labels.smooth_minutes)
    dA = compute_env_quadrant(df, margin=labels.margin, smooth_minutes=labels.smooth_minutes)

    XV_int = build_internal_features(dV)
    XA_int = build_internal_features(dA)

    # Build separate window datasets
    XV, v, idxV = make_window_dataset_binary(
        XV_int, dV["v_label"],
        window_minutes=windowV.window_minutes,
        stride_minutes=windowV.stride_minutes,
        add_trend=windowV.add_trend,
    )
    XA, a, idxA = make_window_dataset_binary(
        XA_int, dA["a_label"],
        window_minutes=windowA.window_minutes,
        stride_minutes=windowA.stride_minutes,
        add_trend=windowA.add_trend,
    )

    # Align arousal features/labels to valence endpoints
    XA_df = pd.DataFrame(XA, index=idxA)
    a_s = pd.Series(a, index=idxA)

    XA_on_V = XA_df.reindex(idxV, method="ffill").to_numpy()
    a_on_V = a_s.reindex(idxV, method="ffill").to_numpy().astype(int)

    q_true = (2 * v + a_on_V).astype(int)

    print("GLOBAL LABEL DISTRIBUTIONS (aligned on valence endpoints)")
    print("Quadrant dist:", np.bincount(q_true, minlength=4))
    print("Valence dist (V-, V+):", np.bincount(v, minlength=2))
    print("Arousal dist (A-, A+):", np.bincount(a_on_V, minlength=2))
    print("Samples:", len(q_true))

    tscv = TimeSeriesSplit(n_splits=cv.n_splits)

    cm_q = np.zeros((4, 4), dtype=int)
    accs, baccs, f1s = [], [], []

    for fold, (tr, te) in enumerate(tscv.split(XV), 1):
        XV_tr, XV_te = XV[tr], XV[te]
        v_tr, v_te = v[tr], v[te]

        XA_tr, XA_te = XA_on_V[tr], XA_on_V[te]
        a_tr, a_te = a_on_V[tr], a_on_V[te]

        q_te = q_true[te]

        scV = StandardScaler()
        scA = StandardScaler()
        XVtr = scV.fit_transform(XV_tr)
        XVte = scV.transform(XV_te)
        XAtr = scA.fit_transform(XA_tr)
        XAte = scA.transform(XA_te)

        clfV = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
        clfA = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)

        clfV.fit(XVtr, v_tr)
        clfA.fit(XAtr, a_tr)

        pv = clfV.predict_proba(XVte)[:, 1]
        pa = clfA.predict_proba(XAte)[:, 1]

        # independent combination assumption
        q_probs = np.column_stack([
            (1 - pv) * (1 - pa),
            (1 - pv) * pa,
            pv * (1 - pa),
            pv * pa,
        ])
        q_pred = np.argmax(q_probs, axis=1)

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
