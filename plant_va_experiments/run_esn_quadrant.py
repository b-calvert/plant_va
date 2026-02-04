import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from plant_va.data import load_sensor_df
from plant_va.labels import compute_env_quadrant
from plant_va.datasets import make_sequence_dataset
from plant_va.models_esn import esn_states, ridge_binary_readout_fit_predict
from plant_va.plant_va_config.presets import DEFAULT_DATA, DEFAULT_LABELS, DEFAULT_ESN


def main(data=DEFAULT_DATA, labels=DEFAULT_LABELS, esn=DEFAULT_ESN):
    df = load_sensor_df(hours=data.hours, rule=data.rule, analog_mode=data.analog_mode)
    d = compute_env_quadrant(df, margin=labels.margin, smooth_minutes=labels.smooth_minutes)

    Uv, v, idx, feat_names = make_sequence_dataset(d, label_col="v_label")
    Ua, a, _, _ = make_sequence_dataset(d, label_col="a_label")

    assert len(Uv) == len(Ua) == len(v) == len(a)

    q_true = (2 * v + a).astype(int)

    print("GLOBAL LABEL DISTRIBUTIONS (sequence-level)")
    print("Quadrant dist:", np.bincount(q_true, minlength=4))
    print("Valence dist (V-, V+):", np.bincount(v, minlength=2))
    print("Arousal dist (A-, A+):", np.bincount(a, minlength=2))
    print(f"Samples: {len(q_true)} | Features: {Uv.shape[1]}")

    tscv = TimeSeriesSplit(n_splits=esn.n_splits)

    cm_q = np.zeros((4, 4), dtype=int)
    accs, baccs, f1s = [], [], []

    for fold, (tr, te) in enumerate(tscv.split(Uv), 1):
        Uv_tr, Uv_te = Uv[tr], Uv[te]
        Ua_tr, Ua_te = Ua[tr], Ua[te]
        v_tr, v_te = v[tr], v[te]
        a_tr, a_te = a[tr], a[te]
        q_te = q_true[te]

        # scale inputs using train only
        scV = StandardScaler()
        scA = StandardScaler()
        Uv_tr_s = scV.fit_transform(Uv_tr)
        Uv_te_s = scV.transform(Uv_te)
        Ua_tr_s = scA.fit_transform(Ua_tr)
        Ua_te_s = scA.transform(Ua_te)

        # reservoir states
        Xv_tr = esn_states(
            Uv_tr_s,
            n_reservoir=esn.n_reservoir,
            spectral_radius=esn.spectral_radius,
            leak=esn.leak,
            input_scale=esn.input_scale,
            ridge_washout=esn.washout,
            seed=esn.seed + 10 * fold,
        )
        Xv_te = esn_states(
            Uv_te_s,
            n_reservoir=esn.n_reservoir,
            spectral_radius=esn.spectral_radius,
            leak=esn.leak,
            input_scale=esn.input_scale,
            ridge_washout=esn.washout,
            seed=esn.seed + 10 * fold,
        )

        Xa_tr = esn_states(
            Ua_tr_s,
            n_reservoir=esn.n_reservoir,
            spectral_radius=esn.spectral_radius,
            leak=esn.leak,
            input_scale=esn.input_scale,
            ridge_washout=esn.washout,
            seed=esn.seed + 10 * fold + 1,
        )
        Xa_te = esn_states(
            Ua_te_s,
            n_reservoir=esn.n_reservoir,
            spectral_radius=esn.spectral_radius,
            leak=esn.leak,
            input_scale=esn.input_scale,
            ridge_washout=esn.washout,
            seed=esn.seed + 10 * fold + 1,
        )

        v_pred, pv = ridge_binary_readout_fit_predict(Xv_tr, v_tr, Xv_te, alpha=esn.ridge_alpha, washout=esn.washout)
        a_pred, pa = ridge_binary_readout_fit_predict(Xa_tr, a_tr, Xa_te, alpha=esn.ridge_alpha, washout=esn.washout)

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
