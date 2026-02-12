import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from plant_va.data import load_sensor_df
from plant_va.labels import compute_env_quadrant
from plant_va.datasets import make_sequence_dataset
from plant_va.models_esn import esn_states, logreg_binary_readout_fit_predict_tuned, logreg_multiclass_readout_fit_predict
from plant_va.plant_va_config.presets import DEFAULT_DATA, DEFAULT_LABELS, DEFAULT_ESN


def main(data=DEFAULT_DATA, labels=DEFAULT_LABELS, esn=DEFAULT_ESN):
    df = load_sensor_df(hours=data.hours, rule=data.rule, analog_mode=data.analog_mode)
    d = compute_env_quadrant(df, margin=labels.margin, smooth_minutes=labels.smooth_minutes)

    Uv, v, idx, feat_names = make_sequence_dataset(d, label_col="v_label")
    Ua, a, _, _ = make_sequence_dataset(d, label_col="a_label")

    assert len(Uv) == len(Ua) == len(v) == len(a)
    # If this fails, Uv/Ua aren't identical and we should not share the scaler/reservoir this way.
    assert Uv.shape == Ua.shape and np.allclose(Uv, Ua), "Uv and Ua differ; shared reservoir may not be valid."

    q_true = (2 * v + a).astype(int)

    print("GLOBAL LABEL DISTRIBUTIONS (sequence-level)")
    print("Quadrant dist:", np.bincount(q_true, minlength=4))
    print("Valence dist (V-, V+):", np.bincount(v, minlength=2))
    print("Arousal dist (A-, A+):", np.bincount(a, minlength=2))
    print(f"Samples: {len(q_true)} | Features: {Uv.shape[1]}")

    tscv = TimeSeriesSplit(n_splits=esn.n_splits)

    # Separate tracking for composed vs direct quadrant decoding
    cm_comp = np.zeros((4, 4), dtype=int)
    cm_direct = np.zeros((4, 4), dtype=int)

    accs_comp, baccs_comp, f1s_comp = [], [], []
    accs_direct, baccs_direct, f1s_direct = [], [], []

    for fold, (tr, te) in enumerate(tscv.split(Uv), 1):
        Uv_tr, Uv_te = Uv[tr], Uv[te]
        v_tr, v_te = v[tr], v[te]
        a_tr, a_te = a[tr], a[te]
        q_te = q_true[te]

        # -----------------------------
        # Shared scaler + shared ESN
        # -----------------------------
        sc = StandardScaler()
        U_tr_s = sc.fit_transform(Uv_tr)
        U_te_s = sc.transform(Uv_te)

        X_tr = esn_states(
            U_tr_s,
            n_reservoir=esn.n_reservoir,
            spectral_radius=esn.spectral_radius,
            leak=esn.leak,
            input_scale=esn.input_scale,
            ridge_washout=esn.washout,
            seed=esn.seed + 10 * fold,
        )
        X_te = esn_states(
            U_te_s,
            n_reservoir=esn.n_reservoir,
            spectral_radius=esn.spectral_radius,
            leak=esn.leak,
            input_scale=esn.input_scale,
            ridge_washout=esn.washout,
            seed=esn.seed + 10 * fold,
        )

        # -----------------------------
        # Binary V/A readouts (tuned thresholds)
        # -----------------------------
        v_pred, pv, tV = logreg_binary_readout_fit_predict_tuned(
            X_tr, v_tr, X_te, C=1.0, washout=esn.washout
        )
        a_pred, pa, tA = logreg_binary_readout_fit_predict_tuned(
            X_tr, a_tr, X_te, C=1.0, washout=esn.washout
        )

        # Composed quadrant (from V/A)
        q_pred_comp = (2 * v_pred + a_pred).astype(int)

        # -----------------------------
        # Direct 4-class readout
        # -----------------------------
        q_pred_direct, pq = logreg_multiclass_readout_fit_predict(
            X_tr, q_true[tr], X_te, C=1.0, washout=esn.washout
        )

        # -----------------------------
        # Scoring (optionally drop test washout)
        # -----------------------------
        sl = slice(esn.washout, None) if len(q_te) > esn.washout else slice(None)

        # composed
        acc_c = accuracy_score(q_te[sl], q_pred_comp[sl])
        bacc_c = balanced_accuracy_score(q_te[sl], q_pred_comp[sl])
        f1_c = f1_score(q_te[sl], q_pred_comp[sl], average="macro", zero_division=0)

        accs_comp.append(acc_c)
        baccs_comp.append(bacc_c)
        f1s_comp.append(f1_c)
        cm_comp += confusion_matrix(q_te[sl], q_pred_comp[sl], labels=[0, 1, 2, 3])

        # direct
        acc_d = accuracy_score(q_te[sl], q_pred_direct[sl])
        bacc_d = balanced_accuracy_score(q_te[sl], q_pred_direct[sl])
        f1_d = f1_score(q_te[sl], q_pred_direct[sl], average="macro", zero_division=0)

        accs_direct.append(acc_d)
        baccs_direct.append(bacc_d)
        f1s_direct.append(f1_d)
        cm_direct += confusion_matrix(q_te[sl], q_pred_direct[sl], labels=[0, 1, 2, 3])

        print(f"Fold {fold} thresholds: tV={tV:.2f} tA={tA:.2f}")
        print(
            f"Fold {fold} | "
            f"V bacc={balanced_accuracy_score(v_te[sl], v_pred[sl]):.3f} F1={f1_score(v_te[sl], v_pred[sl], average='binary', zero_division=0):.3f} | "
            f"A bacc={balanced_accuracy_score(a_te[sl], a_pred[sl]):.3f} F1={f1_score(a_te[sl], a_pred[sl], average='binary', zero_division=0):.3f} | "
            f"Q(comp) bacc={bacc_c:.3f} F1={f1_c:.3f} | "
            f"Q(direct) bacc={bacc_d:.3f} F1={f1_d:.3f}"
        )

    # -----------------------------
    # Summaries
    # -----------------------------
    print("\nSummary (composed):")
    print(
        f"Quadrant: acc={np.mean(accs_comp):.3f}±{np.std(accs_comp):.3f} | "
        f"bacc={np.mean(baccs_comp):.3f}±{np.std(baccs_comp):.3f} | "
        f"F1={np.mean(f1s_comp):.3f}±{np.std(f1s_comp):.3f}"
    )
    print("\nConfusion (composed):")
    print(cm_comp)

    print("\nSummary (direct multiclass):")
    print(
        f"Quadrant: acc={np.mean(accs_direct):.3f}±{np.std(accs_direct):.3f} | "
        f"bacc={np.mean(baccs_direct):.3f}±{np.std(baccs_direct):.3f} | "
        f"F1={np.mean(f1s_direct):.3f}±{np.std(f1s_direct):.3f}"
    )
    print("\nConfusion (direct):")
    print(cm_direct)


if __name__ == "__main__":
    main()
