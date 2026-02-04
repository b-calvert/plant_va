import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from plant_va.data import load_sensor_df
from plant_va.labels import compute_env_quadrant
from plant_va.datasets import make_cnn_sequence_windows
from plant_va.models_cnn import CNN1DBinary
from plant_va.plant_va_config.presets import (
    DEFAULT_DATA,
    DEFAULT_LABELS,
    VALENCE_WINDOW,
    AROUSAL_WINDOWS,
    DEFAULT_CV,
)


def standardise_channels_train_only(X_tr: np.ndarray, X_te: np.ndarray):
    """
    Per-channel standardisation using TRAIN statistics only.
    X: [N, C, T]
    """
    mu = X_tr.mean(axis=(0, 2), keepdims=True)
    sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_tr - mu) / sd, (X_te - mu) / sd


def predict_proba(model: nn.Module, loader: DataLoader, device="cpu") -> np.ndarray:
    """
    Return P(y=1) for a binary CNN using sigmoid(logits).
    """
    model.eval()
    ps = []
    with torch.no_grad():
        for xb, _yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            ps.append(p)
    return np.concatenate(ps).reshape(-1)


def train_one_fold_return_proba(
    model: nn.Module,
    train_loader: DataLoader,
    infer_loader: DataLoader,
    epochs=10,
    lr=1e-3,
    device="cpu",
) -> np.ndarray:
    """
    Train on train_loader, then return probabilities on infer_loader.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.to(device)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    return predict_proba(model, infer_loader, device=device)


def fuse_quadrant_probs(pv: np.ndarray, pa: np.ndarray) -> np.ndarray:
    """
    Combine P(V+) and P(A+) into 4-class quadrant probabilities:

      q0: V-, A- = (1-pv)(1-pa)
      q1: V-, A+ = (1-pv)pa
      q2: V+, A- = pv(1-pa)
      q3: V+, A+ = pv*pa
    """
    pv = pv.reshape(-1)
    pa = pa.reshape(-1)
    return np.column_stack([
        (1 - pv) * (1 - pa),
        (1 - pv) * pa,
        pv * (1 - pa),
        pv * pa,
    ])


def main(
    data=DEFAULT_DATA,
    labels=DEFAULT_LABELS,
    v_window=VALENCE_WINDOW,
    a_window=AROUSAL_WINDOWS,
    cv=DEFAULT_CV,
    sample_seconds=5,   # must match your Influx downsample rule
    epochs=10,
    lr=1e-3,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load + label data
    df = load_sensor_df(hours=data.hours, rule=data.rule, analog_mode=data.analog_mode)
    d = compute_env_quadrant(df, margin=labels.margin, smooth_minutes=labels.smooth_minutes)

    # 2) Build CNN windows for V and A (can be different window lengths)
    Xv, v, idxV, chansV = make_cnn_sequence_windows(
        d,
        label_col="v_label",
        window_minutes=v_window.window_minutes,
        stride_minutes=v_window.stride_minutes,
        sample_seconds=sample_seconds,
    )

    Xa, a, idxA, chansA = make_cnn_sequence_windows(
        d,
        label_col="a_label",
        window_minutes=a_window.window_minutes,
        stride_minutes=a_window.stride_minutes,
        sample_seconds=sample_seconds,
    )

    # Ensure types (torch likes float32)
    Xv = Xv.astype(np.float32)
    Xa = Xa.astype(np.float32)
    v = v.astype(int)
    a = a.astype(int)

    print("GLOBAL LABEL DISTRIBUTIONS (windowed)")
    print("Valence dist (V-, V+):", np.bincount(v, minlength=2), "| N:", len(v))
    print("Arousal dist (A-, A+):", np.bincount(a, minlength=2), "| N:", len(a))
    print(f"Valence windows: C={Xv.shape[1]} T={Xv.shape[2]} stride={v_window.stride_minutes}min window={v_window.window_minutes}min")
    print(f"Arousal windows: C={Xa.shape[1]} T={Xa.shape[2]} stride={a_window.stride_minutes}min window={a_window.window_minutes}min")

    # 3) TimeSeriesSplit anchored on VALENCE windows (the evaluation timeline)
    tscv = TimeSeriesSplit(n_splits=cv.n_splits)

    accs, baccs, f1s = [], [], []
    cm_q = np.zeros((4, 4), dtype=int)

    idxV = pd.DatetimeIndex(idxV)
    idxA = pd.DatetimeIndex(idxA)

    for fold, (trV, teV) in enumerate(tscv.split(Xv), 1):
        # ----- fold time boundaries (based on valence windows) -----
        t_train_end = idxV[trV][-1]
        t_test_start = idxV[teV][0]
        t_test_end = idxV[teV][-1]

        # ----- select arousal windows relevant to this fold by TIME -----
        # Train arousal model only on arousal windows up to train end.
        trA = np.where(idxA <= t_train_end)[0]

        # For inference, we want arousal predictions from (train period + test period) up to test end
        # so that ffill onto idxV_test has prior values available.
        infA = np.where(idxA <= t_test_end)[0]

        # If a fold has too few arousal windows (can happen with sparse data), skip it safely.
        if len(trA) < 10 or len(infA) < 10:
            print(f"Fold {fold} | skipped (insufficient arousal windows: trA={len(trA)}, infA={len(infA)})")
            continue

        # ----- valence train/test -----
        Xv_tr, Xv_te = Xv[trV], Xv[teV]
        v_tr, v_te = v[trV], v[teV]
        idxV_te = idxV[teV]

        Xv_tr, Xv_te = standardise_channels_train_only(Xv_tr, Xv_te)

        trainV_ds = TensorDataset(torch.from_numpy(Xv_tr), torch.from_numpy(v_tr))
        testV_ds  = TensorDataset(torch.from_numpy(Xv_te), torch.from_numpy(v_te))

        trainV_loader = DataLoader(trainV_ds, batch_size=64, shuffle=False)
        testV_loader  = DataLoader(testV_ds, batch_size=256, shuffle=False)

        modelV = CNN1DBinary(n_channels=Xv.shape[1], hidden=32, dropout=0.2)
        pv_te = train_one_fold_return_proba(
            modelV, trainV_loader, testV_loader, epochs=epochs, lr=lr, device=device
        )  # P(V+) at idxV_te

        # ----- arousal train + inference (up to test end) -----
        Xa_tr = Xa[trA]
        a_tr = a[trA]
        Xa_inf = Xa[infA]
        a_inf = a[infA]  # not used for training, but kept in dataset for loader interface
        idxA_inf = idxA[infA]

        Xa_tr, Xa_inf = standardise_channels_train_only(Xa_tr, Xa_inf)

        trainA_ds = TensorDataset(torch.from_numpy(Xa_tr), torch.from_numpy(a_tr))
        infA_ds   = TensorDataset(torch.from_numpy(Xa_inf), torch.from_numpy(a_inf))

        trainA_loader = DataLoader(trainA_ds, batch_size=64, shuffle=False)
        infA_loader   = DataLoader(infA_ds, batch_size=256, shuffle=False)

        modelA = CNN1DBinary(n_channels=Xa.shape[1], hidden=32, dropout=0.2)
        pa_inf = train_one_fold_return_proba(
            modelA, trainA_loader, infA_loader, epochs=epochs, lr=lr, device=device
        )  # P(A+) at idxA_inf (train+test timeline)

        # ----- align arousal probabilities onto valence test endpoints -----
        pa_series = pd.Series(pa_inf, index=idxA_inf).sort_index()

        # For each valence test time, take the most recent arousal prob at or before it.
        pa_on_Vte = pa_series.reindex(idxV_te, method="ffill")

        # If earliest valence test time is before any arousal pred exists, drop those rows.
        keep = pa_on_Vte.notna().to_numpy()
        if not np.all(keep):
            pv_te = pv_te[keep]
            v_te = v_te[keep]
            idxV_te = idxV_te[keep]
            pa_on_Vte = pa_on_Vte.iloc[keep]

        pa_te = pa_on_Vte.to_numpy().astype(float)

        # True arousal at valence endpoints (from environment-defined labels)
        a_true_on_Vte = d["a_label"].reindex(idxV_te, method="ffill").to_numpy().astype(int)

        # True quadrant and predicted quadrant
        q_true = (2 * v_te + a_true_on_Vte).astype(int)

        q_probs = fuse_quadrant_probs(pv_te, pa_te)
        q_pred = np.argmax(q_probs, axis=1).astype(int)

        # Metrics
        acc = accuracy_score(q_true, q_pred)
        bacc = balanced_accuracy_score(q_true, q_pred)
        mf1 = f1_score(q_true, q_pred, average="macro", zero_division=0)

        accs.append(acc); baccs.append(bacc); f1s.append(mf1)
        cm_q += confusion_matrix(q_true, q_pred, labels=[0, 1, 2, 3])

        print(
            f"Fold {fold} | Q acc={acc:.3f} bacc={bacc:.3f} F1={mf1:.3f} | "
            f"V_test_n={len(q_true)} A_inf_n={len(infA)}"
        )

    if len(accs) == 0:
        print("No folds produced results (likely insufficient aligned data).")
        return

    print("\nSummary:")
    print(
        f"CNN Quadrant (two-window fusion): acc={np.mean(accs):.3f}±{np.std(accs):.3f} | "
        f"bacc={np.mean(baccs):.3f}±{np.std(baccs):.3f} | "
        f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}"
    )
    print("\nConfusion (overall):")
    print(cm_q)


if __name__ == "__main__":
    main()
