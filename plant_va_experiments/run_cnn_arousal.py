import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from plant_va.data import load_sensor_df
from plant_va.labels import compute_env_quadrant
from plant_va.datasets import make_cnn_sequence_windows
from plant_va.models_cnn import CNN1DBinary
from plant_va.plant_va_config.presets import DEFAULT_DATA, DEFAULT_LABELS, VALENCE_WINDOW, DEFAULT_CV


def standardise_channels_train_only(X_tr, X_te):
    # X: [N, C, T]
    mu = X_tr.mean(axis=(0, 2), keepdims=True)
    sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_tr - mu) / sd, (X_te - mu) / sd


def train_one_fold(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cpu"):
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

    # eval
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            y_pred.append(pred)
            y_true.append(yb.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred


def main(data=DEFAULT_DATA, labels=DEFAULT_LABELS, window=VALENCE_WINDOW, cv=DEFAULT_CV):
    df = load_sensor_df(hours=data.hours, rule=data.rule, analog_mode=data.analog_mode)
    d = compute_env_quadrant(df, margin=labels.margin, smooth_minutes=labels.smooth_minutes)

    # CNN windows from internal signals, target = valence
    X, y, idx, chans = make_cnn_sequence_windows(
        d,
        label_col="a_label",   # <-- arousal
        window_minutes=window.window_minutes,
        stride_minutes=window.stride_minutes,
        sample_seconds=5,
    )


    print("GLOBAL LABEL DISTRIBUTIONS")
    print("Arousal dist (A-, A+):", np.bincount(y, minlength=2))
    print("Samples:", len(y), "| Channels:", len(chans), "| T:", X.shape[-1])

    tscv = TimeSeriesSplit(n_splits=cv.n_splits)

    accs, baccs, f1s = [], [], []
    cm = np.zeros((2, 2), dtype=int)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        X_tr, X_te = standardise_channels_train_only(X_tr, X_te)

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        test_ds  = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)  # keep time order
        test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

        model = CNN1DBinary(n_channels=X.shape[1], hidden=32, dropout=0.2)

        yt, yp = train_one_fold(model, train_loader, test_loader, epochs=10, lr=1e-3, device=device)

        acc = accuracy_score(yt, yp)
        bacc = balanced_accuracy_score(yt, yp)
        mf1 = f1_score(yt, yp, average="binary", zero_division=0)
        accs.append(acc); baccs.append(bacc); f1s.append(mf1)

        cm += confusion_matrix(yt, yp, labels=[0, 1])

        print(f"Fold {fold} | A acc={acc:.3f} bacc={bacc:.3f} F1={mf1:.3f}")

    print("\nSummary:")
    print(
        f"Arousal CNN: acc={np.mean(accs):.3f}±{np.std(accs):.3f} | "
        f"bacc={np.mean(baccs):.3f}±{np.std(baccs):.3f} | "
        f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}"
    )
    print("\nConfusion (overall):")
    print(cm)


if __name__ == "__main__":
    main()
