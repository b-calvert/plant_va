import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# =============================================================================
# ESN CORE (sequence model: reservoir computing)
# =============================================================================

def _scale_to_spectral_radius(W: np.ndarray, target_rho: float, rng: np.random.Generator) -> np.ndarray:
    """
    Scale random recurrent weight matrix W so its spectral radius ~ target_rho.

    Spectral radius controls the "memory" / stability of the reservoir dynamics.
    Here we approximate spectral radius using power iteration (fast).
    """
    v = rng.normal(size=(W.shape[0],))
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(50):
        v = W @ v
        v /= (np.linalg.norm(v) + 1e-12)
    rho = np.linalg.norm(W @ v) / (np.linalg.norm(v) + 1e-12)
    if rho <= 0:
        return W
    return W * (target_rho / rho)


def esn_states(
    U: np.ndarray,
    n_reservoir: int = 300,
    spectral_radius: float = 0.9,
    leak: float = 0.3,
    input_scale: float = 0.5,
    ridge_washout: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute ESN reservoir states for an input sequence U: shape [T, D].

    Returns:
      X: [T, n_reservoir] reservoir states

    Notes:
      - This ESN is untrained; only the readout is trained.
      - 'leak' controls update speed (low leak = slower dynamics).
      - washout is not applied here; it's applied during readout fitting.
    """
    rng = np.random.default_rng(seed)
    T, D = U.shape

    # Dense random reservoir; you can sparsify later if needed
    W = rng.normal(0, 1.0, size=(n_reservoir, n_reservoir))
    W = _scale_to_spectral_radius(W, spectral_radius, rng)

    # Input weights scale how strongly inputs drive the reservoir
    Win = rng.normal(0, 1.0, size=(n_reservoir, D)) * input_scale

    X = np.zeros((T, n_reservoir), dtype=float)
    x = np.zeros((n_reservoir,), dtype=float)

    for t in range(T):
        pre = Win @ U[t] + W @ x
        x_new = np.tanh(pre)
        x = (1.0 - leak) * x + leak * x_new
        X[t] = x

    return X


def ridge_binary_readout_fit_predict(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    alpha: float = 1.0,
    washout: int = 50,
):
    """
    Train a ridge regression readout on reservoir states (binary y in {0,1}).

    Approach:
      - Fit ridge regression to predict real-valued score s
      - Convert to pseudo-probability with logistic sigmoid
      - Threshold at 0.5

    This is a common quick ESN readout. Not perfectly calibrated, but works for classification.
    """
    # Remove initial transient states in training (washout)
    if washout > 0 and len(Xtr) > washout:
        Xtr2 = Xtr[washout:]
        ytr2 = ytr[washout:]
    else:
        Xtr2, ytr2 = Xtr, ytr

    # Add bias term explicitly (since Ridge fit_intercept=False)
    Xtrb = np.hstack([Xtr2, np.ones((len(Xtr2), 1))])
    Xteb = np.hstack([Xte,  np.ones((len(Xte),  1))])

    rr = Ridge(alpha=alpha, fit_intercept=False)
    rr.fit(Xtrb, ytr2.astype(float))

    s = rr.predict(Xteb)               # real-valued score
    p = 1.0 / (1.0 + np.exp(-s))       # logistic squash
    yhat = (p >= 0.5).astype(int)

    return yhat, p

def _best_threshold(y_true, p, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)
    best_t, best = 0.5, -1.0
    for t in grid:
        yhat = (p >= t).astype(int)
        b = balanced_accuracy_score(y_true, yhat)
        if b > best:
            best, best_t = b, t
    return best_t, best

def logreg_binary_readout_fit_predict_tuned(
    Xtr, ytr, Xte, C=1.0, washout=50, class_weight="balanced"
):
    # washout on train
    if washout > 0 and len(Xtr) > washout:
        Xtr2, ytr2 = Xtr[washout:], ytr[washout:]
    else:
        Xtr2, ytr2 = Xtr, ytr

    # add bias
    Xtrb = np.hstack([Xtr2, np.ones((len(Xtr2), 1))])
    Xteb = np.hstack([Xte,  np.ones((len(Xte),  1))])

    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=3000,
    )
    clf.fit(Xtrb, ytr2.astype(int))

    # probabilities
    p_tr = clf.predict_proba(Xtrb)[:, 1]
    t_star, _ = _best_threshold(ytr2.astype(int), p_tr)

    p_te = clf.predict_proba(Xteb)[:, 1]
    yhat = (p_te >= t_star).astype(int)
    return yhat, p_te, t_star

# Multiclass logreg
def logreg_multiclass_readout_fit_predict(Xtr, ytr, Xte, C=1.0, washout=50, class_weight="balanced"):
    if washout > 0 and len(Xtr) > washout:
        Xtr2, ytr2 = Xtr[washout:], ytr[washout:]
    else:
        Xtr2, ytr2 = Xtr, ytr

    Xtrb = np.hstack([Xtr2, np.ones((len(Xtr2), 1))])
    Xteb = np.hstack([Xte,  np.ones((len(Xte),  1))])

    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=5000,
    )
    clf.fit(Xtrb, ytr2.astype(int))
    proba = clf.predict_proba(Xteb)
    yhat = np.argmax(proba, axis=1)
    return yhat, proba

