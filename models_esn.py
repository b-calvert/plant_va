import numpy as np
from sklearn.linear_model import Ridge

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