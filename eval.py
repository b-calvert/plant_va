import numpy as np

# Used for optional near-transition analyses; not used in the current quadrant assessment
def transition_centres(y: np.ndarray) -> np.ndarray:
    """
    Identify time indices where the class label changes vs the previous step.

    Used for "near-transition" evaluation (performance around change points),
    because those are the hardest and most informative moments.
    """
    return np.where(y[1:] != y[:-1])[0] + 1

# Like trans_centres i stopped used for optional near-transition analyses
def near_transition_mask(n: int, centres: np.ndarray, radius: int) -> np.ndarray:
    """
    Build a boolean mask of length n that is True within +/- radius steps of any
    transition centre.

    Used for:
      - weighting training samples near transitions (optional)
      - evaluating metrics on near-transition region only (optional)
    """
    m = np.zeros(n, dtype=bool)
    for c in centres:
        a = max(0, c - radius)
        b = min(n, c + radius + 1)
        m[a:b] = True
    return m


def persist_baseline(y: np.ndarray) -> np.ndarray:
    """
    Simple persistence baseline: predict that the next label equals the previous.

    This is a strong baseline when labels are slow-changing. Any model should
    beat this to be meaningful.

    Used for sanity-checking: if model ~= persistence, you may just be learning
    inertia rather than internal signal-to-label mapping.
    """
    yp = y.copy()
    if len(yp) > 1:
        yp[1:] = y[:-1]
    return yp