import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_va_trajectory(Vw, Aw, title="V/A trajectory (windowed, coloured by time)"):
    n = min(len(Vw), len(Aw))
    Vw = Vw[:n]; Aw = Aw[:n]

    plt.figure(figsize=(7, 7))
    plt.scatter(Vw, Aw, c=np.arange(n), cmap="viridis", s=40, alpha=0.75)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Valence (environment-defined)")
    plt.ylabel("Arousal (environment-defined)")
    plt.title(title)
    plt.colorbar(label="Window index")
    plt.tight_layout()
    plt.show()

def plot_internal_pca(Xw, yw, title="Internal-state trajectory (windowed), coloured by quadrant"):
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(Xw)

    plt.figure(figsize=(7, 7))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=yw, cmap="tab10", s=40, alpha=0.75)
    plt.xlabel("Internal PC1")
    plt.ylabel("Internal PC2")
    plt.title(title)
    plt.colorbar(sc, label="Quadrant")
    plt.tight_layout()
    plt.show()
