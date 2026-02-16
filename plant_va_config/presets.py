from .base import DataConfig, LabelConfig, WindowConfig, CVConfig, ESNConfig

DEFAULT_DATA = DataConfig(hours="-1000h", rule="5s", analog_mode="features")
DEFAULT_LABELS = LabelConfig(margin=0.1, smooth_minutes=5)

VALENCE_WINDOW = WindowConfig(window_minutes=20, stride_minutes=15, lag_minutes=0, add_trend=True)
AROUSAL_WINDOWS = WindowConfig(window_minutes=1, stride_minutes=1, lag_minutes=0, add_trend=True)


DEFAULT_CV = CVConfig(n_splits=5)
DEFAULT_ESN = ESNConfig(n_splits=5, n_reservoir=300, spectral_radius=0.85, leak=0.2,
                        input_scale=0.5, ridge_alpha=1.0, washout=50, seed=0)
