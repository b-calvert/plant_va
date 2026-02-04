from dataclasses import dataclass

@dataclass(frozen=True)
class DataConfig:
    hours: str = "-300h"
    rule: str = "5s"
    analog_mode: str = "features"

@dataclass(frozen=True)
class LabelConfig:
    margin: float = 0.1
    smooth_minutes: int = 5

@dataclass(frozen=True)
class WindowConfig:
    window_minutes: int = 20
    stride_minutes: int = 15
    lag_minutes: int = 0
    add_trend: bool = True

@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5

@dataclass(frozen=True)
class ESNConfig:
    n_splits: int = 3
    n_reservoir: int = 300
    spectral_radius: float = 0.9
    leak: float = 0.3
    input_scale: float = 0.5
    ridge_alpha: float = 1.0
    washout: int = 50
    seed: int = 0
