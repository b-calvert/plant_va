import pandas as pd
from readinfluxnew import PlantDataModel

def load_sensor_df(hours: str = "-1100h", rule: str = "5s", analog_mode: str = "features") -> pd.DataFrame:
    """
    Load sensor data via PlantDataModel and return a DataFrame indexed by time.
    Ensures DatetimeIndex, sorted ascending.
    """
    plant_model = PlantDataModel(hours=hours, rule=rule, analog_mode=analog_mode)
    df = plant_model.read_sensor_data().copy()

    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"])
        df = df.sort_values("_time").set_index("_time")
    else:
        df = df.sort_index()

    return df

def load_sensor_df_chunked(
    start: str,
    stop: str,
    rule: str = "5s",
    analog_mode: str = "features",
    chunk: str = "3h",
) -> pd.DataFrame:
    """
    Load data in chunks [t, t+chunk] to avoid Influx timeouts.
    start/stop are ISO strings.
    """
    t0 = pd.to_datetime(start, utc=True)
    t1 = pd.to_datetime(stop, utc=True)
    step = pd.Timedelta(chunk)

    parts = []
    t = t0
    while t < t1:
        t_next = min(t + step, t1)
        plant_model = PlantDataModel(
            start=t.isoformat(),
            stop=t_next.isoformat(),
            rule=rule,
            analog_mode=analog_mode,
        )
        df = plant_model.read_sensor_data()
        if len(df):
            parts.append(df)
        t = t_next

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out