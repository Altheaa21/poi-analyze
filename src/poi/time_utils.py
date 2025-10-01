import pandas as pd

def to_local(df: pd.DataFrame, tz: str) -> pd.Series:
    return df["ts"].dt.tz_convert(tz)

def week_hour(ts_local: pd.Series) -> pd.Series:
    return ts_local.dt.dayofweek * 24 + ts_local.dt.hour  # 0..167 (Mon=0)