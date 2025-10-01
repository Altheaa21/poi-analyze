import pandas as pd

NEEDED = ["user_id","venue_id","latitude","longitude","ts"]

def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # 兼容列名（trail_id / timestamp -> ts）
    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp":"ts"})
    if "trail_id" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"trail_id":"ts"})

    keep = [c for c in NEEDED if c in df.columns]
    df = df[keep].dropna(subset=["latitude","longitude"]).copy()

    # 统一为 tz-aware（UTC）
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    df = df.sort_values(["user_id","ts"]).drop_duplicates(subset=["user_id","ts","venue_id"])
    return df

def apply_bbox(df: pd.DataFrame, bbox=None) -> pd.DataFrame:
    if not bbox: return df
    lat1, lat2 = bbox["lat"]; lon1, lon2 = bbox["lon"]
    m = (df["latitude"].between(lat1, lat2)) & (df["longitude"].between(lon1, lon2))
    return df.loc[m].copy()