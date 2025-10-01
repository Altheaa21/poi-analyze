import math, numpy as np, pandas as pd

# ---- q：UTC 小时桶，避开 DST 歧义 ----
def compute_q(traj: pd.DataFrame, freq="1h") -> pd.DataFrame:
    df = traj.copy()
    df["ts_utc_hour"] = df["ts"].dt.floor(freq)
    g = df.groupby("user_id", observed=True)
    obs_hours = g["ts_utc_hour"].nunique().rename("obs_hours")
    span_hours = ((g["ts"].max() - g["ts"].min()).dt.total_seconds()/3600.0 + 1).rename("total_hours")
    q = 1 - (obs_hours / span_hours.replace(0, np.nan))
    out = pd.concat([obs_hours, span_hours, q.rename("q")], axis=1).reset_index()
    return out

# ---- 熵 S ----
def shannon_entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(-(p*np.log2(p)).sum())

def compute_entropy(traj_valid: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for uid, g in traj_valid.groupby("user_id", observed=True):
        vc = g["venue_id"].value_counts()
        N = int(len(vc))
        S_unc = shannon_entropy(vc.values) if N>1 else 0.0
        S_rand = math.log2(N) if N>1 else 0.0
        rows.append({"user_id":uid,"N":N,"S_unc":S_unc,"S_rand":S_rand})
    return pd.DataFrame(rows)

# ---- Π_max：基于 Fano，不求解析，用二分 ----
def _fano_rhs(p, S, N):
    if p<=0 or p>=1 or N<=1: return float("inf")
    H = -(p*math.log2(p) + (1-p)*math.log2(1-p))
    return H + (1-p)*math.log2(N-1) - S

def solve_pimax(S: float, N: int, tol=1e-6, it=100) -> float:
    if N<=1 or S<=0: return 1.0
    if S>=math.log2(N): return 1.0/N
    lo, hi = 0.0, 1.0
    for _ in range(it):
        mid = (lo+hi)/2
        v = _fano_rhs(mid, S, N)
        if abs(v) < tol: return mid
        if v > 0: hi = mid
        else: lo = mid
    return mid

# ---- Rg ----
def haversine_km(lat1,lon1,lat2,lon2):
    R=6371.0088
    lat1=np.deg2rad(lat1); lon1=np.deg2rad(lon1)
    lat2=np.deg2rad(lat2); lon2=np.deg2rad(lon2)
    dlat=lat2-lat1; dlon=lon2-lon1
    a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arcsin(np.sqrt(a))

def compute_rg(traj_valid: pd.DataFrame) -> pd.DataFrame:
    def _rg(g):
        lat_cm=g["latitude"].mean(); lon_cm=g["longitude"].mean()
        d=haversine_km(g["latitude"].to_numpy(), g["longitude"].to_numpy(), lat_cm, lon_cm)
        rg=float(np.sqrt(np.mean(d**2))) if len(d)>=2 else 0.0
        return pd.Series({"n_points":len(g),"Rg_km":rg})
    return (traj_valid.groupby("user_id", observed=True, sort=False)
            .apply(_rg, include_groups=False).reset_index())

# ---- R / R_rand：本地时区 168 小时桶 ----
def compute_R(traj_local: pd.DataFrame) -> pd.DataFrame:
    df = traj_local.copy()  # 需包含 user_id, venue_id, week_hour
    cnt=(df.groupby(["user_id","week_hour","venue_id"], observed=True)
           .size().rename("n").reset_index())
    idx=cnt.groupby(["user_id","week_hour"], observed=True)["n"].idxmax()
    rules=cnt.loc[idx,["user_id","week_hour","venue_id"]].rename(columns={"venue_id":"pred"})
    df=df.merge(rules,on=["user_id","week_hour"],how="left")
    df["hit"]=(df["venue_id"]==df["pred"]).astype(int)
    R=df.groupby("user_id", observed=True)["hit"].mean().rename("R").reset_index()
    N=df.groupby("user_id", observed=True)["venue_id"].nunique().rename("N").reset_index()
    R=R.merge(N, on="user_id", how="left")
    R["R_rand"]=1.0/R["N"].replace(0, np.nan)
    return R