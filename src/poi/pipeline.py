import os, json, yaml, pandas as pd
from .io import load_parquet, apply_bbox
from .time_utils import to_local, week_hour
from .metrics import compute_q, compute_entropy, solve_pimax, compute_rg, compute_R
from .figs import plot_fig3_combined

def _summ(s: pd.Series):
    s = s.dropna()
    if s.empty: 
        return {}
    return dict(
        mean=float(s.mean()),
        median=float(s.median()),
        p25=float(s.quantile(0.25)),
        p75=float(s.quantile(0.75)),
        min=float(s.min()),
        max=float(s.max()),
        count=int(s.shape[0])
    )

def analyze_city(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    city = cfg["city"]; tz = cfg["tz"]; out_dir = cfg.get("out_dir","reports")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(f"{out_dir}/figures", exist_ok=True)

    # 1) 读数据 & 预处理
    df = load_parquet(cfg["data_path"])
    df = apply_bbox(df, cfg.get("bbox"))
    df["ts_local"] = to_local(df, tz)

    # 2) q & 过滤
    user_q = compute_q(df)
    points = df.groupby("user_id", observed=True).size().rename("n_points").reset_index()
    user_q = user_q.merge(points, on="user_id", how="left")
    valid_users = user_q[(user_q["q"] < cfg["q_threshold"]) & (user_q["n_points"] >= cfg["min_points"])]["user_id"]
    traj_valid = df[df["user_id"].isin(valid_users)].reset_index(drop=True)

    # 3) 熵 & Πmax
    user_entropy = compute_entropy(traj_valid)
    user_entropy["Pi_max"] = user_entropy.apply(lambda r: solve_pimax(r["S_unc"], int(r["N"])), axis=1)

    # 4) Rg
    user_rg = compute_rg(traj_valid)

    # 5) R（本地 168 小时）
    tl = traj_valid[["user_id","venue_id","ts_local"]].copy()
    tl["week_hour"] = week_hour(tl["ts_local"])
    user_R = compute_R(tl)

    # 6) 合并用户级矩阵
    user_metrics = (
        user_q[["user_id","q","n_points"]]
        .merge(user_entropy, on="user_id", how="inner")
        .merge(user_rg[["user_id","Rg_km","n_points"]], on="user_id", how="left", suffixes=("","_rg"))
        .merge(user_R[["user_id","R","R_rand","N"]], on="user_id", how="left")
    )

    # 7) 保存用户级 CSV
    cols = [c for c in ["user_id","n_points","q","N","S_unc","S_rand","Pi_max","Rg_km","R","R_rand"] if c in user_metrics.columns]
    user_metrics[cols].to_csv(f"{out_dir}/{city}_user_metrics.csv", index=False)

    # 8) 画三图合一（平滑）
    fig_path = plot_fig3_combined(
        traj_valid.assign(ts_local=traj_valid["ts_local"]),
        user_metrics, city,
        out_dir=f"{out_dir}/figures",
        min_active_users_per_hour=cfg.get("min_active_users_per_hour", 10),

        # 新增：把 YAML 里的平滑参数传进去
        smooth_method=cfg.get("smooth_method", "rolling"),   # "rolling" | "ema" | "none"
        rolling_hours_window=cfg.get("rolling_hours_window", 6),
        ema_alpha=cfg.get("ema_alpha", 0.25),
        bin_hours=cfg.get("bin_hours", 2),

        rg_bins=cfg.get("rg_bins", 12),
        rg_max_clip_km=cfg.get("rg_clip_km", 100.0),
        c_rolling_bins_window=cfg.get("c_rolling_bins_window", 2),
    )


    # 9) 城市级 summary
    summary = {
        "city": city,
        "tz": tz,
        "n_users_total": int(df["user_id"].nunique()),
        "n_users_valid": int(user_metrics["user_id"].nunique()),
        "q": _summ(user_metrics["q"]) if "q" in user_metrics else {},
        # "N": _summ(user_metrics["N"]) if "N" in user_metrics else {},
        "S_unc": _summ(user_metrics["S_unc"]) if "S_unc" in user_metrics else {},
        "S_rand": _summ(user_metrics["S_rand"]) if "S_rand" in user_metrics else {},
        "Pi_max": _summ(user_metrics["Pi_max"]) if "Pi_max" in user_metrics else {},
        "Rg_km": _summ(user_metrics["Rg_km"]) if "Rg_km" in user_metrics else {},
        "R": _summ(user_metrics["R"]) if "R" in user_metrics else {},
        "figure": fig_path,
    }
    with open(f"{out_dir}/{city}_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print(f"[saved] {out_dir}/{city}_user_metrics.csv")
    print(f"[saved] {out_dir}/{city}_summary.json")
    print(f"[figure] {fig_path}")
    return user_metrics, summary