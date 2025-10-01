import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

def _slug(s): return s.lower().replace(" ", "-")

def _apply_smooth(series: pd.Series, method: str, rolling_hours_window: int, ema_alpha: float):
    s = series.copy()
    if method == "rolling":
        return s.rolling(rolling_hours_window, center=True, min_periods=1).mean()
    elif method == "ema":
        return s.ewm(alpha=ema_alpha, adjust=False, min_periods=1).mean()
    else:
        return s

def plot_fig3_combined(
    traj_valid, user_metrics, city, out_dir="reports/figures",
    min_active_users_per_hour=10,
    smooth_method="rolling",          # "rolling" | "ema" | "none"
    rolling_hours_window=6,
    ema_alpha=0.25,
    bin_hours=1,                      # 1=小时，2=两小时合桶
    rg_bins=12, rg_max_clip_km=100.0, c_rolling_bins_window=2
):
    os.makedirs(out_dir, exist_ok=True)

    df = traj_valid.copy()
    # 0..167：本地周内小时
    df["week_hour"] = df["ts_local"].dt.dayofweek * 24 + df["ts_local"].dt.hour

    # 可选合桶：把小时映射到更粗的时间桶（仅用于画图，不影响统计）
    if bin_hours > 1:
        df["time_key"] = (df["week_hour"] // bin_hours) * bin_hours
        time_index = list(range(0, 168, bin_hours))
    else:
        df["time_key"] = df["week_hour"]
        time_index = list(range(168))

    # ========== A) R(t)：命中率 ==========
    # 1) 统计每个用户在每个 time_key 的各地点次数
    cnt = (df.groupby(["user_id", "time_key", "venue_id"], observed=True)
             .size().rename("n").reset_index())
    # 2) 对每个 (user_id, time_key) 取次数最多的地点（预测规则）
    idx = cnt.groupby(["user_id", "time_key"], observed=True)["n"].idxmax()
    rules = cnt.loc[idx, ["user_id", "time_key", "venue_id"]].rename(columns={"venue_id": "pred"})

    # 3) 把规则并回原数据，与实际观测比较是否命中
    a_df = df.merge(rules, on=["user_id", "time_key"], how="left")
    a_df["hit"] = (a_df["venue_id"] == a_df["pred"]).astype(int)

    # 4) 聚合为每个时间桶的命中率
    aggA = (a_df.groupby("time_key", observed=True)
                .agg(hits=("hit", "sum"), total=("hit", "size"))
                .reindex(time_index).fillna(0))
    Rt_raw = pd.Series(
        np.where(aggA["total"] > 0, aggA["hits"] / aggA["total"], np.nan),
        index=time_index
    )
    # 5) 先平滑，再小范围插值补洞，最后再用样本阈值掩蔽稀疏时段
    Rt_s = _apply_smooth(Rt_raw, smooth_method, rolling_hours_window, ema_alpha)
    Rt_s = Rt_s.interpolate(limit=2, limit_direction="both")
    Rt_s = Rt_s.mask(aggA["total"] < min_active_users_per_hour, np.nan)

    # ========== B) N(t)：不同地点均值 ==========
    user_wh = (df.groupby(["user_id", "time_key"], observed=True)["venue_id"]
                 .nunique().rename("N_wh").reset_index())
    aggB = (user_wh.groupby("time_key", observed=True)
                  .agg(N_mean=("N_wh", "mean"), users=("N_wh", "size"))
                  .reindex(time_index).fillna(0))
    Nt_raw = pd.Series(aggB["N_mean"], index=time_index)
    Nt_s = _apply_smooth(Nt_raw, smooth_method, rolling_hours_window, ema_alpha)
    Nt_s = Nt_s.interpolate(limit=2, limit_direction="both")
    Nt_s = Nt_s.mask(aggB["users"] < min_active_users_per_hour, np.nan)

    # ========== C) <R/R^rand> vs r_g ==========
    rr = user_metrics.copy()
    rr = rr[(rr["R_rand"] > 0) & np.isfinite(rr["R"]) & np.isfinite(rr["R_rand"])]
    rr["R_rel"] = rr["R"] / rr["R_rand"]
    rr["Rg_plot"] = rr["Rg_km"].clip(upper=rg_max_clip_km)

    rg_min = max(0.5, rr["Rg_plot"].replace(0, np.nan).min(skipna=True) or 0.5)
    bins = np.logspace(np.log10(rg_min), np.log10(rr["Rg_plot"].max() + 1e-6), rg_bins)
    rr["rg_bin"] = pd.cut(rr["Rg_plot"], bins=bins, include_lowest=True)

    def geom_mid(interval):
        return np.sqrt(interval.left * interval.right)

    summ = (rr.groupby("rg_bin", observed=True)["R_rel"]
              .agg(["count", "mean"]).reset_index().dropna())
    summ["rg_mid"] = summ["rg_bin"].apply(geom_mid)
    summ = summ.sort_values("rg_mid").reset_index(drop=True)
    if c_rolling_bins_window and c_rolling_bins_window > 1:
        summ["mean"] = (summ["mean"]
                          .rolling(c_rolling_bins_window, center=True, min_periods=1)
                          .mean())

    # ========== 绘图 ==========
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))

    # A
    ax[0].plot(time_index, Rt_s.values, lw=1.8)
    ax[0].set_ylim(0, 1.0); ax[0].set_xlim(min(time_index), max(time_index))
    ax[0].set_title(f"R(t) — {city}"); ax[0].set_ylabel("R(t)")
    xticks = [0,24,48,72,96,120,144] if bin_hours == 1 else [t for t in range(0,168,24)]
    ax[0].set_xticks(xticks, ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    for s in [(5*24, 6*24-1),(6*24, 7*24-1)]: ax[0].axvspan(*s, alpha=0.08, color="gray")

    # B
    ax[1].plot(time_index, Nt_s.values, lw=1.8)
    ax[1].set_xlim(min(time_index), max(time_index))
    if np.isfinite(Nt_s.values).any():
        ax[1].set_ylim(bottom=max(1.0, np.nanmin(Nt_s.values) * 0.9))
    ax[1].set_title(f"N(t) — {city}"); ax[1].set_ylabel("N(t)")
    ax[1].set_xticks(xticks, ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    for s in [(5*24, 6*24-1),(6*24, 7*24-1)]: ax[1].axvspan(*s, alpha=0.08, color="gray")

    # C
    ax[2].scatter(summ["rg_mid"], summ["mean"], s=30)
    ax[2].set_xscale("log"); ax[2].grid(alpha=0.2, which="both")
    ax[2].set_xlabel(r"$r_g$ (km)"); ax[2].set_ylabel(r"$\langle R/R^{rand} \rangle$")
    ax[2].set_title(rf"$\langle R/R^{{rand}} \rangle$ vs $r_g$ — {city}")

    fig.tight_layout()
    out = f"{out_dir}/{_slug(city)}_fig3_combined.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    return out