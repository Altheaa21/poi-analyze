import os, json, yaml, pandas as pd
from .io import load_parquet, apply_bbox
from .time_utils import to_local, week_hour
from .metrics import compute_q, compute_entropy, solve_pimax, compute_rg, compute_R
from .figs import plot_fig3_combined


def _summ(s: pd.Series):
    """Safe summary stats helper. Returns {} if the series is empty."""
    if s is None:
        return {}
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
        count=int(s.shape[0]),
    )


def _empty_user_metrics_df():
    """Create an empty user-metrics DataFrame with expected columns.
    This prevents 'cannot set column on DataFrame without columns' errors.
    """
    cols = ["user_id", "n_points", "q", "N", "S_unc", "S_rand",
            "Pi_max", "Rg_km", "R", "R_rand"]
    return pd.DataFrame(columns=cols)


def _write_minimal_summary(out_dir: str, city: str, tz: str,
                           n_users_total: int, n_users_valid: int,
                           note: str, figure_path=None):
    """Write a minimal city-level summary JSON when we skip heavy metrics/plots."""
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "city": city,
        "tz": tz,
        "n_users_total": int(n_users_total),
        "n_users_valid": int(n_users_valid),
        "q": {},
        "S_unc": {},
        "S_rand": {},
        "Pi_max": {},
        "Rg_km": {},
        "R": {},
        "figure": figure_path,
        "note": note,
    }
    with open(f"{out_dir}/{city}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {out_dir}/{city}_summary.json (note: {note})")
    return summary


def analyze_city(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    city = cfg["city"]
    tz = cfg["tz"]
    out_dir = cfg.get("out_dir", "reports")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/figures", exist_ok=True)

    # -----------------------------
    # 1) Load data & pre-processing
    # -----------------------------
    df = load_parquet(cfg["data_path"])
    df = apply_bbox(df, cfg.get("bbox"))
    # local time once, reused by R(t) and plots
    df["ts_local"] = to_local(df, tz)

    n_users_total = int(df["user_id"].nunique())

    # -----------------------------
    # 2) Sparsity q & filtering
    # -----------------------------
    user_q = compute_q(df)  # per-user q
    points = df.groupby("user_id", observed=True).size().rename("n_points").reset_index()
    user_q = user_q.merge(points, on="user_id", how="left")

    valid_mask = (user_q["q"] < cfg["q_threshold"]) & (user_q["n_points"] >= cfg["min_points"])
    valid_users = user_q.loc[valid_mask, "user_id"]
    traj_valid = df[df["user_id"].isin(valid_users)].reset_index(drop=True)

    n_users_valid = int(valid_users.nunique())

    # If nothing survives filtering, write a minimal summary and return early
    if n_users_valid == 0 or traj_valid.empty:
        note = "No valid users after filtering; skipped entropy/predictability, Rg, and figures."
        summary = _write_minimal_summary(out_dir, city, tz, n_users_total, n_users_valid, note, figure_path=None)
        # Also write an empty per-user CSV to keep outputs consistent
        _empty_user_metrics_df().to_csv(f"{out_dir}/{city}_user_metrics.csv", index=False)
        print(f"[saved] {out_dir}/{city}_user_metrics.csv (empty)")
        return _empty_user_metrics_df(), summary

    # -----------------------------------
    # 3) Entropy & theoretical Pi_max
    # -----------------------------------
    user_entropy = compute_entropy(traj_valid)
    # Guard: if entropy stage returns nothing (e.g., each user has <2 unique venues)
    if user_entropy is None or user_entropy.empty:
        # Create a well-formed empty frame with expected columns so column assignment won't fail
        user_entropy = pd.DataFrame(columns=["user_id", "N", "S_unc", "S_rand"])

    # Only compute Pi_max if S_unc and N are available
    if {"S_unc", "N"}.issubset(set(user_entropy.columns)) and not user_entropy.empty:
        # Safe row-wise apply
        user_entropy["Pi_max"] = user_entropy.apply(
            lambda r: solve_pimax(r["S_unc"], int(r["N"])), axis=1
        )
    else:
        # Ensure the column exists for consistent downstream merges
        user_entropy["Pi_max"] = pd.Series(dtype=float)

    # -----------------------------
    # 4) Radius of gyration Rg
    # -----------------------------
    user_rg = compute_rg(traj_valid)
    if user_rg is None or user_rg.empty:
        user_rg = pd.DataFrame(columns=["user_id", "Rg_km", "n_points"])

    # -----------------------------
    # 5) Weekly regularity R (168 h)
    # -----------------------------
    tl = traj_valid[["user_id", "venue_id", "ts_local"]].copy()
    tl["week_hour"] = week_hour(tl["ts_local"])
    user_R = compute_R(tl)
    if user_R is None or user_R.empty:
        user_R = pd.DataFrame(columns=["user_id", "R", "R_rand", "N"])

    # --------------------------------
    # 6) Merge user-level metrics
    # --------------------------------
    user_metrics = (
        user_q[["user_id", "q", "n_points"]]
        .merge(user_entropy, on="user_id", how="inner")  # keep users that have entropy rows
        .merge(user_rg[["user_id", "Rg_km", "n_points"]], on="user_id", how="left", suffixes=("", "_rg"))
        .merge(user_R[["user_id", "R", "R_rand", "N"]], on="user_id", how="left")
    )

    # If inner-join with entropy yields empty (all dropped), fall back to a safe empty table with columns,
    # and continue to summary/plot with a clear note.
    if user_metrics is None or user_metrics.empty:
        user_metrics = _empty_user_metrics_df()

    # --------------------------------
    # 7) Persist per-user CSV
    # --------------------------------
    cols = [c for c in ["user_id", "n_points", "q", "N", "S_unc", "S_rand", "Pi_max", "Rg_km", "R", "R_rand"]
            if c in user_metrics.columns]
    user_metrics[cols].to_csv(f"{out_dir}/{city}_user_metrics.csv", index=False)
    print(f"[saved] {out_dir}/{city}_user_metrics.csv")

    # --------------------------------
    # 8) Plot Fig. 3 (R(t), N(t), <R/R_rand> vs r_g)
    #    Only if we have something to plot; otherwise skip gracefully.
    # --------------------------------
    fig_path = None
    try:
        if not traj_valid.empty and not user_metrics.empty:
            fig_path = plot_fig3_combined(
                traj_valid.assign(ts_local=traj_valid["ts_local"]),
                user_metrics,
                city,
                out_dir=f"{out_dir}/figures",
                min_active_users_per_hour=cfg.get("min_active_users_per_hour", 10),
                # smoothing params from YAML
                smooth_method=cfg.get("smooth_method", "rolling"),   # "rolling" | "ema" | "none"
                rolling_hours_window=cfg.get("rolling_hours_window", 6),
                ema_alpha=cfg.get("ema_alpha", 0.25),
                bin_hours=cfg.get("bin_hours", 2),
                rg_bins=cfg.get("rg_bins", 12),
                rg_max_clip_km=cfg.get("rg_clip_km", 100.0),
                c_rolling_bins_window=cfg.get("c_rolling_bins_window", 2),
            )
            print(f"[figure] {fig_path}")
    except Exception as e:
        # Never let plotting kill the pipeline; still write a summary below.
        print(f"[warn] plotting skipped due to error: {e}")
        fig_path = None

    # --------------------------------
    # 9) City-level summary JSON
    # --------------------------------
    summary = {
        "city": city,
        "tz": tz,
        "n_users_total": int(n_users_total),
        "n_users_valid": int(user_metrics["user_id"].nunique()),
        "q": _summ(user_metrics["q"]) if "q" in user_metrics else {},
        # "N": _summ(user_metrics["N"]) if "N" in user_metrics else {},  # left commented as per your preference
        "S_unc": _summ(user_metrics["S_unc"]) if "S_unc" in user_metrics else {},
        "S_rand": _summ(user_metrics["S_rand"]) if "S_rand" in user_metrics else {},
        "Pi_max": _summ(user_metrics["Pi_max"]) if "Pi_max" in user_metrics else {},
        "Rg_km": _summ(user_metrics["Rg_km"]) if "Rg_km" in user_metrics else {},
        "R": _summ(user_metrics["R"]) if "R" in user_metrics else {},
        "figure": fig_path,
    }
    with open(f"{out_dir}/{city}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {out_dir}/{city}_summary.json")

    return user_metrics, summary