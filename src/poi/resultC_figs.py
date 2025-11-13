# resultC_figs_labels.py  (fixed arr() + label de-overlap)
# Outputs:
#   reports_test/figures/figC1_pimax_vs_R_scatter.png
#   reports_test/figures/figC2_city_median_spearman_heatmap.png
#   reports_test/tables/tableC_city_spearman.csv
#   reports_test/combined_summary_c_enriched.csv

from pathlib import Path
import json, math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Path discovery
# =========================
def find_reports_base() -> Path:
    cwd = Path.cwd()
    for up in [cwd] + list(cwd.parents):
        cand = up / "reports_test"
        if cand.exists() and cand.is_dir():
            return cand
    cand = cwd / "reports_test"
    cand.mkdir(parents=True, exist_ok=True)
    return cand

# =========================
# Column normalization
# =========================
ALIAS_MAP = {
    "Pi_max.median": ["Pi_max.median", "pimax_median", "pi_max_median", "pimax.median"],
    "Pi_max.p25":    ["Pi_max.p25", "pimax_p25", "pi_max_p25", "pimax.p25"],
    "Pi_max.p75":    ["Pi_max.p75", "pimax_p75", "pi_max_p75", "pimax.p75"],
    "Rg_km.median":  ["Rg_km.median", "rg_median_km", "rg_km_median"],
    "Rg_km.p25":     ["Rg_km.p25", "rg_p25_km", "rg_km_p25"],
    "Rg_km.p75":     ["Rg_km.p75", "rg_p75_km", "rg_km_p75"],
    "R.median":      ["R.median", "r_median", "R_median"],
    "q.median":      ["q.median", "q_median"],
    "N.median":      ["N.median", "n_median"],
    "n_users_valid": ["n_users_valid", "n_valid", "valid_users"],
    "city":          ["city", "City", "CITY"],
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for canon, aliases in ALIAS_MAP.items():
        for a in aliases:
            if a in df.columns:
                col_map[a] = canon
                break
    df = df.rename(columns=col_map)
    # ensure city
    if "city" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "city"})
        elif df.index.name == "city":
            df = df.reset_index()
        else:
            first = df.columns[0]
            if df[first].dtype == object and "median" not in first.lower():
                df = df.rename(columns={first: "city"})
    if "city" not in df.columns:
        raise KeyError(f"combined has no 'city' column. cols={df.columns.tolist()}")
    df["city"] = df["city"].astype(str).str.strip()
    return df

def load_or_build_combined(base_dir: Path) -> pd.DataFrame:
    csv_path = base_dir / "combined_summary.csv"
    if csv_path.exists():
        return normalize_columns(pd.read_csv(csv_path))
    # build quickly from *_summary.json (fallback)
    rows = []
    for fp in sorted(base_dir.glob("*_summary.json")):
        d = json.loads(fp.read_text())
        city = d.get("city", fp.stem.replace("_summary", ""))
        def g(stats, k):
            if not isinstance(stats, dict): return np.nan
            return stats.get(k, np.nan)
        rows.append({
            "city":          city,
            "n_users_valid": d.get("n_users_valid", np.nan),
            "Pi_max.median": g(d.get("Pi_max"), "median"),
            "Pi_max.p25":    g(d.get("Pi_max"), "p25"),
            "Pi_max.p75":    g(d.get("Pi_max"), "p75"),
            "R.median":      g(d.get("R"), "median"),
            "Rg_km.median":  g(d.get("Rg_km"), "median"),
            "Rg_km.p25":     g(d.get("Rg_km"), "p25"),
            "Rg_km.p75":     g(d.get("Rg_km"), "p75"),
            "q.median":      g(d.get("q"), "median"),
        })
    df = normalize_columns(pd.DataFrame(rows))
    csv_path.write_text(df.to_csv(index=False))
    print(f"[build] saved {csv_path}")
    return df

# =========================
# Safe access
# =========================
def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([np.nan]*len(df), index=df.index, dtype="float64")

# =========================
# Optional enrichment from user_metrics (N via R_rand/S_rand)
# =========================
def infer_N_from_user_metrics(dfu: pd.DataFrame) -> pd.Series:
    rr = pd.to_numeric(dfu.get("R_rand", np.nan), errors="coerce")
    sr = pd.to_numeric(dfu.get("S_rand", np.nan), errors="coerce")
    N1 = 1.0 / rr.replace(0, np.nan)
    N2 = np.power(2.0, sr)
    return pd.to_numeric(N1.where(np.isfinite(N1), N2), errors="coerce")

def enrich_from_user_metrics(base_dir: Path, df_comb: pd.DataFrame) -> pd.DataFrame:
    need_R = ("R.median" not in df_comb.columns) or df_comb["R.median"].isna().all()
    need_q = ("q.median" not in df_comb.columns) or df_comb["q.median"].isna().all()
    need_N = ("N.median" not in df_comb.columns) or df_comb["N.median"].isna().all()

    if not (need_R or need_q or need_N):
        return df_comb

    files = sorted(base_dir.glob("*_user_metrics.csv"))
    rows = []
    for fp in files:
        city = fp.name.replace("_user_metrics.csv", "").strip()
        try:
            dfu = pd.read_csv(fp)
        except Exception:
            dfu = pd.DataFrame()
        rec = {"city": city}
        if need_R and "R" in dfu.columns:
            rec["R.median"] = pd.to_numeric(dfu["R"], errors="coerce").dropna().median()
        if need_q and "q" in dfu.columns:
            rec["q.median"] = pd.to_numeric(dfu["q"], errors="coerce").dropna().median()
        if need_N:
            N_user = infer_N_from_user_metrics(dfu)
            if N_user.notna().sum() > 0:
                rec["N.median"] = N_user.dropna().median()
        rows.append(rec)

    add = pd.DataFrame(rows)
    add["city"] = add["city"].astype(str).str.strip()
    keep = ["city"] + [c for c in ["R.median","q.median","N.median"] if c in add.columns]
    return df_comb.merge(add[keep], on="city", how="left")

# =========================
# Improved Fig C1 with de-overlap
# =========================
def _select_labels(df, top_k=10, outlier_z=1.0):
    """Return indices to annotate: union of top-K by n_users_valid and simple outliers."""
    x = safe_series(df, "R.median").to_numpy(float)
    y = safe_series(df, "Pi_max.median").to_numpy(float)
    n = pd.to_numeric(df.get("n_users_valid", 0), errors="coerce").fillna(0).to_numpy(float)

    # top-K by sample size
    k = min(top_k, len(df))
    top_idx = set(np.argsort(-n)[:k])

    # outliers by robust z (MAD)
    def robust_z(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.array([])
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-9
        return (v - med) / (1.4826 * mad)

    # compute z scores on valid rows only
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    idxv = np.where(mask)[0]

    # If too few points skip outliers
    out_idx = set()
    if len(idxv) >= 5:
        zx = robust_z(xv); zy = robust_z(yv)
        z2 = np.sqrt(zx**2 + zy**2)
        out_idx = set(idxv[np.where(z2 >= outlier_z)[0]])

    return sorted(top_idx.union(out_idx))

def _label_with_offsets(ax, xs, ys, labels, min_dist=0.02):
    """
    Place text labels with a cycle of offsets; avoid placing two labels too close.
    min_dist is in data-units (approx). We cycle offsets and fallback if too close.
    """
    offsets = [(6,6),(6,-6),(-6,6),(-6,-6),(10,0),(-10,0),(0,10),(0,-10),(12,4),(-12,4)]
    placed = []
    for i, (x, y, text) in enumerate(zip(xs, ys, labels)):
        placed_ok = False
        for j, (ox, oy) in enumerate(offsets):
            ann = ax.annotate(text, (x, y), xytext=(ox, oy),
                              textcoords="offset points", fontsize=9)
            too_close = False
            for (px, py) in placed:
                if (abs(px - x) + abs(py - y)) < min_dist:
                    too_close = True
                    break
            if not too_close:
                placed.append((x, y))
                placed_ok = True
                break
            ann.remove()
        if not placed_ok:
            ax.annotate(text, (x, y), xytext=(4, 2), textcoords="offset points", fontsize=9)

def plot_figC1_deoverlap(df_comb: pd.DataFrame,
                         fig_dir: Path,
                         label_top_k=10,
                         outlier_z=1.0,
                         jitter_sigma=0.003,
                         only_yerr=True) -> Path:
    df = df_comb.dropna(subset=["Pi_max.median", "R.median"]).copy()

    x = pd.to_numeric(df["R.median"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["Pi_max.median"], errors="coerce").to_numpy(float)
    n = pd.to_numeric(df.get("n_users_valid", 0), errors="coerce").fillna(0).to_numpy(float)

    # optional micro-jitter
    if jitter_sigma and jitter_sigma > 0:
        rng = np.random.default_rng(2025)
        x = x + rng.normal(0, jitter_sigma, size=x.shape)
        y = y + rng.normal(0, jitter_sigma, size=y.shape)

    # -------- FIXED arr(): always return np.ndarray of len(df) --------
    def arr(col):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        else:
            return np.full(len(df), np.nan, dtype=float)
    # ------------------------------------------------------------------

    Px25, Px75 = arr("Pi_max.p25"), arr("Pi_max.p75")
    Px75 = np.minimum(Px75, 0.6)
    yerr = None
    if np.isfinite(Px25).any() and np.isfinite(Px75).any():
        yerr = np.vstack([np.maximum(0, y - Px25), np.maximum(0, Px75 - y)])

    Rx25, Rx75 = arr("R.p25"), arr("R.p75")
    xerr = None
    if not only_yerr and np.isfinite(Rx25).any() and np.isfinite(Rx75).any():
        xerr = np.vstack([np.maximum(0, x - Rx25), np.maximum(0, Rx75 - x)])

    sizes = 20 + 220 * (n / np.nanmax(n))**0.35 if np.nanmax(n) > 0 else np.full_like(x, 60, dtype=float)

    plt.figure(figsize=(10.2, 7.6))
    if xerr is not None or yerr is not None:
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", lw=0.7, alpha=0.65)
    plt.scatter(x, y, s=sizes, alpha=0.85, edgecolors="k", linewidths=0.4)

    idx_to_label = _select_labels(df.assign(**{"R.median": x, "Pi_max.median": y,
                                               "n_users_valid": n}),
                                  top_k=label_top_k, outlier_z=outlier_z)
    xs = x[idx_to_label]; ys = y[idx_to_label]; labels = df.iloc[idx_to_label]["city"].tolist()
    ax = plt.gca()
    _label_with_offsets(ax, xs, ys, labels, min_dist=0.02)

    plt.xlabel(r"$R$ (median)")
    plt.ylabel(r"$\Pi_{\max}$ (median)")
    plt.title("Fig C1. City-level medians: predictability vs regularity (Test)")
    plt.grid(alpha=0.25)
    plt.xlim(0.0, 1.0)
    out = fig_dir / "figC1_pimax_vs_R_scatter.png"
    plt.tight_layout(); plt.savefig(out, dpi=240); plt.close()
    print(f"[figure] {out} (labels={len(idx_to_label)})")
    return out

# =========================
# Table & Heatmap
# =========================
def build_tableC(df_comb: pd.DataFrame, tab_dir: Path) -> Path:
    try:
        from scipy.stats import spearmanr
        use_scipy = True
    except Exception:
        use_scipy = False
        print("[warn] scipy not available; p-values will be NaN.")

    target = safe_series(df_comb, "Pi_max.median")
    features = {
        "R.median":     safe_series(df_comb, "R.median"),
        "Rg_km.median": safe_series(df_comb, "Rg_km.median"),
        "q.median":     safe_series(df_comb, "q.median"),
        "N.median":     safe_series(df_comb, "N.median"),
    }

    rows = []
    for fname, series in features.items():
        mask = target.notna() & series.notna()
        if mask.sum() >= 3:
            if use_scipy:
                rho, p = spearmanr(target[mask], series[mask])
            else:
                rho = target[mask].corr(series[mask], method="spearman")
                p = np.nan
            rows.append({"feature": fname, "rho": float(rho),
                         "p": (float(p) if isinstance(p, (int,float)) else np.nan),
                         "n_cities": int(mask.sum())})
        else:
            rows.append({"feature": fname, "rho": np.nan, "p": np.nan, "n_cities": int(mask.sum())})
    out = tab_dir / "tableC_city_spearman.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[table] {out}")
    return out

def plot_figC2_heatmap(df_comb: pd.DataFrame, fig_dir: Path) -> Path | None:
    cols = ["Pi_max.median", "R.median", "Rg_km.median", "q.median"]
    if "N.median" in df_comb.columns and not df_comb["N.median"].isna().all():
        cols.append("N.median")
    df_heat = df_comb[cols].dropna(how="any").copy()
    if df_heat.shape[0] < 3:
        print("[warn] Not enough complete cities to draw heatmap (need >=3).")
        return None

    corr_mat = df_heat.corr(method="spearman").to_numpy()
    labels = df_heat.columns.tolist()

    plt.figure(figsize=(8.6, 7.4))
    im = plt.imshow(corr_mat, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_mat[i, j]
            txt = f"{val:.2f}" if np.isfinite(val) else ""
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)
    plt.title("Fig C2. Spearman correlations among city-level medians")
    out = fig_dir / "figC2_city_median_spearman_heatmap.png"
    plt.tight_layout(); plt.savefig(out, dpi=240); plt.close()
    print(f"[figure] {out}")
    return out

# =========================
# Main
# =========================
def main():
    base = find_reports_base()
    fig_dir = base / "figures"
    tab_dir = base / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print(f"[path] BASE_DIR = {base}")

    df_comb = load_or_build_combined(base)
    df_comb = normalize_columns(df_comb)

    # optional enrichment for N/R/q from user_metrics if missing
    df_comb = enrich_from_user_metrics(base, df_comb)

    # save enriched snapshot
    (base / "combined_summary_c_enriched.csv").write_text(df_comb.to_csv(index=False))
    print(f"[saved] {base/'combined_summary_c_enriched.csv'}")

    # outputs
    out_c1 = plot_figC1_deoverlap(df_comb, fig_dir,
                                  label_top_k=10,
                                  outlier_z=1.0,
                                  jitter_sigma=0.003,
                                  only_yerr=True)
    out_tab = build_tableC(df_comb, tab_dir)
    out_c2 = plot_figC2_heatmap(df_comb, fig_dir)

if __name__ == "__main__":
    main()
