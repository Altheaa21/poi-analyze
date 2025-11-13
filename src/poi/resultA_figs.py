# ==== paper_figs.py (改进版) ====
# - Fig.1：横向误差线改为 winsorized IQR（上限封顶 0.6），x 轴缩放到 [0, 0.4]；
#           标签自动“错位标注”（左右交替 + 高度微抖动），尽量避免重叠；点径映射更拉开差距；
# - Fig.2：优先画 Π_max 箱线图 + y 轴聚焦 [0.05, 0.35]；同时输出“Π_max=1 占比”条形小图；
#           若缺 CSV，则回退为“条形(中位)+IQR”，并同样聚焦 y 轴与排序、标注 n。
#
# 运行：
#   PYTHONPATH=. python -m src.poi.paper_figs
# 或 python src/poi/paper_figs.py

from pathlib import Path
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------
# 路径：自动定位“项目根”
# ------------------------
ROOT = Path(__file__).resolve().parents[2]  # 项目根（包含 configs/, src/, reports_test/ 等）
BASE_DIR = ROOT / "reports_test"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("[path] ROOT     =", ROOT)
print("[path] BASE_DIR =", BASE_DIR)
print("[path] FIG_DIR  =", FIG_DIR)

# ------------------------
# 实用函数
# ------------------------
def _winsorize_upper(a: np.ndarray, ub: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a[~np.isfinite(a)] = np.nan
    return np.minimum(a, ub)

def _label_positions(x, y):
    """为散点生成非重叠的标注偏移（简单启发式：按 y 排序，左右交替 + 轻微垂直抖动）。"""
    idx = np.argsort(y)
    offsets = []
    for k, i in enumerate(idx):
        dx = 8 if (k % 2 == 0) else -8    # 左右交替
        dy = (k % 3 - 1) * 3              # -3, 0, 3 循环
        offsets.append((i, dx, dy))
    # 恢复原顺序
    out = [None]*len(x)
    for i, dx, dy in offsets:
        out[i] = (dx, dy)
    return out

# ========== Fig.1：Π_max(median) vs R_g(median) + winsorized IQR ==========
def build_combined_summary(base_dir: Path) -> pd.DataFrame:
    files = sorted((base_dir).glob("*_summary.json"))
    print(f"[Fig1] found summary files: {len(files)}")
    rows = []
    for fp in files:
        try:
            d = json.loads(Path(fp).read_text())
            city = d.get("city", fp.name.replace("_summary.json", ""))

            def get_nested(stats: dict, key: str):
                if not isinstance(stats, dict):
                    return np.nan
                return stats.get(key, np.nan)

            rows.append({
                "city": city,
                "n_users_valid": d.get("n_users_valid", np.nan),
                "pimax_median": get_nested(d.get("Pi_max"), "median"),
                "pimax_p25":    get_nested(d.get("Pi_max"), "p25"),
                "pimax_p75":    get_nested(d.get("Pi_max"), "p75"),
                "rg_median_km": get_nested(d.get("Rg_km"), "median"),
                "rg_p25_km":    get_nested(d.get("Rg_km"), "p25"),
                "rg_p75_km":    get_nested(d.get("Rg_km"), "p75"),
            })
        except Exception as e:
            print(f"[warn] failed to read {fp}: {e}")

    df = pd.DataFrame(rows)
    print("[Fig1] combined columns:", df.columns.tolist())
    print("[Fig1] combined head:\n", df.head(3))

    need_cols = ["pimax_median", "rg_median_km"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[Fig1] missing required columns: {missing}")

    df_plot = df.dropna(subset=need_cols, how="any").sort_values("city").reset_index(drop=True)
    # 保存合并结果
    out_json = BASE_DIR / "combined_summary.json"
    out_csv  = BASE_DIR / "combined_summary.csv"
    df.to_json(out_json, orient="records", indent=2, force_ascii=False)
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_json}\n[saved] {out_csv}")
    return df_plot

def plot_fig1(df_plot: pd.DataFrame, fig_dir: Path):
    x = df_plot["pimax_median"].to_numpy()
    y = df_plot["rg_median_km"].to_numpy()

    # winsorized IQR：把 p75 上封顶到 0.6，避免横向误差线“冲顶”
    p25 = df_plot.get("pimax_p25", np.nan).to_numpy(dtype=float)
    p75 = df_plot.get("pimax_p75", np.nan).to_numpy(dtype=float)
    cap = 0.6
    p75_w = _winsorize_upper(p75, cap)

    xerr = np.vstack([
        np.maximum(0, x - p25),
        np.maximum(0, p75_w - x)
    ])
    yerr = np.vstack([
        np.maximum(0, y - df_plot.get("rg_p25_km", np.nan).to_numpy(dtype=float)),
        np.maximum(0, df_plot.get("rg_p75_km", np.nan).to_numpy(dtype=float) - y)
    ])

    # 点径：更拉开差距
    n = df_plot.get("n_users_valid", pd.Series([0]*len(df_plot))).to_numpy(dtype=float)
    if np.nanmax(n) > 0:
        s = 20 + 220 * (n / np.nanmax(n))**0.35
    else:
        s = np.full_like(x, 60, dtype=float)

    plt.figure(figsize=(9.8, 7.2))
    # 误差线
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", lw=0.8, alpha=0.7)
    # 散点
    plt.scatter(x, y, s=s, alpha=0.9)

    # 标签：左右交替 + 轻微抖动；并画细线连接
    offsets = _label_positions(x, y)
    for (xi, yi, city), (dx, dy) in zip(zip(x, y, df_plot["city"].tolist()), offsets):
        plt.annotate(
            city, (xi, yi), xytext=(dx, dy), textcoords="offset points",
            fontsize=9, ha="left" if dx>0 else "right",
            arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.6)
        )

    plt.xlabel(r"$\Pi_{\max}$ (median)")
    plt.ylabel(r"$R_g$ (median, km)")
    plt.title("Cross-city map: predictability vs spatial range (Test)")
    plt.grid(alpha=0.25)
    plt.xlim(0.0, 0.4)  # 聚焦主变动区间
    out_path = fig_dir / "fig1_predictability_vs_range.png"
    plt.tight_layout(); plt.savefig(out_path, dpi=240); plt.close()
    print(f"[figure] {out_path}")

# ========== Fig.2：箱线（聚焦 0.05–0.35） + “Π_max=1 占比”小图；无CSV则回退条形+IQR ==========
def plot_fig2(base_dir: Path, fig_dir: Path):
    csv_files = sorted((base_dir).glob("*_user_metrics.csv"))
    print(f"[Fig2] found user_metrics csvs: {len(csv_files)}")
    city_to_series = {}
    n_map = {}
    share_one = {}

    for fp in csv_files:
        try:
            city = fp.name.replace("_user_metrics.csv", "")
            dfu = pd.read_csv(fp)
            col = None
            for cand in ["Pi_max", "pi_max", "PI_MAX"]:
                if cand in dfu.columns:
                    col = cand; break
            if col is None:
                print(f"[Fig2] skip {city}: no Pi_max column")
                continue
            s = pd.to_numeric(dfu[col], errors="coerce").dropna()
            n_map[city] = int(s.shape[0])
            if s.shape[0] >= 1:
                city_to_series[city] = s
                share_one[city] = float((s >= 0.9999).mean())
            else:
                print(f"[Fig2] {city}: n=0")
        except Exception as e:
            print(f"[warn] failed to read {fp}: {e}")

    if len(city_to_series) >= 1:
        # 按中位排序
        cities = sorted(city_to_series.keys(), key=lambda c: float(np.median(city_to_series[c])))
        data = [city_to_series[c] for c in cities]

        # 箱线图（聚焦 y 轴）
        plt.figure(figsize=(max(10, 0.7*len(cities)+3), 6.5))
        plt.boxplot(data, labels=[f"{c}\n(n={n_map.get(c,0)})" for c in cities],
                    showfliers=False, patch_artist=False, vert=True)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(r"$\Pi_{\max}$ (per-user distribution)")
        plt.title("City-wise distribution of $\Pi_{max}$ (Test, boxplots; zoomed)")
        plt.ylim(0.05, 0.35)  # 仅展示可比区间
        plt.grid(axis="y", alpha=0.25)
        out_box = fig_dir / "fig2_pi_max_boxplot_zoom.png"
        plt.tight_layout(); plt.savefig(out_box, dpi=240); plt.close()
        print(f"[figure] {out_box}")

        # “Π_max=1 占比”条形小图
        plt.figure(figsize=(max(10, 0.7*len(cities)+3), 4.2))
        y = [share_one.get(c, 0.0) for c in cities]
        plt.bar(range(len(cities)), y)
        plt.xticks(range(len(cities)), cities, rotation=35, ha="right")
        plt.ylabel("Share of $\Pi_{max}=1$")
        plt.title("Prevalence of $\Pi_{max}=1$ users (Test)")
        plt.ylim(0.0, 1.0)
        plt.grid(axis="y", alpha=0.25)
        out_share = fig_dir / "fig2_share_pimax1.png"
        plt.tight_layout(); plt.savefig(out_share, dpi=240); plt.close()
        print(f"[figure] {out_share}")
        return

    # ---- fallback：从 summary 画条形+IQR ----
    files = sorted((base_dir).glob("*_summary.json"))
    rows = []
    for fp in files:
        try:
            d = json.loads(Path(fp).read_text())
            city = d.get("city", fp.name.replace("_summary.json", ""))
            Pi = d.get("Pi_max", {})
            rows.append({
                "city": city,
                "median": Pi.get("median", np.nan),
                "p25":    Pi.get("p25", np.nan),
                "p75":    Pi.get("p75", np.nan),
                "n_users_valid": d.get("n_users_valid", np.nan),
            })
        except Exception as e:
            print(f"[warn] failed to read {fp}: {e}")

    df = pd.DataFrame(rows)
    print("[Fig2] fallback columns:", df.columns.tolist())
    print("[Fig2] fallback head:\n", df.head(3))

    if "median" not in df.columns:
        raise RuntimeError("[Fig2] fallback missing column: 'median'")

    df = df.dropna(subset=["median"]).sort_values("median").reset_index(drop=True)
    y = df["median"].to_numpy()
    yerr = np.vstack([
        np.maximum(0, y - df.get("p25", np.nan).to_numpy(dtype=float)),
        np.maximum(0, df.get("p75", np.nan).to_numpy(dtype=float) - y)
    ])

    plt.figure(figsize=(max(10, 0.6*len(df)+3), 6.2))
    plt.bar(range(len(df)), y)
    plt.errorbar(range(len(df)), y, yerr=yerr, fmt="none", lw=1.0, alpha=0.8)
    xt = [f"{c}\n(n={int(n)})" for c, n in zip(df["city"].tolist(), df["n_users_valid"].fillna(0))]
    plt.xticks(range(len(df)), xt, rotation=35, ha="right")
    plt.ylabel(r"$\Pi_{\max}$ (median ± IQR)")
    plt.title("City-wise $\Pi_{max}$ (Test, bar + IQR; zoomed)")
    plt.ylim(0.05, 0.35)  # 聚焦区间
    plt.grid(axis="y", alpha=0.25)
    out_path = fig_dir / "fig2_pi_max_distribution_zoom.png"
    plt.tight_layout(); plt.savefig(out_path, dpi=240); plt.close()
    print(f"[figure] {out_path}")

# ------------------------
# 主流程
# ------------------------
def main():
    df_plot = build_combined_summary(BASE_DIR)
    plot_fig1(df_plot, FIG_DIR)
    plot_fig2(BASE_DIR, FIG_DIR)

if __name__ == "__main__":
    main()
