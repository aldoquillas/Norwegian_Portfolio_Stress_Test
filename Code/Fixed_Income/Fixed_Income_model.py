# ============================ IR Models (Normal, Student-t, Filtered Historical Sim)   ============================
# Libraries
from __future__ import annotations

import argparse, math, re, warnings, json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, t as student_t, kstest

# ---------------------------- Data ----------------------------
def _z(alpha: float) -> float:
    return float(norm.ppf(alpha))

def _as_2d(df: pd.DataFrame) -> np.ndarray:
    a = np.asarray(df, float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("Matrix must be non-empty 2D.")
    return a

def _as_1d(x) -> np.ndarray:
    r = pd.Series(x).dropna().values
    if r.ndim != 1 or r.size == 0:
        raise ValueError("Vector must be non-empty 1D.")
    return r.astype(float, copy=False)

def _to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.resample("ME").last()  
    return df
    """
    Loading data 
    """
def _parse_curve_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]:"Date"}, inplace=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="raise")
        except Exception:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            if df["Date"].isna().mean() > 0.05:
                raise ValueError(f"Failed to parse 'Date' in {csv_path}")
    df = df.set_index("Date").sort_index()

    mat_cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            m = re.search(r"(\d+)", str(c))
            if m:
                mat_cols.append((c, m.group(1)))

    if not mat_cols:
        raise ValueError(f"No numeric maturity columns found in {csv_path}")

    out = pd.DataFrame(index=df.index)
    for raw, yr in mat_cols:
        out[yr] = pd.to_numeric(df[raw], errors="coerce")

    out = _to_month_end(out).dropna(how="all")
    out = out.ffill().bfill()
    return out

def _detect_percent_units(yld: pd.DataFrame) -> bool:
    sample = yld.stack().dropna()
    if sample.empty:
        return True
    med = sample.median()
    return med < 1.0

def yields_to_dy_bp(yld: pd.DataFrame) -> pd.DataFrame:
    decimals = _detect_percent_units(yld)
    if decimals:
        dy = (yld - yld.shift(1)) * 10000.0
    else:
        dy = (yld - yld.shift(1)) * 100.0
    return dy.dropna()

def align_maturities(de: pd.DataFrame, us: pd.DataFrame, mats: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    de2 = de.copy(); us2 = us.copy()
    missing_de = [m for m in mats if m not in de2.columns]
    missing_us = [m for m in mats if m not in us2.columns]
    if missing_de: raise ValueError(f"German curve missing maturities: {missing_de}")
    if missing_us: raise ValueError(f"US curve missing maturities: {missing_us}")
    de2 = de2[mats]; us2 = us2[mats]
    common_idx = de2.index.intersection(us2.index)
    de2 = de2.loc[common_idx].copy()
    us2 = us2.loc[common_idx].copy()
    return de2, us2

# Single matrix for USA & German yields
def stack_curves_bp(dy_de: pd.DataFrame, dy_us: pd.DataFrame, mats: List[str]) -> pd.DataFrame:
    de_cols = [f"DE_{m}" for m in mats]; us_cols = [f"US_{m}" for m in mats]
    X = pd.DataFrame(index=dy_de.index, columns=de_cols + us_cols, dtype=float)
    X.loc[:, de_cols] = dy_de[mats].values
    X.loc[:, us_cols] = dy_us[mats].values
    return X

# Fixed Income sensitivities by maturity and Economy
def read_sensitivities(sens_path: Path, mats: List[str]) -> pd.Series:
    df = pd.read_csv(sens_path)
    req = {"Curve","Maturity","KRD"}
    if not req.issubset(set(df.columns)):
        raise ValueError("Sensitivities CSV must have columns: Curve,Maturity,KRD")
    df["Curve"] = df["Curve"].astype(str).str.upper().str.strip()
    df["Maturity"] = df["Maturity"].astype(int).astype(str)
    w = {}
    for curve in ["DE","US"]:
        for m in mats:
            krd = df.loc[(df["Curve"]==curve) & (df["Maturity"]==m), "KRD"]
            w[f"{curve}_{m}"] = float(krd.iloc[0]) if not krd.empty else 0.0
    return pd.Series(w, dtype=float)
# weighted vector
def unit_sensitivities(mats: List[str]) -> pd.Series:
    idx = [*(f"DE_{m}" for m in mats), *(f"US_{m}" for m in mats)]
    return pd.Series(1.0, index=idx, dtype=float)

# ---------------------------- P&L Sensitivities & Backtesting Models ----------------------------
def pnl_from_dy_bp(dy_vec: np.ndarray, w_vec: np.ndarray) -> float:
    return -float(np.dot(w_vec, dy_vec))

def hs_var_es(pnl_hist: np.ndarray, alpha: float) -> Tuple[float, float]:
    pnl = _as_1d(pnl_hist)
    q = np.quantile(pnl, 1 - alpha, method="lower") if hasattr(np, "quantile") else np.quantile(pnl, 1 - alpha)
    tail = pnl[pnl <= q]
    es = float(tail.mean()) if tail.size else float(q)
    return float(q), es

def var_es_normal(m: float, s: float, alpha: float) -> Tuple[float, float]:
    z = _z(alpha)
    var = -(m + s * z)
    es  = -(m + s * (norm.pdf(z) / (1 - alpha)))
    return float(var), float(es)

def var_es_student_t(m: float, s: float, alpha: float, nu: float) -> Tuple[float, float]:
    q = float(student_t.ppf(alpha, df=nu))
    pdf_q = float(student_t.pdf(q, df=nu))
    var = -(m + s * q)
    es  = -(m + s * ((nu + q*q) / (nu - 1)) * (pdf_q / (1 - alpha)))
    return float(var), float(es)

def ewma_var(series: np.ndarray, lam: float = 0.97) -> float:
    x = np.asarray(series, float)
    x = x - x.mean()
    s2 = 0.0
    for xi in x:
        s2 = lam * s2 + (1 - lam) * (xi * xi)
    return float(s2)

# ---------------------------- PCA to explain 95% of variance ----------------------------
def pca_from_window(X_win: np.ndarray, var_target: float = 0.95):
    Xc = X_win - X_win.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = (S**2) / (Xc.shape[0] - 1)           # eigenvalues of covariance
    cum = np.cumsum(lam) / lam.sum()
    K = int(np.searchsorted(cum, var_target) + 1)
    V = Vt[:K].T
    lamK = lam[:K]
    return V, lamK, Xc, K, cum[K-1]

# ---------------------------- Backtesting Tests ----------------------------
def kupiec_pof(hits: np.ndarray, alpha: float) -> dict:
    h = np.asarray(hits, int); x = int(h.sum()); n = int(h.size); p = 1 - alpha
    t1 = 0.0 if x == 0 else x * math.log(x / n)
    t2 = 0.0 if (n - x) == 0 else (n - x) * math.log((n - x) / n)
    lr = 2 * (t1 + t2 - (x * math.log(p) + (n - x) * math.log(1 - p)))
    return {"LR_pof": float(lr), "p_value": float(1 - chi2.cdf(lr, 1)), "exceptions": x, "n": n, "hit_rate": x / n}

def christoffersen_independence(hits: np.ndarray) -> Dict[str, float]:
    h = np.asarray(hits, int)
    if h.size < 2: return {"LR_ind": np.nan, "p_value": np.nan}
    n00 = n01 = n10 = n11 = 0
    for i in range(1, h.size):
        a, b = h[i-1], h[i]
        if a==0 and b==0: n00 += 1
        if a==0 and b==1: n01 += 1
        if a==1 and b==0: n10 += 1
        if a==1 and b==1: n11 += 1
    n0, n1 = n00 + n01, n10 + n11
    if n0==0 or n1==0: return {"LR_ind": np.nan, "p_value": np.nan}
    p01, p11 = n01/n0, n11/n1
    p = (n01 + n11) / (n0 + n1)
    ll_re = n00*math.log(1-p)+n01*math.log(p)+n10*math.log(1-p)+n11*math.log(p)
    ll_un = 0.0
    if n00: ll_un += n00*math.log(1-p01)
    if n01: ll_un += n01*math.log(p01)
    if n10: ll_un += n10*math.log(1-p11)
    if n11: ll_un += n11*math.log(p11)
    lr = -2*(ll_re-ll_un)
    return {"LR_ind": float(lr), "p_value": float(1 - chi2.cdf(lr, 1))}

def christoffersen_cc(hits: np.ndarray, alpha: float) -> Dict[str, float]:
    pof = kupiec_pof(hits, alpha); ind = christoffersen_independence(hits)
    if np.isnan(ind["p_value"]): return {"LR_cc": np.nan, "p_value": np.nan}
    lr_cc = pof["LR_pof"] + ind["LR_ind"]
    return {"LR_cc": float(lr_cc), "p_value": float(1 - chi2.cdf(lr_cc, 2))}

def es_score_stat(pnl: np.ndarray, var_f: np.ndarray, es_f: np.ndarray, alpha: float) -> Dict[str, float]:
    r = _as_1d(pnl); v = _as_1d(var_f); e = _as_1d(es_f)
    I = (r < v).astype(float)
    S = (I * (v - r) / alpha) - (e - r)
    m = float(S.mean())
    se = float(S.std(ddof=1)) / math.sqrt(S.size) if S.size > 1 else np.inf
    t_stat = m / se if se > 0 else np.inf
    p = 2 * (1 - norm.cdf(abs(t_stat)))
    return {"t_stat": float(t_stat), "p_value": float(p), "mean_score": m}

# ---------------------------- Goodness of Fit ----------------------------
def pit_u_sequence(pnl_realized: np.ndarray, var_mu: np.ndarray, var_sig: np.ndarray, dist_name: str, nu_seq: np.ndarray | None) -> np.ndarray:
    r = np.asarray(pnl_realized, float)
    m = np.asarray(var_mu, float)
    s = np.asarray(var_sig, float)
    z = (r - m) / np.where(s>0, s, np.nan)
    if dist_name == "normal":
        u = norm.cdf(z)
    elif dist_name == "student_t":
        if nu_seq is None:
            raise ValueError("nu sequence required for student_t PIT")
        u = np.array([student_t.cdf(zi, df=max(2.5, float(nui))) for zi, nui in zip(z, nu_seq)])
    else:
        raise ValueError("Unsupported dist_name in PIT")
    return u

def ks_on_pit(u: np.ndarray) -> Dict[str, float]:
    u = np.asarray(u, float)
    u = u[np.isfinite(u)]
    if u.size < 8:
        return {"ks_stat": np.nan, "ks_p": np.nan}
    stat, p = kstest(u, 'uniform')
    return {"ks_stat": float(stat), "ks_p": float(p)}

def plot_backtest(dates, pnl, var, alpha, title, outpath: Path):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dates, pnl, linewidth=1, label="P&L")
    ax.plot(dates, var, linewidth=2, label=f"VaR {alpha:.3f}")
    breaches = pnl < var
    if breaches.any():
        ax.scatter(np.array(dates)[breaches], np.array(pnl)[breaches], s=30, label="Exceptions")
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("P&L")
    ax.legend(); fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_pit_hist(u: np.ndarray, title: str, outpath: Path):
    u = np.asarray(u, float)
    u = u[np.isfinite(u)]
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(1,1,1)
    ax.hist(u, bins=20, density=True, alpha=0.7)
    ax.set_title(title + " (PIT histogram)")
    ax.set_xlabel("u ~ Uniform(0,1) under correct model")
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)

def rolling_models_with_pca_and_fhs(
    X_bp: pd.DataFrame,
    w: pd.Series,
    alpha: float,
    window: int,
    pca_var: float,
    ewma_lambda: float,
    t_grid: Tuple[float,float,float]=(3.0, 20.0, 0.5),
    include_hs: bool=True
) -> Dict[str, pd.DataFrame]:

    X = X_bp.copy().dropna()
    idx = X.index
    w = w.reindex(X.columns).fillna(0.0).astype(float)
    pnl = -np.dot(X.values, w.values)
    n = len(X)
    if n <= window + 5:
        raise ValueError(f"Not enough observations {n} for window={window}")

    # Containers for each model
    res = {
        "PCA_Normal": {"pnl": [], "VaR": [], "ES": [], "mu": [], "sig": [], "nu": []},
        "PCA_StudentT": {"pnl": [], "VaR": [], "ES": [], "mu": [], "sig": [], "nu": []},
        "FHS": {"pnl": [], "VaR": [], "ES": []}
    }

    for t in range(window, n):
        X_win = X.iloc[t-window:t].values
        # ----- PCA variance for parametric models -----
        V, lamK, Xc, K, cumvar = pca_from_window(X_win, var_target=pca_var)
        w_pc = V.T @ w.values  
        m = 0.0
        s2 = float(np.sum(lamK * (w_pc**2)))
        s = math.sqrt(max(s2, 1e-12))

        # ----- PCA Normal -----
        varN, esN = var_es_normal(m, s, alpha)
        res["PCA_Normal"]["VaR"].append(varN)
        res["PCA_Normal"]["ES"].append(esN)
        res["PCA_Normal"]["mu"].append(m)
        res["PCA_Normal"]["sig"].append(s)
        res["PCA_Normal"]["nu"].append(np.nan)
        res["PCA_Normal"]["pnl"].append(pnl[t])

        # ----- PCA Student-t -----
        pnl_win = -np.dot(X_win, w.values)
        z_win = (pnl_win - m) / (s if s>0 else 1.0)
        nu_grid = np.arange(t_grid[0], t_grid[1] + 1e-9, t_grid[2])
        ll_best, nu_best = -np.inf, 6.0
        for nu_try in nu_grid:
            if nu_try <= 2.1:  # finite variance
                continue
            ll = np.sum(student_t.logpdf(z_win, df=nu_try))
            if np.isfinite(ll) and ll > ll_best:
                ll_best, nu_best = ll, nu_try
        varT, esT = var_es_student_t(m, s, alpha, nu_best)
        res["PCA_StudentT"]["VaR"].append(varT)
        res["PCA_StudentT"]["ES"].append(esT)
        res["PCA_StudentT"]["mu"].append(m)
        res["PCA_StudentT"]["sig"].append(s)
        res["PCA_StudentT"]["nu"].append(nu_best)
        res["PCA_StudentT"]["pnl"].append(pnl[t])

        # ----- Filtered Historical Simulation (EWMA scaling) -----
        var_ewma = ewma_var(pnl_win, lam=ewma_lambda)
        s_ewma = math.sqrt(max(var_ewma, 1e-12))
        resids = (pnl_win - pnl_win.mean()) / (s_ewma if s_ewma>0 else 1.0)
        pnl_synth = resids * s_ewma  
        vFHS, eFHS = hs_var_es(pnl_synth, alpha)
        res["FHS"]["VaR"].append(vFHS)
        res["FHS"]["ES"].append(eFHS)
        res["FHS"]["pnl"].append(pnl[t])

    # Build DataFrames and hits
    out = {}
    for name, d in res.items():
        df = pd.DataFrame(d, index=X.index[window:])
        df["hit"] = (df["pnl"] < df["VaR"]).astype(int)
        out[name] = df
    return out

def main():
    ap = argparse.ArgumentParser(description="IR VaR backtest on DE & US curves with PCA-Normal / PCA-t / FHS (+HS benchmark).")
    ap.add_argument("--de_file", required=True, help="CSV path for German curve (Date + maturities 1..10,15,20)")
    ap.add_argument("--us_file", required=True, help="CSV path for US curve (Date + maturities 1..10,15,20)")
    ap.add_argument("--sens", help="Optional CSV of key-rate DV01s with columns: Curve (DE/US), Maturity, KRD (per 1bp)")
    ap.add_argument("--alpha", type=float, default=0.995, help="confidence level (e.g., 0.995)")
    ap.add_argument("--window", type=int, default=60, help="rolling lookback in months")
    ap.add_argument("--pca_var", type=float, default=0.95, help="PCA cumulative variance target, e.g., 0.95")
    ap.add_argument("--lam", type=float, default=0.97, help="EWMA lambda for FHS")
    ap.add_argument("--outdir", default="outputs_ir_pca", help="output dir")
    ap.add_argument("--mats", default="1,2,3,4,5,6,7,8,9,10,15,20", help="maturities (years) comma-separated")
    ap.add_argument("--include_hs", action="store_true", help="include plain HS benchmark as extra model")
    ap.add_argument("--t_grid", default="3,20,0.5", help="nu grid for t-Student: start,end,step (e.g., '3,20,0.5')")
    args = ap.parse_args()

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    mats = [m.strip() for m in args.mats.split(",") if m.strip()]
    t_grid = tuple(float(x) for x in args.t_grid.split(","))

    de_y = _parse_curve_csv(Path(args.de_file))
    us_y = _parse_curve_csv(Path(args.us_file))

    de_y, us_y = align_maturities(de_y, us_y, mats)
    dy_de = yields_to_dy_bp(de_y)
    dy_us = yields_to_dy_bp(us_y)

    X_bp = stack_curves_bp(dy_de, dy_us, mats)

    if args.sens:
        w = read_sensitivities(Path(args.sens), mats)
    else:
        w = unit_sensitivities(mats)

    bt = rolling_models_with_pca_and_fhs(
        X_bp, w, alpha=args.alpha, window=args.window,
        pca_var=args.pca_var, ewma_lambda=args.lam,
        t_grid=t_grid, include_hs=args.include_hs
    )

    rows = []
    for name, df in bt.items():
        df.to_csv(out_dir / f"bt_{name}.csv")
        # Tests
        hits = df["hit"].to_numpy()
        pof = kupiec_pof(hits, args.alpha)
        ind = christoffersen_independence(hits)
        cc = christoffersen_cc(hits, args.alpha)
        es_stat = es_score_stat(df["pnl"].to_numpy(), df["VaR"].to_numpy(), df["ES"].to_numpy(), args.alpha)

        pit_ks_p = np.nan
        if name in ["PCA_Normal", "PCA_StudentT"]:
            mu_seq = df["mu"].to_numpy()
            sig_seq = df["sig"].to_numpy()
            if name == "PCA_Normal":
                u = pit_u_sequence(df["pnl"].to_numpy(), mu_seq, sig_seq, "normal", None)
            else:
                nu_seq = df["nu"].to_numpy()
                u = pit_u_sequence(df["pnl"].to_numpy(), mu_seq, sig_seq, "student_t", nu_seq)
            ks_res = ks_on_pit(u)
            pit_ks_p = ks_res["ks_p"]
            plot_pit_hist(u, f"{name} PIT", out_dir / f"plot_pit_{name}.png")

        plot_backtest(df.index, df["pnl"].values, df["VaR"].values, args.alpha,
                      title=f"{name}: P&L vs VaR (α={args.alpha:.3f})",
                      outpath=out_dir / f"plot_bt_{name}.png")

        rows.append({
            "model": name,
            "n": pof["n"],
            "exceptions": pof["exceptions"],
            "hit_rate": pof["hit_rate"],
            "Kupiec_p": pof["p_value"],
            "Indep_p": ind["p_value"],
            "CC_p": cc["p_value"],
            "ES_p": es_stat["p_value"],
            "PIT_KS_p": pit_ks_p
        })

    summary = pd.DataFrame(rows).sort_values("model")
    summary.to_csv(out_dir / "backtest_summary.csv", index=False)

    # Model selection 
    alpha = args.alpha; expected = 1 - alpha
    def rank_row(r):
        cc_pass = 0 if (not np.isnan(r["CC_p"]) and r["CC_p"] > 0.05) else 1
        hit_gap = abs(r["hit_rate"] - expected)
        pit_penalty = - (r["PIT_KS_p"] if not np.isnan(r["PIT_KS_p"]) else 0.0)
        return (cc_pass, hit_gap, pit_penalty, -r["ES_p"])
    chosen = min(rows, key=rank_row)["model"]

    lastX = X_bp.dropna().iloc[-args.window:].values
    final = []
    # last-window PCA stats once
    V, lamK, Xc, K, _ = pca_from_window(lastX, var_target=args.pca_var)
    w_pc = V.T @ w.values
    s2 = float(np.sum(lamK * (w_pc**2))); s = math.sqrt(max(s2, 1e-12)); m = 0.0
    # PCA_Normal
    varN, esN = var_es_normal(m, s, alpha)
    final.append({"model": "PCA_Normal", "final_VaR": varN, "final_ES": esN})
    # PCA_StudentT (re-fit nu on last window)
    pnl_win_last = -np.dot(lastX, w.values)
    z_last = (pnl_win_last - m) / (s if s>0 else 1.0)
    nu_grid = np.arange(3.0, 20.0 + 1e-9, 0.5)
    ll_best, nu_best = -np.inf, 6.0
    for nu_try in nu_grid:
        if nu_try <= 2.1: continue
        ll = np.sum(student_t.logpdf(z_last, df=nu_try))
        if np.isfinite(ll) and ll > ll_best:
            ll_best, nu_best = ll, nu_try
    varT, esT = var_es_student_t(m, s, alpha, nu_best)
    final.append({"model": "PCA_StudentT", "final_VaR": varT, "final_ES": esT})
    # FHS: use last-window EWMA vol, empirical VaR/ES on scaled residuals
    var_ewma = ewma_var(pnl_win_last, lam=args.lam)
    s_ewma = math.sqrt(max(var_ewma, 1e-12))
    resids = (pnl_win_last - pnl_win_last.mean()) / (s_ewma if s_ewma>0 else 1.0)
    pnl_synth = resids * s_ewma
    vFHS, eFHS = hs_var_es(pnl_synth, alpha)
    final.append({"model": "FHS", "final_VaR": vFHS, "final_ES": eFHS})

    pd.DataFrame(final).to_csv(out_dir / "final_static_var_es.csv", index=False)

    print(f"\nCurves aligned on {len(X_bp)} monthly obs | Window={args.window} | Alpha={args.alpha} | PCA_var={args.pca_var} | Lambda(FHS)={args.lam}")
    print(summary.to_string(index=False))
    print(f"\nSelected model: {chosen}")
    print(f"Saved CSVs and plots in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
