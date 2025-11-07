# ============================ VaR/ES Backtest Engine (HS / StudentT / EWMA) ============================

# ---------------------------- Imports Libraries ----------------------------
from __future__ import annotations
import argparse, math
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

from scipy.stats import chi2, norm, t, jarque_bera, kstest
import matplotlib.pyplot as plt

# ---------------------------- Core helpers ----------------------------
def _z(alpha: float) -> float:
    return float(norm.ppf(alpha)) # Standard normal quantile

def _as_1d(x) -> np.ndarray:
    r = pd.Series(x).dropna().values
    if r.ndim != 1 or r.size == 0:
        raise ValueError("Input returns must be a non-empty 1-D array/Series.")
    return r.astype(float, copy=False) # 1D array of float & no NaN

def _quantile_lower(a: np.ndarray, q: float) -> float: # order statistic quantile (no interpolation) - actual observed loss
    try:
        return float(np.quantile(a, q, method="lower"))  # NumPy >= 1.22
    except TypeError:
        return float(np.quantile(a, q, interpolation="lower"))  # older NumPy

# ---------------------------- Models ----------------------------

# Historical Simulation (Nonparametric)
def hs_var(returns, alpha=0.995) -> float:
    r = _as_1d(returns)
    q = _quantile_lower(r, 1 - alpha)
    return float(q)

def hs_es(returns, alpha=0.995) -> float:
    r = _as_1d(returns)
    q = _quantile_lower(r, 1 - alpha)
    tail = r[r <= q]
    return float(tail.mean())

# Student-t
def t_params(returns) -> tuple[float, float, float]:
    r = _as_1d(returns)
    nu, mu, s = t.fit(r)  # (df, loc, scale)
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Scale must be positive for Student-t VaR/ES.")
    if not np.isfinite(nu) or nu <= 2:
        nu = 2.001  # ensure finite ES & variance
    return float(nu), float(mu), float(s)

def studentt_var(returns, alpha: float = 0.99) -> float:
    nu, mu, s = t_params(returns)
    q = t.ppf(alpha, df=nu)            # right tail
    return -(mu + s * q)               # negative threshold

def studentt_es(returns, alpha: float = 0.99) -> float:
    nu, mu, s = t_params(returns)
    q = t.ppf(alpha, df=nu)            # right tail
    pdf_q = t.pdf(q, df=nu)
    es_std = ((nu + q*q) / (nu - 1)) * (pdf_q / (1 - alpha))
    return -(mu + s * es_std)

# EWMA (Normal innovations)
def ewma_vol(returns, lam=0.97) -> float:
    if not (0 < lam < 1):
        raise ValueError("lam must be in (0,1) for EWMA.")
    r = _as_1d(returns)
    v = 0.0
    for x in r:
        v = lam * v + (1 - lam) * x * x
    s = float(np.sqrt(v))
    if s <= 0 or not np.isfinite(s):
        raise ValueError("EWMA variance degenerated to non-positive.")
    return s # Conditional volatility

def ewma_var(returns, alpha=0.99, lam=0.97) -> float:
    r = _as_1d(returns)
    mu = float(np.mean(r))
    s = ewma_vol(r, lam)
    return -(mu + s * _z(alpha)) # VaR under Normal assumption

def ewma_es(returns, alpha=0.99, lam=0.97) -> float:
    r = _as_1d(returns)
    mu = float(np.mean(r))
    s = ewma_vol(r, lam)
    z = _z(alpha)
    return -(mu + s * (norm.pdf(z) / (1 - alpha))) # ES under Normal assumption

# ---------------------------- Backtesting & GoF statistics ----------------------------

def kupiec_pof(hits: np.ndarray, alpha: float) -> dict:
    hits = np.asarray(hits, dtype=int)
    x = int(hits.sum())
    n = int(hits.size)
    p = 1 - alpha
    # guarded logs
    t1 = 0.0 if x == 0 else x * math.log(x / n)
    t2 = 0.0 if (n - x) == 0 else (n - x) * math.log((n - x) / n)
    lr = 2 * (t1 + t2 - (x * math.log(p) + (n - x) * math.log(1 - p)))
    pval = 1 - chi2.cdf(lr, df=1)
    return {"LR_pof": float(lr), "p_value": float(pval), "exceptions": x, "n": n, "hit_rate": x / n} # Proportion of failures

def christoffersen_independence(hits: np.ndarray) -> Dict[str, float]:
    h = np.asarray(hits, dtype=int)
    if h.size < 2:
        return {"LR_ind": np.nan, "p_value": np.nan}
    n00 = n01 = n10 = n11 = 0
    for i in range(1, h.size):
        a, b = int(h[i - 1]), int(h[i])
        if a == 0 and b == 0: n00 += 1
        if a == 0 and b == 1: n01 += 1
        if a == 1 and b == 0: n10 += 1
        if a == 1 and b == 1: n11 += 1
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0:
        return {"LR_ind": np.nan, "p_value": np.nan}
    p01, p11 = n01 / n0, n11 / n1
    p = (n01 + n11) / (n0 + n1)
    ll_re = n00 * math.log(1 - p) + n01 * math.log(p) + n10 * math.log(1 - p) + n11 * math.log(p)
    ll_un = 0.0
    if n00: ll_un += n00 * math.log(1 - p01)
    if n01: ll_un += n01 * math.log(p01)
    if n10: ll_un += n10 * math.log(1 - p11)
    if n11: ll_un += n11 * math.log(p11)
    lr = -2 * (ll_re - ll_un)
    pval = float(1 - chi2.cdf(lr, df=1))
    return {"LR_ind": float(lr), "p_value": pval} # Exception Cluster Tests

def christoffersen_cc(hits: np.ndarray, alpha: float) -> Dict[str, float]:
    pof = kupiec_pof(hits, alpha)
    ind = christoffersen_independence(hits)
    if np.isnan(ind["p_value"]):
        return {"LR_cc": np.nan, "p_value": np.nan}
    lr_cc = pof["LR_pof"] + ind["LR_ind"]
    pval = float(1 - chi2.cdf(lr_cc, df=2))
    return {"LR_cc": float(lr_cc), "p_value": pval} # Conditional Coverage

def es_score_stat(returns: np.ndarray, var_f: np.ndarray, es_f: np.ndarray, alpha: float) -> Dict[str, float]:
    r = np.asarray(returns, float)
    v = np.asarray(var_f, float)
    e = np.asarray(es_f, float)
    I = (r < v).astype(float)  # VaR negative; exceedance if r < VaR
    S = (I * (v - r) / alpha) - (e - r)
    m = float(S.mean())
    se = float(S.std(ddof=1)) / math.sqrt(S.size) if S.size > 1 else np.inf
    t_stat = m / se if se > 0 else np.inf
    p = 2 * (1 - norm.cdf(abs(t_stat)))
    return {"t_stat": float(t_stat), "p_value": float(p), "mean_score": m}

def gof_normal_std(r: np.ndarray) -> Dict[str, float]:
    mu, s = float(np.mean(r)), float(np.std(r, ddof=1))
    if s <= 0 or not np.isfinite(s):
        return {"JB_p": np.nan, "KS_p": np.nan}
    std = (r - mu) / s
    jb_stat, jb_p = jarque_bera(std)
    ks_stat, ks_p = kstest(std, 'norm')
    return {"JB_p": float(jb_p), "KS_p": float(ks_p)}

def gof_ewma_std(r: np.ndarray, lam: float = 0.97) -> Dict[str, float]:
    v = 0.0
    sig = np.zeros_like(r)
    for i, x in enumerate(r):
        v = lam * v + (1 - lam) * x * x
        sig[i] = math.sqrt(v) if v > 0 else 0.0
    std = np.divide(r, sig, out=np.zeros_like(r), where=sig > 0)
    std = std[sig > 0]
    if std.size < 10:
        return {"JB_p": np.nan, "KS_p": np.nan}
    jb_stat, jb_p = jarque_bera(std)
    ks_stat, ks_p = kstest(std, 'norm')
    return {"JB_p": float(jb_p), "KS_p": float(ks_p)}

# Student-t GoF (fit + KS against fitted t)
def gof_student_fit(r: np.ndarray) -> Dict[str, float]:
    r = _as_1d(r)
    nu, mu, s = t.fit(r)
    z = (r - mu) / s
    ks_stat, ks_p = kstest(z, 't', args=(nu,))
    return {"nu": float(nu), "mu": float(mu), "s": float(s), "KS_p": float(ks_p)}

# ---------------------------- Plotting ----------------------------

def _hist_with_pdf(ax, r: np.ndarray, pdf_x: np.ndarray | None = None, pdf_y: np.ndarray | None = None,
                   bins: int = 40, title: str = ""):
    r = _as_1d(r)
    ax.hist(r, bins=bins, density=True, alpha=0.5, label="Empirical")
    if pdf_x is not None and pdf_y is not None:
        ax.plot(pdf_x, pdf_y, linewidth=2, label="Fitted PDF")
    ax.set_title(title)
    ax.legend()

def _qqplot(ax, sample: np.ndarray, theo_q: np.ndarray, xlabel: str, ylabel: str, title: str):
    sr = np.sort(sample)
    ax.scatter(theo_q, sr, s=9)
    lo, hi = np.percentile(theo_q, [5, 95])
    rs = np.interp([lo, hi], theo_q, sr)
    ax.plot([lo, hi], rs, linewidth=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)

def plot_gof_student(r_insample: np.ndarray, outpath: Path):
    fit = gof_student_fit(r_insample)
    mu, s, nu = fit["mu"], fit["s"], fit["nu"]
    xs = np.linspace(np.min(r_insample), np.max(r_insample), 600)
    pdf_y = t.pdf((xs - mu) / s, df=nu) / s

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    _hist_with_pdf(ax1, r_insample, xs, pdf_y, title=f"Student-t PDF fit (ν={nu:.2f})")
    ax2 = fig.add_subplot(1, 2, 2)
    # QQ vs fitted t (work with standardized residuals)
    z = (r_insample - mu) / s
    n = z.size
    p = (np.arange(1, n + 1) - 0.5) / n
    q_theo = t.ppf(p, df=nu)
    _qqplot(ax2, z, q_theo, "Theoretical t-quantiles", "Standardized sample", "QQ vs fitted t")
    fig.suptitle(f"Student-t GoF (KS p={fit['KS_p']:.3f})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_gof_ewma(r_insample: np.ndarray, lam: float, outpath: Path):
    # Standardize by EWMA sigma
    v = 0.0
    sig = np.zeros_like(r_insample)
    for i, x in enumerate(r_insample):
        v = lam * v + (1 - lam) * x * x
        sig[i] = math.sqrt(v) if v > 0 else 0.0
    std = np.divide(r_insample, sig, out=np.zeros_like(r_insample), where=sig > 0)
    std = std[sig > 0]
    if std.size < 10:
        std = r_insample  # fallback: unstandardized (plot will still render)

    xs = np.linspace(np.min(std), np.max(std), 600)
    pdf_y = norm.pdf(xs)

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    _hist_with_pdf(ax1, std, xs, pdf_y, title="EWMA-standardized ~ N(0,1) fit")
    ax2 = fig.add_subplot(1, 2, 2)
    n = std.size
    p = (np.arange(1, n + 1) - 0.5) / n
    q_theo = norm.ppf(p)
    _qqplot(ax2, std, q_theo, "Theoretical Normal quantiles", "EWMA-std. sample", "QQ vs Normal")
    fig.suptitle("EWMA GoF")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_gof_hs(r_insample: np.ndarray, alpha: float, outpath: Path):
    # Nonparametric: histogram + ECDF with VaR/ES markers
    r = _as_1d(r_insample)
    var_hs = hs_var(r, alpha)
    es_hs  = hs_es(r, alpha)
    # ECDF
    sr = np.sort(r)
    y = np.arange(1, sr.size + 1) / sr.size

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(r, bins=40, density=True, alpha=0.6, label="Empirical")
    ax1.axvline(var_hs, color="k", linestyle="--", label=f"VaR (α={alpha:.3f})")
    ax1.axvline(es_hs, color="k", linestyle=":", label="ES")
    ax1.set_title("HS: Histogram with VaR/ES"); ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sr, y, drawstyle="steps-post")
    ax2.axvline(var_hs, color="k", linestyle="--")
    ax2.axvline(es_hs, color="k", linestyle=":")
    ax2.set_xlabel("Return"); ax2.set_ylabel("ECDF")
    ax2.set_title("HS: Empirical CDF with VaR/ES")
    fig.suptitle("Historical Simulation GoF (nonparametric)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_backtest(df_bt: pd.DataFrame, alpha: float, title: str, outpath: Path):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df_bt.index, df_bt["r"].values, linewidth=1, label="Returns")
    ax.plot(df_bt.index, df_bt["VaR"].values, linewidth=2, label=f"VaR {alpha:.3f}")
    # Mark exceptions
    breaches = df_bt[df_bt["hit"] == 1]
    if not breaches.empty:
        ax.scatter(breaches.index, breaches["r"].values, marker="o", s=30, label="Exceptions")
    ax.set_title(title)
    ax.set_xlabel("Date"); ax.set_ylabel("Monthly log-return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------- Rolling engine ----------------------------

def rolling_var_es(returns: pd.Series, window: int, alpha: float, model: str, lam: float = 0.97) -> pd.DataFrame:
    r = pd.Series(returns).dropna().copy()
    var_f, es_f, idx = [], [], []
    for t_ix in range(window, len(r)):
        hist = r.iloc[t_ix - window : t_ix].values
        if model == "HS":
            v = hs_var(hist, alpha);            e = hs_es(hist, alpha)
        elif model == "StudentT":
            v = studentt_var(hist, alpha);      e = studentt_es(hist, alpha)
        elif model == "EWMA":
            v = ewma_var(hist, alpha, lam=lam); e = ewma_es(hist, alpha, lam=lam)
        else:
            raise ValueError(f"Unknown model: {model}")
        var_f.append(v); es_f.append(e); idx.append(r.index[t_ix])
    out = pd.DataFrame({"r": r.loc[idx].values, "VaR": var_f, "ES": es_f}, index=idx)
    out["hit"] = (out["r"] < out["VaR"]).astype(int)
    return out

# ---------------------------- Data IO ----------------------------

def read_price_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
        df.index.name = "Date"
    # Price
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        raise ValueError(f"No numeric price column found in {csv_path}. Columns: {df.columns.tolist()}")
    price = df[num_cols[0]].astype(float).sort_index()
    return price

def to_monthly_log_returns(price: pd.Series) -> pd.Series:
    price_m = price.resample("M").last()  # month-end
    return np.log(price_m / price_m.shift(1)).dropna()

# ---------------------------- Model selection ----------------------------

def select_model(bt_results: dict, alpha: float, r_insample: np.ndarray, lam: float = 0.97) -> str:
    """
    1) Prefer CC pass (p>0.05)
    2) Among passers, hit rate closest to expected (1-alpha)
    3) Highest ES-score p-value
    4) Better GoF (EWMA vs Normal (JB+KS); StudentT KS vs fitted t; HS no param GoF)
    """
    expected = 1 - alpha
    gof = {
        "EWMA":     gof_ewma_std(r_insample, lam=lam),
        "HS":       {},
        "StudentT": gof_student_fit(r_insample)
    }
    rows = []
    for name, df in bt_results.items():
        hits = df["hit"].to_numpy()
        pof = kupiec_pof(hits, alpha)
        ind = christoffersen_independence(hits)
        cc  = christoffersen_cc(hits, alpha)
        es  = es_score_stat(df["r"].to_numpy(), df["VaR"].to_numpy(), df["ES"].to_numpy(), alpha)
        if name == "StudentT":
            gof_score = np.nan_to_num(gof["StudentT"].get("KS_p", np.nan))
        elif name == "EWMA":
            gof_score = np.nan_to_num(gof["EWMA"].get("JB_p", np.nan)) + np.nan_to_num(gof["EWMA"].get("KS_p", np.nan))
        else:
            gof_score = 0.0  # HS: nonparametric
        rows.append({
            "name": name,
            "cc_pass": (not np.isnan(cc["p_value"])) and (cc["p_value"] > 0.05),
            "hit_gap": abs(pof["hit_rate"] - expected),
            "es_p": es["p_value"],
            "gof_p": gof_score
        })
    cand = [r for r in rows if r["cc_pass"]] or rows
    cand.sort(key=lambda r: (0 if r["cc_pass"] else 1, r["hit_gap"], -r["es_p"], -r["gof_p"]))
    return cand[0]["name"]

def final_var(returns: pd.Series, model: str, alpha: float, lam: float = 0.97) -> float:
    r = returns.dropna().values
    if model == "HS":       return hs_var(r, alpha)
    if model == "StudentT": return studentt_var(r, alpha)
    if model == "EWMA":     return ewma_var(r, alpha, lam=lam)
    raise ValueError(model)

# ---------------------------- Main (CLI) ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Equity VaR backtest & selection (HS/StudentT/EWMA) on monthly returns.")
    ap.add_argument("--file", required=True, help='CSV filename in Input_data or full path (two columns: Date, Price)')
    ap.add_argument("--alpha", type=float, default=0.995, help="confidence level (default 0.995)")
    ap.add_argument("--window", type=int, default=60, help="rolling lookback window in months")
    ap.add_argument("--lam", type=float, default=0.97, help="EWMA lambda for monthly data (default 0.97)")
    ap.add_argument("--outdir", default="outputs", help="output directory relative to repo root")
    args = ap.parse_args()

    # Resolve paths relative to repo root (Modelling's parent)
    repo_root = Path(__file__).resolve().parents[1]
    in_dir = repo_root / "Input_data"
    out_dir = repo_root / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.file)
    if not csv_path.is_file():
        csv_path = in_dir / args.file
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {args.file}. Looked in {in_dir}")

    # Data
    price = read_price_series(csv_path)
    r_m = to_monthly_log_returns(price)

    if len(r_m) <= args.window + 5:
        raise ValueError(f"Not enough observations ({len(r_m)}) for window={args.window}")

    # Backtests
    models = ["HS", "StudentT", "EWMA"]
    bt = {m: rolling_var_es(r_m, args.window, args.alpha, m, lam=args.lam) for m in models}

    # Summarize KPIs
    rows = []
    for name, df in bt.items():
        hits = df["hit"].to_numpy()
        pof = kupiec_pof(hits, args.alpha)
        ind = christoffersen_independence(hits)
        cc  = christoffersen_cc(hits, args.alpha)
        es  = es_score_stat(df["r"].to_numpy(), df["VaR"].to_numpy(), df["ES"].to_numpy(), args.alpha)
        rows.append({
            "model": name,
            "n": pof["n"],
            "exceptions": pof["exceptions"],
            "hit_rate": pof["hit_rate"],
            "Kupiec_p": pof["p_value"],
            "Indep_p": ind["p_value"],
            "CC_p": cc["p_value"],
            "ES_p": es["p_value"],
        })
        # Persist rolling series
        df.to_csv(out_dir / f"bt_{name}.csv", index=True)
        # Backtest plot
        plot_backtest(df, args.alpha, title=f"{name}: Monthly returns vs VaR (α={args.alpha:.3f})",
                      outpath=out_dir / f"plot_bt_{name}.png")

    summary = pd.DataFrame(rows).sort_values("model")
    summary.to_csv(out_dir / "backtest_summary.csv", index=False)

    # GoF plots on *in-sample* chunk (first window) for fairness
    r_in = r_m.iloc[:args.window].to_numpy()
    plot_gof_hs(r_in, args.alpha, out_dir / "plot_gof_HS.png")
    plot_gof_student(r_in, out_dir / "plot_gof_StudentT.png")
    plot_gof_ewma(r_in, args.lam, out_dir / "plot_gof_EWMA.png")

    # Select model
    chosen = select_model(bt, args.alpha, r_in, lam=args.lam)

    # Final VaR on full sample
    var_final = final_var(r_m, chosen, args.alpha, lam=args.lam)

    print(f'\nFile: {csv_path.name} | Alpha={args.alpha:.3f} | Window={args.window}m | Lambda={args.lam}')
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary.to_string(index=False))
    expected = 1 - args.alpha
    print(f"\nDecision: prefer CC_p>0.05; then hit_rate closest to {expected:.3%}; then highest ES_p; then GoF.")
    print(f"Selected model: {chosen}")
    print(f"Final monthly VaR (alpha={args.alpha:.3f}): {var_final:.6f}")

    pd.DataFrame([{
        "file": csv_path.name, "alpha": args.alpha, "window": args.window, "lambda": args.lam,
        "selected_model": chosen, "final_monthly_VaR": var_final
    }]).to_csv(out_dir / "final_selection.csv", index=False)

if __name__ == "__main__":
    main()
