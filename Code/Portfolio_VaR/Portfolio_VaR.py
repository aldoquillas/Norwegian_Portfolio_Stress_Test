from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
import matplotlib.pyplot as plt


# ===================== CONFIGURATION =====================
@dataclass
class Config:
    # Simulation
    n_sims: int = 100_000
    alpha: float = 0.995       # VaR tail percentile
    seed: int | None = 42

    # t-Copula degrees of freedom (copula layer). If None -> MLE calibration
    nu_copula: float | None = None

    n_pc: int = 3

    # File paths
    date_col: str = "Date"          
    price_col: str = "PX_LAST"      
    return_col: str = "return"      

    spx_csv: str = r"C:\Coding\Github\Norweigan_Model_VaR\Input_data\SP_500.csv"
    emu_csv: str = r"C:\Coding\Github\Norweigan_Model_VaR\Input_data\MSCI_EMU.csv"
    de_yields_csv: str = r"C:\Coding\Github\Norweigan_Model_VaR\Input_data\German_Gov_Bonds.csv"
    us_yields_csv: str = r"C:\Coding\Github\Norweigan_Model_VaR\Input_data\USD_Nominal_Rates.csv"
    weights_csv: str = r"C:\Coding\Github\Norweigan_Model_VaR\Input_data\Portfolio_weights.csv"
    durations_csv: str = r"C:\Coding\Github\Norweigan_Model_VaR\Input_data\Durations.csv"

    log_returns: bool = True  
)
    yields_in_percent: bool = True

CFG = Config()

def read_equity_returns(path: str, date_col: str, price_col: str, return_col: str, log_returns: bool) -> pd.Series:
    df = pd.read_csv(path)
    if date_col in df:       
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df = df.sort_values(date_col)
    cols_lower = {c.lower(): c for c in df.columns}

    ret_c = cols_lower.get(return_col.lower())
    price_c = cols_lower.get(price_col.lower())

    if ret_c is not None:
        r = pd.Series(df[ret_c].astype(float)).dropna()
    else:
        if price_c is None:
            guess = None
            for k in ["px_last", "price", "close", "adj_close", "adj close", "px_close", "px last"]:
                if k in cols_lower:
                    guess = cols_lower[k]
                    break
            if guess is None:
                raise ValueError(f"Equity file {path}: provide a price or return column")
            price_c = guess
        p = pd.Series(df[price_c].astype(float)).dropna()
        r = np.log(p).diff().dropna() if log_returns else p.pct_change().dropna()

    r.index = pd.RangeIndex(len(r))
    return r

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    return df2

def read_yield_curve_four(path: str, date_col: str, label_prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    year_cols = [c for c in df.columns if "YEAR_" in c.upper()]
    date_cols = [c for c in df.columns if c.upper().startswith("DATE")]
    if not year_cols or not date_cols:
        raise ValueError(f"File {path} must contain 'Date' and 'YEAR_x' columns")

    dates = pd.to_datetime(df[date_cols[0]], dayfirst=True, errors="coerce")
    y = pd.DataFrame({"Date": dates})

    for c in year_cols:
        try:
            mat = int(c.split("_")[1])  
        except Exception:
            continue
        y[f"{label_prefix}_{mat}"] = pd.to_numeric(df[c], errors="coerce")

    # Keep Maturities 1,5,10,20
    keep_cols = [f"{label_prefix}_{m}" for m in [1, 5, 10, 20] if f"{label_prefix}_{m}" in y.columns]
    if not keep_cols:
        raise ValueError(f"File {path}: maturities 1/5/10/20 not found after parsing.")
    out = y["Date"].to_frame()
    for m in [1, 5, 10, 20]:
        col = f"{label_prefix}_{m}"
        if col in y.columns:
            out[col] = y[col]
    out = out.dropna().sort_values("Date")
    return out

def fit_student_t(x: np.ndarray) -> Tuple[float, float, float]:
    nu, loc, scale = stats.t.fit(x)
    nu = float(max(nu, 2.1))
    scale = float(max(scale, 1e-10))
    return nu, float(loc), scale


class EmpiricalPPF:
    def __init__(self, samples: np.ndarray):
        self.x = np.sort(samples)
        n = self.x.size
        self.u = (np.arange(1, n + 1) - 0.5) / n

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return np.interp(u, self.u, self.x)


# ===================== PCA MODEL =====================
@dataclass
class PCAModel:
    mean: np.ndarray
    loadings: np.ndarray   
    pc_scores: np.ndarray  
    t_params: List[Tuple[float, float, float]]  


def fit_pca_t(Y: np.ndarray, n_pc: int) -> PCAModel:
    Yc = Y - Y.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    loadings = Vt.T[:, :n_pc]            
    scores = U[:, :n_pc] * S[:n_pc]      
    t_params = [fit_student_t(scores[:, j]) for j in range(n_pc)]
    return PCAModel(mean=Y.mean(axis=0), loadings=loadings, pc_scores=scores, t_params=t_params)

def reconstruct_yields_from_pc(model: PCAModel, pc_draws: np.ndarray) -> np.ndarray:
    Yc = pc_draws @ model.loadings.T
    return model.mean + Yc


# ============ Copula ============
def pit_to_gauss(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 1e-10, 1 - 1e-10)
    return stats.norm.ppf(u)

def gauss_to_pit(z: np.ndarray) -> np.ndarray:
    return stats.norm.cdf(z)

def make_copula_training_matrix(
    r_spx: np.ndarray,
    r_emu: np.ndarray,
    dY: np.ndarray,
    pca: PCAModel,
) -> np.ndarray:

    T = min(len(r_spx), len(r_emu), dY.shape[0], pca.pc_scores.shape[0])
    r_spx = r_spx[-T:]
    r_emu = r_emu[-T:]
    pcs = pca.pc_scores[-T:, :]

    nu_spx, loc_spx, scl_spx = fit_student_t(r_spx)

    u_spx = stats.t.cdf(r_spx, df=nu_spx, loc=loc_spx, scale=scl_spx)
    ranks = stats.rankdata(r_emu, method="average")
    u_emu = (ranks - 0.5) / len(r_emu)

    u_pcs = []
    for j in range(pca.loadings.shape[1]):
        nu, loc, scl = fit_student_t(pcs[:, j])
        u_pcs.append(stats.t.cdf(pcs[:, j], df=nu, loc=loc, scale=scl))

    U = np.column_stack([u_spx, u_emu] + u_pcs)
    Z = pit_to_gauss(U)
    return Z


def estimate_copula_corr(Z: np.ndarray) -> np.ndarray:
    R = np.corrcoef(Z.T)
    eps = 1e-6
    R = (1 - eps) * R + eps * np.eye(R.shape[0])  
    return R


def t_copula_loglike(Z: np.ndarray, R: np.ndarray, nu: float) -> float:
    n, d = Z.shape
    c, lower = cho_factor(R, check_finite=False)
    logdet = 2.0 * np.sum(np.log(np.diag(c)))
    sol = cho_solve((c, lower), Z.T, check_finite=False)   # shape (d, n)
    q = np.sum(Z.T * sol, axis=0)                          # length n

    ll = -0.5 * n * logdet
    ll += n * (gammaln((nu + d) / 2) - gammaln(nu / 2))
    ll -= 0.5 * n * (d * np.log(nu * np.pi))
    ll -= 0.5 * (nu + d) * np.sum(np.log1p(q / nu))
    return float(ll)


def estimate_nu_copula(Z: np.ndarray, R: np.ndarray) -> float:
    def neg_ll(nu):
        nu = float(nu)
        if nu <= 2.1:
            return 1e12
        return -t_copula_loglike(Z, R, nu)

    res = minimize_scalar(
        neg_ll, bounds=(2.1, 50.0), method="bounded", options={"xatol": 1e-3}
    )
    return float(res.x)


# ===================== Simulation =====================
def rmultivariate_t(n: int, R: np.ndarray, nu: float, rng: np.random.Generator) -> np.ndarray:
    d = R.shape[0]
    g = rng.standard_normal(size=(n, d))
    L = np.linalg.cholesky(R)
    z = g @ L.T
    w = rng.chisquare(df=nu, size=n) / nu
    return z / np.sqrt(w)[:, None]

def simulate_scenarios(
    cfg: Config,
    r_spx_hist: np.ndarray,
    r_emu_hist: np.ndarray,
    dY_hist: np.ndarray,
    pca: PCAModel,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Tuple[float, float, float]], float]:
    rng = np.random.default_rng(cfg.seed)

    # Fit marginals 
    nu_spx, loc_spx, scl_spx = fit_student_t(r_spx_hist)
    emu_emp = EmpiricalPPF(r_emu_hist)

    # Build copula training Z and correlation
    Zhist = make_copula_training_matrix(r_spx_hist, r_emu_hist, dY_hist, pca)
    R = estimate_copula_corr(Zhist)

    # Calibrate ν 
    nu_cop = cfg.nu_copula if cfg.nu_copula is not None else estimate_nu_copula(Zhist, R)

    # Simulate latent t and map to uniforms (Φ)
    Tdraws = rmultivariate_t(cfg.n_sims, R, nu_cop, rng)
    U = gauss_to_pit(Tdraws)

    # Map U -> marginals
    u_spx = U[:, 0]
    u_emu = U[:, 1]
    u_pcs = U[:, 2:2 + cfg.n_pc]

    spx_sim = stats.t.ppf(u_spx, df=nu_spx, loc=loc_spx, scale=scl_spx)
    emu_sim = emu_emp.ppf(u_emu)

    pc_draws = np.empty((cfg.n_sims, cfg.n_pc))
    for j in range(cfg.n_pc):
        nu, loc, scl = pca.t_params[j]
        pc_draws[:, j] = stats.t.ppf(u_pcs[:, j], df=nu, loc=loc, scale=scl)

    # Reconstruct yields variation
    dY_sim = reconstruct_yields_from_pc(pca, pc_draws)

    cols = ["SPX", "EMU", "DE_1", "DE_5", "DE_10", "DE_20", "US_1", "US_5", "US_10", "US_20"]
    factors = np.column_stack([spx_sim, emu_sim, dY_sim])
    sim_df = pd.DataFrame(factors, columns=cols)

    t_params = {"SPX": (nu_spx, loc_spx, scl_spx)}
    return sim_df, R, Zhist, t_params, nu_cop


# ===================== Weights & durations =====================
def parse_weights(path: str) -> Tuple[float, float, Dict[str, float]]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if df.shape[1] < 2:
        raise ValueError("Weights CSV must have at least two columns: Instrument, Weight")
    inst_col = df.columns[0]
    w_col = df.columns[1]

    def to_num(x):
        if isinstance(x, str) and x.strip().endswith('%'):
            return float(x.strip().replace('%', '')) / 100.0
        return float(x)

    w_spx = w_emu = 0.0
    w_bonds: Dict[str, float] = {k: 0.0 for k in
                                 ["DE_1", "DE_5", "DE_10", "DE_20", "US_1", "US_5", "US_10", "US_20"]}

    for _, row in df.iterrows():
        name = str(row[inst_col]).strip().lower()
        w = to_num(row[w_col])
        if "msci" in name and ("emu" in name or "meu" in name):
            w_emu += w
        elif "s&p" in name or "sp 500" in name or "s&p 500" in name or "s&p500" in name or name.startswith("s&p"):
            w_spx += w
        else:
            if "usa" in name or "us " in name or name.startswith("usa") or name.startswith("us") or "american" in name:
                curve = "US"
            elif "german" in name or name.startswith("german") or name == "de":
                curve = "DE"
            else:
                continue
            mat = None
            for m in ["1", "5", "10", "20"]:
                if f" {m} " in f" {name} ":
                    mat = m
                    break
            if mat is None:
                continue
            key = f"{curve}_{mat}"
            if key in w_bonds:
                w_bonds[key] += w

    total = w_spx + w_emu + sum(w_bonds.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1. Found {total:.6f}")
    return w_spx, w_emu, w_bonds


def parse_durations(path: str) -> Dict[str, float]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(' ', '').replace('\t', '') for c in df.columns]
    dur_col_candidates = [c for c in df.columns if 'modified' in c and 'duration' in c]
    if not dur_col_candidates:
        raise ValueError("Durations CSV: couldn't find 'Modified Duration' column")
    dur_col = dur_col_candidates[0]
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        curve = str(row[df.columns[0]]).strip().upper()  # 'DE' or 'US'
        maturity = str(row[df.columns[1]]).strip()
        if maturity.endswith('Y'):
            maturity = maturity[:-1]
        if maturity not in {"1", "5", "10", "20"}:
            continue
        key = f"{curve}_{maturity}"
        out[key] = float(row[dur_col])

    for k, v in {"DE_1": 0.9, "DE_5": 4.5, "DE_10": 8.5, "DE_20": 12.0,
                 "US_1": 1.0, "US_5": 4.5, "US_10": 8.5, "US_20": 12.0}.items():
        out.setdefault(k, v)
    return out


# ===================== P&L, VaR/ES, VaR scenario =====================
def compute_pnl_pct(sim_df: pd.DataFrame,
                    w_spx: float, w_emu: float,
                    w_bonds: Dict[str, float],
                    durations: Dict[str, float],
                    yields_in_percent: bool = True) -> np.ndarray:

    pnl_eq = w_spx * sim_df["SPX"].to_numpy() + w_emu * sim_df["EMU"].to_numpy()

    scale = 0.01 if yields_in_percent else 1.0 
    pnl_rates = np.zeros(sim_df.shape[0])
    for k, w in w_bonds.items():
        if k not in sim_df.columns or w == 0.0:
            continue
        Dmod = durations.get(k, 0.0)
        dy = sim_df[k].to_numpy() * scale
        pnl_rates += w * (-Dmod * dy)

    return pnl_eq + pnl_rates


def var_es(pnl: np.ndarray, alpha: float) -> Tuple[float, float, int]:
    idx = np.argsort(pnl)
    pnl_sorted = pnl[idx]
    n = len(pnl)
    k = max(int(np.floor((1 - alpha) * n)) - 1, 0)
    var = pnl_sorted[k]
    es = pnl_sorted[:k + 1].mean() if k >= 0 else pnl_sorted[0]
    return float(var), float(es), int(idx[k])


# ===================== Correlation Plots =====================
def save_heatmap(df: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(df.values, interpolation='nearest', aspect='equal')
    ax.set_title(title)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.4)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.values[i, j]:.2f}", va='center', ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===================== MAIN =====================
def main(cfg: Config = CFG):
    script_dir = Path(__file__).resolve().parent
    outdir = script_dir / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing outputs to: {outdir}")

    print("[INFO] Loading equities…")
    r_spx = read_equity_returns(cfg.spx_csv, cfg.date_col, cfg.price_col, cfg.return_col, cfg.log_returns).to_numpy()
    r_emu = read_equity_returns(cfg.emu_csv, cfg.date_col, cfg.price_col, cfg.return_col, cfg.log_returns).to_numpy()

    print("[INFO] Loading yields (DE & US)…")
    de_lvls = read_yield_curve_four(cfg.de_yields_csv, cfg.date_col, "DE")
    us_lvls = read_yield_curve_four(cfg.us_yields_csv, cfg.date_col, "US")

    y = pd.merge_asof(
        de_lvls.sort_values("Date"), us_lvls.sort_values("Date"),
        on="Date"
    ).dropna()
    cols = ["DE_1", "DE_5", "DE_10", "DE_20", "US_1", "US_5", "US_10", "US_20"]
    dY = y[cols].diff().dropna().to_numpy()

    print("[INFO] Fitting PCA on Δy…")
    pca = fit_pca_t(dY, cfg.n_pc)

    print("[INFO] Calibrating copula and simulating scenarios…")
    sim_df, R, Zhist, t_params, nu_cop = simulate_scenarios(cfg, r_spx, r_emu, dY, pca)

    print("[INFO] Parsing portfolio weights and durations…")
    w_spx, w_emu, w_bonds = parse_weights(cfg.weights_csv)
    durations = parse_durations(cfg.durations_csv)

    print("[INFO] Computing portfolio P&L…")
    pnl = compute_pnl_pct(sim_df, w_spx, w_emu, w_bonds, durations,
                          yields_in_percent=cfg.yields_in_percent)

    var995, es995, idx_var = var_es(pnl, cfg.alpha)
    var_scn = sim_df.iloc[idx_var].to_frame(name="shock").T

    print("[INFO] Saving CSV outputs…")
    R_labels = ["Z_SPX", "Z_EMU"] + [f"Z_PC{j+1}" for j in range(cfg.n_pc)]
    R_df = pd.DataFrame(R, columns=R_labels, index=R_labels)
    R_df.to_csv(outdir / "copula_latent_corr.csv", index=True)
    sim_df.to_csv(outdir / "simulated_factor_shocks.csv", index=False)
    var_scn.to_csv(outdir / "var_99_5_scenario_shocks.csv", index=False)
    
    hist = pd.DataFrame({
        "SPX": r_spx[-dY.shape[0]:],
        "EMU": r_emu[-dY.shape[0]:],
    })
    for i, c in enumerate(["DE_1", "DE_5", "DE_10", "DE_20", "US_1", "US_5", "US_10", "US_20"]):
        hist[c] = dY[:, i]
    hist_corr = hist.corr()
    hist_corr.to_csv(outdir / "historical_raw_correlation.csv")

    print("[INFO] Saving heatmap plots…")
    save_heatmap(hist_corr, outdir / "historical_corr_heatmap.png",
                 "Historical Correlation (Equity returns & Yield changes)")
    save_heatmap(R_df, outdir / "copula_latent_corr_heatmap.png",
                 "Copula Latent Correlation (Z-layer)")

    results_path = outdir / "Portfolio_VaR_Results.csv"
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "alpha": cfg.alpha,
        'n_sims': cfg.n_sims,
        "n_pc": cfg.n_pc,
        "nu_df": nu_cop,
        "VaR_pct": var995 * 100.0,
        "ES_pct": es995 * 100.0,
    }])
    if results_path.exists():
        row.to_csv(results_path, mode='a', header=False, index=False)
    else:
        row.to_csv(results_path, index=False)

    print("\n=== Portfolio Risk Summary (percentage P&L) ===")
    print(f"Simulations: {cfg.n_sims:,}")
    print(f"t-Copula ν (df): {nu_cop:.3f} — smaller ν ⇒ heavier joint tails (stronger extreme co-moves)")
    print(f"VaR 99.5%: {var995:.4%}")
    print(f"ES  99.5%: {es995:.4%}")
    print("\nVaR scenario shocks (apply to positions):")
    print(var_scn.to_string(index=False))

    print("\nFiles written:")
    print(f" - {outdir / 'copula_latent_corr.csv'} (R used by t-Copula)")
    print(f" - {outdir / 'copula_latent_corr_heatmap.png'} (latent R heatmap)")
    print(f" - {outdir / 'historical_raw_correlation.csv'} (reference)")
    print(f" - {outdir / 'historical_corr_heatmap.png'} (historical corr heatmap)")
    print(f" - {outdir / 'simulated_factor_shocks.csv'} (all scenarios)")
    print(f" - {outdir / 'var_99_5_scenario_shocks.csv'} (one row)")
    print(f" - {outdir / 'Portfolio_VaR_Results.csv'} (append-only run log)")
    print("\n[INFO] Done.")


if __name__ == "__main__":
    print("[INFO] Starting simulation…")
    main()
