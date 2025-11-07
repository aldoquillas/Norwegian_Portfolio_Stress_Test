import re, warnings
import numpy as np
import pandas as pd
from numpy.linalg import svd

# ---------- helpers ----------
def _parse_curve_csv(path: str) -> pd.DataFrame:
    """Read curve CSV with a Date column and maturity columns.
       Keep only numeric maturity columns; strip names to year digits.
       Resample to month-end and fill small gaps.
    """
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.set_index("Date").sort_index()

    # keep only numeric columns whose name contains a number
    cols = []
    for c in df.columns:
        m = re.search(r"(\d+)", str(c))
        if m and pd.api.types.is_numeric_dtype(df[c]):
            cols.append((c, m.group(1)))  # (raw name, year digits)
        elif m:
            # try coerce to numeric
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                df[c] = s
                cols.append((c, m.group(1)))

    if not cols:
        raise ValueError(f"No numeric maturity columns found in {path}")

    out = pd.DataFrame(index=df.index)
    for raw, yr in cols:
        out[yr] = pd.to_numeric(df[raw], errors="coerce")

    # month-end and fill
    out = out.resample("ME").last()
    out = out.ffill().bfill()
    return out

def _detect_percent_units(yld: pd.DataFrame) -> bool:
    """True if decimals (0.025), False if percent (2.5)."""
    s = yld.stack().dropna()
    if s.empty: return True
    return s.median() < 1.0

def yields_to_dy_bp(yld: pd.DataFrame) -> pd.DataFrame:
    dec = _detect_percent_units(yld)
    if dec:
        dy = (yld - yld.shift(1)) * 10000.0
    else:
        dy = (yld - yld.shift(1)) * 100.0
    return dy.dropna(how="all").dropna()

# ---------- load, align, PCA ----------
# adjust paths if needed
de = _parse_curve_csv(r'.\Input_data\German_Gov_Bonds.csv')
us = _parse_curve_csv(r'.\Input_data\USD_Nominal_Rates.csv')

# align maturities and dates to intersection
mats = sorted(set(de.columns).intersection(us.columns), key=lambda x: int(x))
if not mats:
    raise ValueError("No common maturities between DE and US.")
de = de[mats]
us = us[mats]
common_idx = de.index.intersection(us.index)
de = de.loc[common_idx]
us = us.loc[common_idx]

# convert to monthly bp changes
dy_de = yields_to_dy_bp(de)
dy_us = yields_to_dy_bp(us)
common_idx = dy_de.index.intersection(dy_us.index)
dy_de = dy_de.loc[common_idx]
dy_us = dy_us.loc[common_idx]

# stack features: [DE_*, US_*]
X = pd.DataFrame(index=common_idx)
for m in mats:
    X[f"DE_{m}"] = dy_de[m].values
for m in mats:
    X[f"US_{m}"] = dy_us[m].values

# ensure numeric
X = X.apply(pd.to_numeric, errors="coerce").dropna()

# PCA on centered data
Xc = X - X.mean(0)
U, S, Vt = svd(Xc.values, full_matrices=False)
lam = (S**2) / (len(Xc) - 1)
expl = lam / lam.sum()
cum = np.cumsum(expl)

print(f"Samples: {len(Xc)}, Dimensions: {Xc.shape[1]}")
print("Variance explained by PCs:")
k95 = None
for i, (v, c) in enumerate(zip(expl, cum), 1):
    print(f"  PC{i:02d}: {v*100:6.2f}%   (cumulative {c*100:6.2f}%)")
    if k95 is None and c >= 0.95:
        k95 = i
print(f"\n=> Number of PCs to reach 95% variance: {k95}")
