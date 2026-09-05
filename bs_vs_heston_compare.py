# ============================================================
# Black–Scholes vs Heston on the SAME single-option time-series
# - Loads your single-option CSV + SPX + VIX/VIX9D
# - Prices each day with:
#     (1) Black–Scholes (sigma from VIX or constant)
#     (2) Heston (semi-analytical Fourier pricing)
# - Produces ONE clean, poster-ready comparison plot
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import exp, sqrt, log
from scipy.stats import norm


# ---------------------- USER INPUTS ---------------------- #
OPTIONS_CSV = Path(r"Data/30day134606830.csv")
SPX_CSV     = Path(r"Data/S&P 500 Historical Data 2015-2022.csv")

# Use one volatility source for BOTH models (so the comparison is apples-to-apples)
# If VIX9D CSV exists, you can point at it; otherwise use standard VIX history.
VIX_CSV     = Path(r"Data/VIX_History.csv")       # e.g. columns: DATE, CLOSE
VIX9D_CSV   = Path(r"Data/VIX9D_History.csv")     # e.g. columns: DATE, CLOSE (or VIX9D)

VOL_SOURCE  = "constant"     # "vix", "vix9d", or "constant"
SIGMA_CONST = 0.1573    # only used if VOL_SOURCE == "constant"

R = 0.0005              # annual risk-free rate
Q = 0.00                # dividend yield (SPX index dividends often captured via forward; keep 0 unless you model q)

# Option type
CP_FLAG = "C"           # "C" call, "P" put

# Heston parameters
# For a poster comparison, keep calibration OFF and use a stable, reasonable set.
# If you already calibrated, paste your values here.
HESTON_PARAMS = dict(
    kappa=3.0,
    theta=0.04,
    xi=0.5,
    rho=-0.7,
)

# Heston numerical integration settings (higher = more accurate, slower)
HESTON_UMAX = 80.0
HESTON_N    = 1500      # even-ish (Simpson rule uses even intervals)
# -------------------------------------------------------- #


# ---------------------- Helpers -------------------------- #
def to_float_clean(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.replace(",", "").replace("%", "")
    try:
        return float(x)
    except Exception:
        return np.nan


def parse_date(series: pd.Series, fmts: list[str], dayfirst_fallback=True) -> pd.Series:
    out = pd.Series(pd.NaT, index=series.index)
    for fmt in fmts:
        mask = out.isna()
        out.loc[mask] = pd.to_datetime(series[mask], format=fmt, errors="coerce")
    if dayfirst_fallback:
        out = out.fillna(pd.to_datetime(series, errors="coerce", dayfirst=True))
    else:
        out = out.fillna(pd.to_datetime(series, errors="coerce"))
    return out


def _pick_vix_value_col(vix: pd.DataFrame) -> str:
    # Prefer CLOSE, else something that looks like close/value
    for c in vix.columns:
        if c.strip().lower() == "close":
            return c
    for c in vix.columns:
        if c.strip().lower() in {"vix", "vix9d", "value", "adj close", "adj_close"}:
            return c
    # Fallback: first numeric-ish column after DATE
    for c in vix.columns:
        if c.strip().lower() != "date" and c.strip().lower() != "dte" and c.strip().lower() != "dt":
            return c
    raise ValueError("Could not find a VIX value column.")


# ----------------- Black–Scholes ------------------------- #
def black_scholes_euro(S0, K, T, r, sigma, option="call", q=0.0):
    if T <= 0:
        return max(S0 - K, 0.0) if option == "call" else max(K - S0, 0.0)

    if sigma <= 0:
        fwd = S0 * exp(-q*T)
        df = exp(-r*T)
        return df * max(fwd - K, 0.0) if option == "call" else df * max(K - fwd, 0.0)

    sqrtT = sqrt(T)
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    df = exp(-r * T)
    dq = exp(-q * T)

    if option == "call":
        return S0 * dq * norm.cdf(d1) - K * df * norm.cdf(d2)
    else:
        return K * df * norm.cdf(-d2) - S0 * dq * norm.cdf(-d1)


# --------------------- Heston ---------------------------- #
def heston_charfunc(u, S0, T, r, kappa, theta, xi, rho, v0):
    """
    Risk-neutral characteristic function φ(u) of log(S_T) under Heston.
    Uses a stable "little Heston trap" style formulation.
    """
    u = np.asarray(u, dtype=complex)
    i = 1j

    lnS0 = np.log(S0)
    a = kappa * theta
    b = kappa

    d = np.sqrt((rho * xi * i * u - b) ** 2 + (xi ** 2) * (i * u + u ** 2))
    g = (b - rho * xi * i * u - d) / (b - rho * xi * i * u + d)

    exp_neg_dT = np.exp(-d * T)
    one_minus_g_exp = 1.0 - g * exp_neg_dT
    one_minus_g = 1.0 - g

    C = (
        r * i * u * T
        + (a / (xi ** 2))
        * ((b - rho * xi * i * u - d) * T - 2.0 * np.log(one_minus_g_exp / one_minus_g))
    )
    D = ((b - rho * xi * i * u - d) / (xi ** 2)) * ((1.0 - exp_neg_dT) / one_minus_g_exp)

    return np.exp(C + D * v0 + i * u * lnS0)


def _heston_Pj(j, S0, K, T, r, kappa, theta, xi, rho, v0, umax=100.0, N=2000):
    assert j in (1, 2)
    N = int(N)
    if N % 2 == 1:
        N += 1  # Simpson needs even number of intervals
    du = umax / N
    u = np.linspace(du, umax, N)  # avoid u=0

    i = 1j
    lnK = np.log(K)

    phi_minus_i = heston_charfunc(-i, S0, T, r, kappa, theta, xi, rho, v0)

    if j == 1:
        phi_u = heston_charfunc(u - i, S0, T, r, kappa, theta, xi, rho, v0)
        numer = np.exp(-i * u * lnK) * np.exp(-r * T) * phi_u
        denom = i * u * phi_minus_i
    else:
        phi_u = heston_charfunc(u, S0, T, r, kappa, theta, xi, rho, v0)
        numer = np.exp(-i * u * lnK) * np.exp(-r * T) * phi_u
        denom = i * u

    integrand = np.real(numer / denom)

    w = np.ones_like(u)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0

    integral = (du / 3.0) * np.sum(w * integrand)
    return 0.5 + (1.0 / np.pi) * integral


def heston_price(S0, K, T, r, kappa, theta, xi, rho, v0, cp_flag="C", umax=100.0, N=2000):
    if T <= 0:
        intrinsic = max(S0 - K, 0.0)
        if cp_flag.upper() == "P":
            intrinsic = max(K - S0, 0.0)
        return intrinsic

    P1 = _heston_Pj(1, S0, K, T, r, kappa, theta, xi, rho, v0, umax=umax, N=N)
    P2 = _heston_Pj(2, S0, K, T, r, kappa, theta, xi, rho, v0, umax=umax, N=N)

    call = S0 * P1 - K * np.exp(-r * T) * P2
    if cp_flag.upper() == "C":
        return float(np.real(call))
    put = call - S0 + K * np.exp(-r * T)
    return float(np.real(put))


# ---------------------- Load data ------------------------ #
opt = pd.read_csv(OPTIONS_CSV)

# Parse dates (OptionMetrics single-option files often day-first)
opt["date"] = parse_date(opt["date"], ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"], dayfirst_fallback=True)
opt["exdate"] = parse_date(opt["exdate"], ["%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"], dayfirst_fallback=True)

opt["mid"] = (pd.to_numeric(opt["best_bid"], errors="coerce") + pd.to_numeric(opt["best_offer"], errors="coerce")) / 2.0
opt["K"] = pd.to_numeric(opt["strike_price"], errors="coerce") / 1000.0

# Daily reduce (mean mid, first K/exdate)
daily = (
    opt.sort_values("date")
       .groupby("date", as_index=False)
       .agg(mid=("mid", "mean"), K=("K", "first"), exdate=("exdate", "first"))
)

# SPX
spx = pd.read_csv(SPX_CSV)
spx["date"] = parse_date(spx["Date"], ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"], dayfirst_fallback=False)
spx["S"] = spx["Price"].apply(to_float_clean)
spx = spx[["date", "S"]].dropna(subset=["date", "S"])

# Merge
df = daily.merge(spx, on="date", how="inner").sort_values("date").reset_index(drop=True)

# Time-to-maturity
df["T_days"] = (df["exdate"] - df["date"]).dt.days.clip(lower=0)
df["T"] = df["T_days"] / 365.0

# Vol source
if VOL_SOURCE.lower() in {"vix", "vix9d"}:
    vix_path = VIX_CSV if VOL_SOURCE.lower() == "vix" else VIX9D_CSV
    vix = pd.read_csv(vix_path)

    # date col might be DATE or Date
    date_col = "DATE" if "DATE" in vix.columns else ("Date" if "Date" in vix.columns else "date")
    vix["date"] = parse_date(vix[date_col], ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"], dayfirst_fallback=False)

    val_col = _pick_vix_value_col(vix)
    vix["vix_level"] = pd.to_numeric(vix[val_col], errors="coerce")

    vix = vix[["date", "vix_level"]].dropna(subset=["date"]).sort_values("date")
    df = df.merge(vix, on="date", how="left")

    # Fill missing (holidays) by forward-fill so plot is continuous
    df["vix_level"] = df["vix_level"].ffill()

    # Annualised implied vol ~ VIX/100
    df["sigma"] = (df["vix_level"] / 100.0)
elif VOL_SOURCE.lower() == "constant":
    df["sigma"] = float(SIGMA_CONST)
else:
    raise ValueError(f"Unknown VOL_SOURCE: {VOL_SOURCE}")

# Heston v0 uses instantaneous variance. A clean poster comparison is:
# v0 = sigma^2 (using the same implied vol source as BS).
df["v0"] = df["sigma"] ** 2

# ---------------------- Price models --------------------- #
option_word = "call" if CP_FLAG.upper() == "C" else "put"

df["bs_price"] = df.apply(
    lambda r: black_scholes_euro(r["S"], r["K"], r["T"], R, r["sigma"], option_word, Q),
    axis=1
)

p = HESTON_PARAMS
df["heston_price"] = df.apply(
    lambda r: heston_price(
        r["S"], r["K"], r["T"], R,
        p["kappa"], p["theta"], p["xi"], p["rho"],
        r["v0"],
        cp_flag=CP_FLAG,
        umax=HESTON_UMAX,
        N=HESTON_N
    ),
    axis=1
)

# ---------------------- Metrics -------------------------- #
def _metrics(model_col: str):
    err = df[model_col] - df["mid"]
    rmse = float(np.sqrt(np.nanmean(err**2)))
    mae  = float(np.nanmean(np.abs(err)))
    bias = float(np.nanmean(err))
    return rmse, mae, bias

bs_rmse, bs_mae, bs_bias = _metrics("bs_price")
he_rmse, he_mae, he_bias = _metrics("heston_price")

print("\n=== Pricing metrics (model - market mid) ===")
print(f"Black–Scholes | RMSE={bs_rmse:.4f}  MAE={bs_mae:.4f}  Bias={bs_bias:.4f}")
print(f"Heston        | RMSE={he_rmse:.4f}  MAE={he_mae:.4f}  Bias={he_bias:.4f}")

# ---------------------- Poster plot ---------------------- #

title_suffix = f"Expiry={df['exdate'].iloc[0].date()} | Initial DTE={int(df['T_days'].iloc[0])}d"

fig = plt.figure(figsize=(14, 5.8))
plt.plot(df["date"], df["mid"],        linewidth=1.5, label="Market Price")
plt.plot(df["date"], df["bs_price"],   linewidth=1.5,linestyle="--", label="Black–Scholes")
plt.plot(df["date"], df["heston_price"], linewidth=1.5, linestyle="--", label="Heston")

plt.title(f"Black-Scholes vs Heston", fontsize=20)
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Option price ($)", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=15)
plt.grid(True, alpha=0.25)
plt.legend(frameon=False, loc="best", fontsize=17)
plt.tight_layout()

out_png = Path("bs_vs_heston_poster.png")
plt.savefig(out_png, dpi=300)
plt.show()

# Optional: save a poster-ready PNG next to the script

print(f"\nSaved poster-ready plot to: {out_png.resolve()}\n")
