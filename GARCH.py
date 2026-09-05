# === GARCH(1,1) + Black–Scholes vs Market Mid ===
# Uses GARCH-implied volatility to price a *single option* each day until expiry.
# Structure mirrors binomial.py / montecarlo.py:
#   - T_days = exdate - date (calendar days)
#   - model price per day
#   - daily market IV vs model IV
#   - pricing + IV error metrics and plots

from __future__ import annotations
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import exp, sqrt, log
from scipy.stats import norm
from scipy.optimize import brentq, minimize


# ---------------------- USER INPUTS ---------------------- #
OPTIONS_CSV = Path(r"Data\yearlongatmoption.csv")
# OPTIONS_CSV = Path(r"Data\\30day134606830.csv")
SPX_CSV     = Path(r"Data\\S&P 500 Historical Data 2015-2022.csv")

R = 0.05
# R=0.0012          # annual risk-free rate
Q = 0.00          # dividend yield (set !=0 if you want)
TRADING_DAYS_PER_YEAR = 365  # GARCH convention: trading days, not calendar

FORCE_OPTION_TYPE = "call"   # "call", "put", or None to infer from delta

# GARCH controls
GARCH_WINDOW_MIN =1000    # need at least this many returns to fit
# -------------------------------------------------------- #


# ---------- helpers ----------
def to_float_clean(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.replace(",", "").replace("%", "")
    try:
        return float(x)
    except Exception:
        return np.nan


def normalize_iv(iv_series: pd.Series) -> pd.Series:
    """Normalize IV to decimal form if stored in percent."""
    iv = pd.to_numeric(iv_series, errors="coerce")
    if iv.notna().sum() == 0:
        return iv
    med = np.nanmedian(iv.values)
    if med > 1.0:   # likely in percent
        iv = iv / 100.0
    return iv


def infer_option_type_from_delta(delta_series: pd.Series) -> str:
    med = np.nanmedian(pd.to_numeric(delta_series, errors="coerce"))
    return "call" if med >= 0 else "put"


def parse_date(series: pd.Series, fmts: list[str], dayfirst_fallback=True) -> pd.Series:
    """Try several explicit formats, then fall back to pandas with/without dayfirst."""
    out = pd.Series(pd.NaT, index=series.index)
    for fmt in fmts:
        mask = out.isna()
        out.loc[mask] = pd.to_datetime(series[mask], format=fmt, errors="coerce")
    if dayfirst_fallback:
        out = out.fillna(pd.to_datetime(series, errors="coerce", dayfirst=True))
    else:
        out = out.fillna(pd.to_datetime(series, errors="coerce"))
    return out


# ---------- pricing functions ----------
def black_scholes_euro(S0, K, T, r, sigma, option="call", q=0.0):
    """Standard BSM with continuous dividend yield q."""
    if (T is None) or (T <= 0) or (sigma is None) or (np.isnan(sigma)) or (sigma <= 0) \
       or (S0 is None) or (K is None) or (S0 <= 0) or (K <= 0):
        # For exact expiry, fall back to intrinsic if S,K are OK
        if T is not None and T <= 0 and S0 is not None and K is not None and S0 > 0 and K > 0:
            return max(S0 - K, 0.0) if option == "call" else max(K - S0, 0.0)
        return np.nan

    sqrtT = sqrt(T)
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    df_r = exp(-r * T)
    df_q = exp(-q * T)

    if option == "call":
        return S0 * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    else:
        return K * df_r * norm.cdf(-d2) - S0 * df_q * norm.cdf(-d1)


def implied_vol_bs(S, K, T, r, price, option, q=0.0):
    """BSM implied volatility via 1D root-finding."""
    if (T is None) or (T <= 0) or (price is None) or (price <= 0) \
       or (S is None) or (K is None) or (S <= 0) or (K <= 0):
        return np.nan

    def f(vol):
        return black_scholes_euro(S, K, T, r, vol, option, q) - price

    try:
        return brentq(f, 1e-9, 5.0, maxiter=200)
    except Exception:
        return np.nan


# ---------- 1) Load OPTIONS ----------
opt = pd.read_csv(OPTIONS_CSV)

col_best_bid   = "best_bid"
col_best_offer = "best_offer"
col_date       = "date"
col_exdate     = "exdate"
col_strike     = "strike_price"
col_delta      = "delta"
col_iv         = "impl_volatility"

opt[col_date]   = parse_date(opt[col_date],   fmts=["%d/%m/%Y","%Y-%m-%d","%m/%d/%Y"])
opt[col_exdate] = parse_date(opt[col_exdate], fmts=["%Y%m%d","%Y-%m-%d","%d/%m/%Y","%m/%d/%Y"])

# mid-price and strike scaling (OptionMetrics-style strikes / 1000)
opt["mid"] = (
    pd.to_numeric(opt[col_best_bid],   errors="coerce") +
    pd.to_numeric(opt[col_best_offer], errors="coerce")
) / 2.0
opt["K"] = pd.to_numeric(opt[col_strike], errors="coerce") / 1000.0

# Market IV for IV-error metrics later
opt["norm_iv"] = normalize_iv(opt[col_iv])

# Global option type
if FORCE_OPTION_TYPE in ("call", "put"):
    opt_type = FORCE_OPTION_TYPE
else:
    opt_type = infer_option_type_from_delta(opt.get(col_delta, pd.Series(dtype=float)))


# ---------- Daily reduction for price/strike/exdate ----------
daily = (
    opt.sort_values([col_date])
       .groupby(col_date, as_index=False)
       .agg({
           "mid": "mean",
           "K": "first",
           col_exdate: "first"
        })
       .rename(columns={col_date:"date", col_exdate:"exdate"})
)


# ---------- Load SPX (underlying) ----------
spx = pd.read_csv(SPX_CSV)
spx["Date"]  = parse_date(spx["Date"], fmts=["%m/%d/%Y","%Y-%m-%d","%d/%m/%Y"], dayfirst_fallback=False)
spx["Price"] = spx["Price"].apply(to_float_clean)

spx_daily = spx[["Date","Price"]].rename(columns={"Date":"date","Price":"S"}).dropna(subset=["date","S"])
spx_daily = spx_daily.sort_values("date").reset_index(drop=True)


# ---------- Merge option daily with underlying ----------
df = (
    daily.merge(spx_daily, on="date", how="inner")
         .sort_values("date")
         .reset_index(drop=True)
)


# ---------- Compute T (calendar days to expiry) ----------
# df["T_days"] = (df["exdate"] - df["date"]).dt.days.clip(lower=0)
# df["T"]      = df["T_days"] / 365.0
date_d   = df["date"].values.astype("datetime64[D]")
exdate_d = df["exdate"].values.astype("datetime64[D]")

hol_idx = USFederalHolidayCalendar().holidays(
    start=df["date"].min(),
    end=df["exdate"].max()
)

# Convert to numpy datetime64[D] array for np.busday_count
hol = hol_idx.values.astype("datetime64[D]")

df["T_bd"] = np.busday_count(date_d, exdate_d, holidays=hol).astype(int)
df["T_bd"] = df["T_bd"].clip(lower=0)
df["T"] = df["T_bd"] / TRADING_DAYS_PER_YEAR

df["r"] = R
df["q"] = Q
df["option_type"] = opt_type

# --------------------------------------------------------
# Lock GARCH calibration date at option start
# --------------------------------------------------------
CALIB_DATE = df["date"].iloc[0]   # option start date


# =================================================================
# 2) Fit GARCH(1,1) on underlying daily log-returns
# =================================================================
und = spx_daily.set_index("date").sort_index().copy()
und["ret"] = np.log(und["S"]).diff()

#r_series = und["ret"].dropna()
r_series = und.loc[und.index < CALIB_DATE, "ret"].dropna()

r_all = und.loc[und.index < CALIB_DATE, "ret"].dropna()
r_series = r_all.tail(1000)   



if len(r_series) < GARCH_WINDOW_MIN:
    raise ValueError(f"Need at least {GARCH_WINDOW_MIN} daily returns to fit GARCH, have {len(r_series)}.")

mu0 = r_series.mean()
eps = r_series - mu0


def garch_negll(theta):
    """
    Negative log-likelihood for Gaussian GARCH(1,1):
        r_t = mu + eps_t
        eps_t ~ N(0, h_t)
        h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
    """
    omega, alpha, beta = theta
    if (omega <= 0) or (alpha < 0) or (beta < 0) or (alpha + beta >= 0.999):
        return 1e12
    e = eps.values
    h = np.empty_like(e)
    # initialise at unconditional variance or sample var
    h[0] = max(np.var(e), omega / max(1e-8, (1.0 - alpha - beta)))
    for t in range(1, len(h)):
        h[t] = omega + alpha * (e[t-1] ** 2) + beta * h[t-1]
        if h[t] <= 0:
            return 1e12
    # Gaussian log-likelihood
    ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(h) + e * e / h)
    return -ll


# crude but effective multi-start search
best_val = np.inf
best_theta = None
rng = np.random.default_rng(1234)
for _ in range(10):
    omega0 = 0.1 * eps.var()
    alpha0 = rng.uniform(0.01, 0.20)
    beta0  = rng.uniform(0.70, 0.98)
    init = np.array([omega0, alpha0, beta0])
    res = minimize(garch_negll, init, method="L-BFGS-B")
    if res.success and res.fun < best_val:
        best_val = res.fun
        best_theta = res.x

if best_theta is None:
    raise RuntimeError("GARCH(1,1) fit failed.")

omega, alpha, beta = best_theta
phi = alpha + beta

# rebuild conditional variance series on full und index (including NaN ret on first date)
h_full = np.empty(len(und))
e_full = (und["ret"] - mu0).fillna(0.0).values
h_full[0] = max(np.var(eps.values), omega / max(1e-8, (1.0 - alpha - beta)))
for t in range(1, len(h_full)):
    h_full[t] = omega + alpha * (e_full[t-1] ** 2) + beta * h_full[t-1]
und["h"] = h_full
und["eps"] = und["ret"] - mu0


# e = eps.values  # calibration shocks
# h = np.empty_like(e)

# # initialise at unconditional variance or sample var
# h[0] = max(np.var(e), omega / max(1e-8, (1.0 - alpha - beta)))

# for t in range(1, len(h)):
#     h[t] = omega + alpha * (e[t-1] ** 2) + beta * h[t-1]
#     if h[t] <= 0:
#         raise RuntimeError("Non-positive variance encountered in calibration recursion.")

# # Locked calibration state (end of calibration window)
# h_calib = float(h[-1])
# eps_calib_sq = float(e[-1] ** 2)


# =================================================================
# 3) Forward variance → GARCH-implied annual sigma for each option date
# =================================================================
def sigma_forward_T(h_t, eps_t_sq, horizon_days: int, trad_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Compute annualised BS volatility for a horizon of `horizon_days` trading days,
    starting from conditional variance h_t and last shock eps_t_sq.

    We sum expected future daily variances over H days (sumH), then use:
        Var(R_T) = sumH = sigma^2 * T,  T = H / trad_days
        => sigma^2 = sumH * trad_days / H
    """
    if (h_t is None) or np.isnan(h_t) or (eps_t_sq is None) or np.isnan(eps_t_sq) or (horizon_days is None) or (horizon_days <= 0):
        return np.nan

    # one-step-ahead variance
    h_next = omega + alpha * eps_t_sq + beta * h_t
    h_inf  = omega / max(1e-8, (1.0 - phi))

    H = int(horizon_days)
    if H <= 0:
        return np.nan

    if abs(1.0 - phi) < 1e-6:
        # near-IGARCH: treat all future days at h_next
        sumH = h_next * H
    else:
        # sum of a geometric approach back to long-run variance
        sumH = h_inf * H + (h_next - h_inf) * ((1.0 - (phi ** H)) / (1.0 - phi))

    sigma2_annual = sumH * trad_days / H
    if sigma2_annual <= 0:
        return np.nan
    return float(np.sqrt(sigma2_annual))


# Build convenient lookup on underlying side
und_aligned = und.copy()

def garch_sigma_for_row(row) -> float:
    date = row["date"]
    H = row["T_bd"]
    if pd.isna(date) or pd.isna(H) or H <= 0:
        return np.nan

    # Use information strictly prior to this date (lag-1), like original code
    mask_prev = und_aligned.index < date
    if not mask_prev.any():
        return np.nan
    h_t = und_aligned.loc[mask_prev, "h"].iloc[-1]
    eps_t_sq = und_aligned.loc[mask_prev, "eps"].iloc[-1] ** 2
    return sigma_forward_T(h_t, eps_t_sq, horizon_days=int(H), trad_days=TRADING_DAYS_PER_YEAR)

# Conditional variance at calibration date (last available)
# h_calib = und.loc[und.index < CALIB_DATE, "h"].iloc[-1]
# eps_calib_sq = und.loc[und.index < CALIB_DATE, "eps"].iloc[-1] ** 2

# def garch_sigma_for_row(row):
#     H = int(row["T_bd"])
#     if H <= 0:
#         return np.nan

#     return sigma_forward_T(
#         h_t=h_calib,
#         eps_t_sq=eps_calib_sq,
#         horizon_days=H,
#         trad_days=TRADING_DAYS_PER_YEAR
#     )


df["sigma_garch"] = df.apply(garch_sigma_for_row, axis=1)


# =================================================================
# 4) Price with BSM using GARCH-implied sigma
# =================================================================
df["model"] = df.apply(
    lambda row: black_scholes_euro(
        S0=row["S"],
        K=row["K"],
        T=row["T"],
        r=row["r"],
        sigma=row["sigma_garch"],
        option=row["option_type"],
        q=row["q"],
    ),
    axis=1,
)

df["residual"] = df["mid"] - df["model"]


# =================================================================
# 5) Market IV vs GARCH model IV
# =================================================================
market_iv_daily = (
    opt.sort_values([col_date])
       .groupby(col_date, as_index=False)
       .agg({"norm_iv":"mean"})
       .rename(columns={col_date:"date"})
)

df = df.merge(market_iv_daily, on="date", how="left")
df.rename(columns={"norm_iv":"market_iv"}, inplace=True)

df["model_iv"] = df.apply(
    lambda row: implied_vol_bs(
        row["S"], row["K"], row["T"], row["r"], row["model"], row["option_type"], row["q"]
    ),
    axis=1,
)

df["iv_error"] = df["model_iv"] - df["market_iv"]


# ---------- Save preview ----------
out_csv = "garch_bs_vs_market_daily_exdateT.csv"
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")



print("\nPreview:")
print(
    df[["date","exdate","T_bd","T","S","K","mid","model","sigma_garch","market_iv","model_iv"]]
      .head(40)
      .to_string(index=False)
)

print("\n=== GARCH(1,1) Fitted Parameters ===")
print(f"omega: {omega:.6e}")
print(f"alpha: {alpha:.6f}")
print(f"beta : {beta:.6f}")
print(f"alpha + beta (persistence): {phi:.6f}")
print(f"Unconditional variance: {omega / (1 - phi):.6e}")
print(f"Unconditional sigma   : {np.sqrt(omega / (1 - phi)):.6f}")


# =================================================================
# 6) Plots (comment out if running headless)
# =================================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Output folder
save_dir = Path(r"C:\Users\samib\OneDrive - Imperial College London\Imperial\Education\Year 4\Masters Project\Code\MastersProject\Results\long option")
#save_dir = Path(r"C:\Users\sb1922\OneDrive - Imperial College London\Imperial\Education\Year 4\Masters Project\Code\MastersProject\Results\DeepITM option")
save_dir.mkdir(parents=True, exist_ok=True)

# Make sure date is datetime
df["date"] = pd.to_datetime(df["date"])

# Sort by date just in case
plot_df = df.sort_values("date").copy()

# Figure + axes
fig, ax = plt.subplots(figsize=(20, 6), dpi=150)

# Lines
ax.plot(
    plot_df["date"], plot_df["mid"],
    linewidth=3.5,
    label="Market"
)
ax.plot(
    plot_df["date"], plot_df["model"],
    linewidth=3.5,
    linestyle="--",
    label="GARCH",
    color= "#55A868" 
)

# # Titles and labels
# ax.set_title(
#     f"Market vs GARCH",
#     fontsize=15,
#     pad=14,
#     weight="bold"
# )
ax.set_xlabel("Date", fontsize=30)
ax.set_ylabel("Option Price ($)", fontsize=30)

# Grid
ax.grid(True, which="major", linestyle="--", alpha=0.35)

# Clean up spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Tick formatting
ax.tick_params(axis="both", labelsize=28)



# # Legend
leg = ax.legend(frameon=True, fontsize=28, loc="best")
leg.get_frame().set_alpha(0.95)

# leg = ax.legend(
#     frameon=True,
#     fontsize=28,
#     loc="upper right",
#     bbox_to_anchor=(1.065, 0.98)
# )
# leg.get_frame().set_alpha(0.95)

# Tight layout
plt.tight_layout()

# Save
png_path = save_dir / "market_vs_garch_yearlong_atm.png"
pdf_path = save_dir / "market_vs_garch_yearlong_atm.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
#fig.savefig(pdf_path, bbox_inches="tight")

plt.show()

print(f"Saved PNG to: {png_path}")
#print(f"Saved PDF to: {pdf_path}")

plt.figure(figsize=(10,4))
plt.plot(df["date"], df["residual"], label="Residual (mid - model)")
plt.axhline(0, linestyle="--", color="k", lw=1)
plt.title("Pricing Residuals Over Time (GARCH model)")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(df["mid"], df["model"], s=20, alpha=0.8)
mn = float(np.nanmin([df["mid"].min(), df["model"].min()]))
mx = float(np.nanmax([df["mid"].max(), df["model"].max()]))
plt.plot([mn, mx], [mn, mx], lw=1)
plt.title("Market vs GARCH(1,1)+BS Model")
plt.xlabel("Market mid")
plt.ylabel("Model")
plt.tight_layout()
plt.show()


# =================================================================
# 7) Pricing + IV Metrics (same style as binomial.py)
# =================================================================
df["error"]     = df["model"] - df["mid"]
df["abs_error"] = df["error"].abs()
df["sq_error"]  = df["error"] ** 2

mask = df[["mid","model","error"]].apply(np.isfinite).all(axis=1)

RMSE = float(np.sqrt(df.loc[mask, "sq_error"].mean()))
MAE  = float(df.loc[mask, "abs_error"].mean())
bias = float(df.loc[mask, "error"].mean())

tol = 0.05 * df["mid"]
df["bae_component"] = (df["error"] - bias).abs()
BAE = float(df.loc[mask, "bae_component"].mean())

df["hit"] = (df["bae_component"] <= tol).astype(int)
hit_rate = float(df.loc[mask, "hit"].mean())

# IV metrics
iv_mask = df["model_iv"].apply(np.isfinite) & df["market_iv"].apply(np.isfinite)

IV_MAE  = float(np.nanmean(np.abs(df.loc[iv_mask, "iv_error"])))
IV_RMSE = float(np.sqrt(np.nanmean(df.loc[iv_mask, "iv_error"] ** 2)))
IV_BIAS = float(np.nanmean(df.loc[iv_mask, "iv_error"]))


print("\n=== Pricing Metrics on Single Option Time-Series (GARCH) ===")
print(f"Rows used: {mask.sum()} / {len(df)}")
print(f"RMSE: {RMSE:.6f}")
print(f"MAE : {MAE:.6f}")
print(f"Bias (model - market): {bias:.6f}")
print(f"BAE : {BAE:.6f}")
print(f"Hit-rate: {hit_rate:.2%}")

print("\n=== IV Metrics (GARCH-implied vs Market IV) ===")
print(f"IV MAE : {IV_MAE:.6f}")
print(f"IV RMSE: {IV_RMSE:.6f}")
print(f"IV Bias: {IV_BIAS:.6f}")


if __name__ == "__main__":
    # everything runs on import; having the guard avoids accidental re-execution if imported
    pass

