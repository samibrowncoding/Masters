# === Binomial vs Market Mid: S&P underlying + Options CSV ===
# Fully corrected version with working IV inversion + metrics
# Market-IV is now aligned properly, avoiding all NaN issues.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import exp, sqrt, log
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------- USER INPUTS ---------------------- #
OPTIONS_CSV = Path(r"Data\yearlongatmoption.csv")
# OPTIONS_CSV = Path(r"Data\30day134606830.csv")
SPX_CSV     = Path(r"Data\S&P 500 Historical Data 2015-2022.csv")

R = 0.0012       # annual risk-free rate
Q = 0.00         # dividend yield
N_STEPS = 100  # binomial steps

FORCE_OPTION_TYPE = "call"   # or "put", or None
#SIGMA_OVERRIDE = 0.155      # None = use file IV
VIX_CSV = Path(r"Data\VIX_History.csv")

SIGMA_MODE = "constant"      # "vix", "option_iv", "constant"
SIGMA_CONSTANT = 0.2822 # only used if SIGMA_MODE == "constant"



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
    if med > 1.0:   # likely percent
        iv = iv / 100.0
    return iv


def infer_option_type_from_delta(delta_series: pd.Series) -> str:
    med = np.nanmedian(pd.to_numeric(delta_series, errors="coerce"))
    return "call" if med >= 0 else "put"


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


# ---------- pricing functions ----------
def black_scholes_euro(S0, K, T, r, sigma, option="call", q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0) if option == "call" else max(K - S0, 0.0)

    sqrtT = sqrt(T)
    d1 = (log(S0/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    df = exp(-r*T)
    dq = exp(-q*T)

    if option == "call":
        return S0*dq*norm.cdf(d1) - K*df*norm.cdf(d2)
    else:
        return K*df*norm.cdf(-d2) - S0*dq*norm.cdf(-d1)


def binomial_option_price(S0, K, T, r, sigma, N, option_type="call", q=0.0):
    if T <= 0:
        return max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)

    dt = T / N
    disc = exp(-r * dt)
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp((r - q) * dt) - d) / (u - d)
    if not (0 < p < 1):
        return np.nan

    j = np.arange(N+1)
    ST = S0 * (u**j) * (d**(N-j))

    if option_type == "call":
        vals = np.maximum(ST - K, 0.0)
    else:
        vals = np.maximum(K - ST, 0.0)

    for _ in range(N):
        vals = disc * (p*vals[1:] + (1-p)*vals[:-1])

    return float(vals[0])


def implied_vol_bs(S, K, T, r, price, option, q=0.0):
    """BSM implied volatility (bisection)."""
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return np.nan

    def f(vol):
        return black_scholes_euro(S, K, T, r, vol, option, q) - price

    try:
        return brentq(f, 1e-9, 5.0, maxiter=200)
    except:
        return np.nan


# ---------- Load OPTIONS ----------
opt = pd.read_csv(OPTIONS_CSV)

col_best_bid   = "best_bid"
col_best_offer = "best_offer"
col_date       = "date"
col_exdate     = "exdate"
col_strike     = "strike_price"
col_delta      = "delta"
col_iv         = "impl_volatility"

opt[col_date]   = parse_date(opt[col_date], fmts=["%d/%m/%Y","%Y-%m-%d","%m/%d/%Y"])
opt[col_exdate] = parse_date(opt[col_exdate],fmts=["%Y%m%d","%Y-%m-%d","%d/%m/%Y","%m/%d/%Y"])

opt["mid"] = (pd.to_numeric(opt[col_best_bid], errors="coerce") +
              pd.to_numeric(opt[col_best_offer], errors="coerce")) / 2
opt["K"] = pd.to_numeric(opt[col_strike], errors="coerce") / 1000.0

# if SIGMA_OVERRIDE is None:
#     opt["sigma_row"] = normalize_iv(opt[col_iv])
# else:
#     opt["sigma_row"] = SIGMA_OVERRIDE


if FORCE_OPTION_TYPE in ("call", "put"):
    opt_type = FORCE_OPTION_TYPE
else:
    opt_type = infer_option_type_from_delta(opt.get(col_delta, pd.Series(dtype=float)))


# ---------- Daily reduction ----------
daily = (
    opt.sort_values([col_date])
       .groupby(col_date, as_index=False)
       .agg({
           "mid": "mean",
           "K": "first",
           col_exdate: "first",
           
        })
       .rename(columns={col_date:"date", col_exdate:"exdate", "sigma_row":"sigma"})
)


# ---------- Load SPX ----------
spx = pd.read_csv(SPX_CSV)
spx["Date"] = parse_date(spx["Date"], fmts=["%m/%d/%Y","%Y-%m-%d","%d/%m/%Y"], dayfirst_fallback=False)
spx["Price"] = spx["Price"].apply(to_float_clean)

spx_daily = spx[["Date","Price"]].rename(columns={"Date":"date","Price":"S"})


# ---------- Merge ----------
df = (
    daily.merge(spx_daily, on="date", how="inner")
         .sort_values("date")
         .reset_index(drop=True)
)


vix = pd.read_csv(VIX_CSV)

# Parse VIX dates (your file looks like "10/09/2019")
vix["date"] = parse_date(vix["DATE"], ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"], dayfirst_fallback=False)
vix["VIX_close"] = pd.to_numeric(vix["CLOSE"], errors="coerce")

vix = vix[["date", "VIX_close"]].dropna(subset=["date"]).sort_values("date")

# Merge onto df
df = df.merge(vix, on="date", how="left")

# If some dates are missing in VIX file (e.g. holidays), forward-fill is reasonable
#df["VIX_close"] = df["VIX_close"].ffill()
df["sigma_vix"] = df["VIX_close"] / 100.0


# ---------- Compute T ----------
df["T_days"] = (df["exdate"] - df["date"]).dt.days.clip(lower=0)
df["T"]      = df["T_days"] / 365

df["r"] = R
df["q"] = Q
df["option_type"] = opt_type

if SIGMA_MODE == "vix":
    df["sigma"] = df["sigma_vix"]
elif SIGMA_MODE == "constant":
    df["sigma"] = SIGMA_CONSTANT
elif SIGMA_MODE == "option_iv":
    # use the daily option IV already computed in `daily` (renamed to df["sigma"] earlier)
    # (no change needed)
    pass
else:
    raise ValueError(f"Unknown SIGMA_MODE: {SIGMA_MODE}")



# ---------- Binomial pricing ----------
df["model"] = df.apply(lambda row:
        binomial_option_price(
            S0=row["S"], K=row["K"], T=row["T"],
            r=row["r"], sigma=row["sigma"], N=N_STEPS,
            option_type=row["option_type"], q=row["q"]
        ),
    axis=1
)


# =================================================================
# === NEW: PROPER DAILY MARKET IV → ALIGN TO df
# =================================================================
opt["norm_iv"] = normalize_iv(opt[col_iv])

market_iv_daily = (
    opt.sort_values([col_date])
       .groupby(col_date, as_index=False)
       .agg({"norm_iv":"mean"})
       .rename(columns={col_date:"date"})
)

df = df.merge(market_iv_daily, on="date", how="left")

# =================================================================
# === Model IV inversion (BSM), using binomial price
# =================================================================
df["model_iv"] = df.apply(
    lambda row: implied_vol_bs(
        row["S"], row["K"], row["T"], row["r"],
        row["model"], row["option_type"], row["q"]
    ),
    axis=1
)

df["iv_error"] = df["model_iv"] - df["norm_iv"]


# ---------- Save preview ----------
out_csv = "binomial_vs_market_daily_from_exdate.csv"
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

print("\nPreview:")
print(df[["date","exdate","T_days","T","S","K","mid", "sigma_vix"]].head(40).to_string(index=False))


# ---------- PLOTS ----------
# --------------------------------------------------------
# PROFESSIONAL-STYLE PRICE PLOT
# --------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Output folder
save_dir = Path(r"C:\Users\samib\OneDrive - Imperial College London\Imperial\Education\Year 4\Masters Project\Code\MastersProject\Results\long option")
#save_dir = Path(r"C:\Users\sb1922\OneDrive - Imperial College London\Imperial\Education\Year 4\Masters Project\Code\MastersProject\Results\DeepOTM option")
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
    label="Binomial",
    color="#C44E52"   # muted red
)

# # Titles and labels
# ax.set_title(
#     f"Market vs Binomial",
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



# Legend
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
png_path = save_dir / "market_vs_binomial_yearlong_atm.png"
pdf_path = save_dir / "market_vs_binomial_yearlong_atm.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
#fig.savefig(pdf_path, bbox_inches="tight")

plt.show()

print(f"Saved PNG to: {png_path}")
#print(f"Saved PDF to: {pdf_path}")

plt.figure(figsize=(10,4))
plt.plot(df["date"], df["mid"]-df["model"])
plt.axhline(0,color='black')
plt.title("Residuals")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(df["mid"], df["model"], s=20)
mn = float(min(df["mid"].min(), df["model"].min()))
mx = float(max(df["mid"].max(), df["model"].max()))
plt.plot([mn,mx],[mn,mx])
plt.title("Market vs Model")
plt.xlabel("Market"); plt.ylabel("Model")
plt.tight_layout()
plt.show()


# ---------- Pricing Metrics ----------
df["error"]     = df["model"] - df["mid"]
df["abs_error"] = df["error"].abs()
df["sq_error"]  = df["error"]**2

mask = df[["mid","model","error"]].apply(np.isfinite).all(axis=1)

RMSE = np.sqrt(df.loc[mask,"sq_error"].mean())
MAE  = df.loc[mask,"abs_error"].mean()
bias = df.loc[mask,"error"].mean()

tol = 0.05 * df["mid"]
df["bae_component"] = (df["error"] - bias).abs()
BAE = df.loc[mask,"bae_component"].mean()

df["hit"] = (df["bae_component"] <= tol).astype(int)
hit_rate = df.loc[mask,"hit"].mean()


# ---------- IV Metrics ----------
iv_mask = df["model_iv"].apply(np.isfinite) & df["norm_iv"].apply(np.isfinite)

IV_MAE  = float(np.nanmean(np.abs(df.loc[iv_mask,"iv_error"])))
IV_RMSE = float(np.sqrt(np.nanmean(df.loc[iv_mask,"iv_error"]**2)))
IV_BIAS = float(np.nanmean(df.loc[iv_mask,"iv_error"]))


print("\n=== Pricing Metrics ===")
print(f"RMSE: {RMSE:.6f}")
print(f"MAE : {MAE:.6f}")
print(f"Bias: {bias:.6f}")
print(f"BAE : {BAE:.6f}")
print(f"Hit-rate: {hit_rate:.2%}")

print("\n=== IV Metrics ===")
print(f"IV MAE : {IV_MAE:.6f}")
print(f"IV RMSE: {IV_RMSE:.6f}")
print(f"IV Bias: {IV_BIAS:.6f}")
