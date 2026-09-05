# ============================================================
# === Black–Scholes vs Market Mid: S&P Underlying + Options CSV
# === Fully parallels the structure of your binomial file
# === Adds model IV, IV error, IV MAE / RMSE / Bias
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import exp, sqrt, log
from scipy import stats
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------- USER INPUTS ---------------------- #
OPTIONS_CSV = Path(r"Data\yearlongatmoption.csv")
# OPTIONS_CSV = Path(r"Data\30day134606830.csv")
# OPTIONS_CSV = Path(r"Data\moneyness\DeepITMoption.csv")

SPX_CSV     = Path(r"Data\S&P 500 Historical Data 2015-2022.csv")

R = 0.0012        # annual risk-free rate
Q = 0.00         # dividend yield

# If you want to override BS sigma, set a constant; else None to use file IV
#SIGMA_OVERRIDE = 0.1573
VIX_CSV = Path(r"Data\VIX_History.csv")

SIGMA_MODE = "constant"      # "vix", "option_iv", "constant"
SIGMA_CONSTANT = 0.2822 # 0.1573 # only used if SIGMA_MODE == "constant"


FORCE_OPTION_TYPE = "call"   # "call", "put", or None (infer from delta)
# -------------------------------------------------------- #
print("hello")


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
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
    """Coerce IV to numeric and divide by 100 if stored as percent."""
    iv = pd.to_numeric(iv_series, errors="coerce")
    if iv.notna().sum() == 0:
        return iv
    med = np.nanmedian(iv.values)
    if med > 1.0:
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



# --------------------------------------------------------
# Black–Scholes functions
# --------------------------------------------------------
def black_scholes_euro(S0, K, T, r, sigma, option="call", q=0.0):
    """Closed-form Black–Scholes (European)."""
    if T <= 0:
        if option == "call":
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)

    if sigma <= 0:
        fwd = S0 * exp(-q*T)
        df = exp(-r*T)
        if option == "call":
            return df * max(fwd - K, 0.0)
        else:
            return df * max(K - fwd, 0.0)

    sqrtT = sqrt(T)
    d1 = (log(S0/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    df = exp(-r*T)
    dq = exp(-q*T)

    if option == "call":
        return S0*dq*norm.cdf(d1) - K*df*norm.cdf(d2)
    else:
        return K*df*norm.cdf(-d2) - S0*dq*norm.cdf(-d1)


def implied_vol_bs(S, K, T, r, price, option, q=0.0):
    """Bisection inversion of BSM IV. Returns np.nan if no solution."""
    if T <= 0: return np.nan
    if price <= 0: return np.nan

    def f(vol):
        return black_scholes_euro(S, K, T, r, vol, option, q) - price

    try:
        return brentq(f, 1e-9, 5.0, maxiter=200)
    except:
        return np.nan



# --------------------------------------------------------
# Load & Parse Options CSV
# --------------------------------------------------------
opt = pd.read_csv(OPTIONS_CSV)

col_best_bid = "best_bid"
col_best_offer = "best_offer"
col_date = "date"
col_exdate = "exdate"
col_strike = "strike_price"
col_delta = "delta"
col_iv = "impl_volatility"

# Parse dates
opt[col_date]  = parse_date(opt[col_date],  ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"])
opt[col_exdate]= parse_date(opt[col_exdate],["%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"])

if opt[col_exdate].isna().any():
    bad = opt[opt[col_exdate].isna()].head(5)
    raise ValueError(f"Could not parse some exdates:\n{bad[[col_date,col_exdate]].to_string(index=False)}")


# Mid price
opt["mid"] = (pd.to_numeric(opt[col_best_bid], errors="coerce") +
              pd.to_numeric(opt[col_best_offer], errors="coerce")) / 2.0

# Strike
opt["K"] = pd.to_numeric(opt[col_strike], errors="coerce") / 1000.0





# Option type
if FORCE_OPTION_TYPE in ("call", "put"):
    opt_type = FORCE_OPTION_TYPE
else:
    opt_type = infer_option_type_from_delta(opt.get(col_delta, pd.Series(dtype=float)))


# Reduce to daily
daily = (
    opt.sort_values([col_date])
    .groupby(col_date, as_index=False)
    .agg({
        "mid": "mean",
        "K": "first",
        col_exdate: "first",

    })
)

daily = daily.rename(columns={col_date: "date", col_exdate: "exdate", "sigma_row": "sigma"})


# --------------------------------------------------------
# Load SPX
# --------------------------------------------------------
spx = pd.read_csv(SPX_CSV)
spx["Date"]  = parse_date(spx["Date"], ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"], dayfirst_fallback=False)
spx["Price"] = spx["Price"].apply(to_float_clean)
spx_daily = spx[["Date", "Price"]].dropna(subset=["Date"]).rename(columns={"Date":"date", "Price":"S"})


# --------------------------------------------------------
# Merge
# --------------------------------------------------------
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




# --------------------------------------------------------
# Compute T using actual exdate difference
# --------------------------------------------------------
df["T_days"] = (df["exdate"] - df["date"]).dt.days.clip(lower=0)
df["T"] = df["T_days"] / 365

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


# --------------------------------------------------------
# Black–Scholes Pricing
# --------------------------------------------------------
df["model"] = df.apply(lambda row:
                       black_scholes_euro(
                           row["S"], row["K"], row["T"], row["r"],
                           row["sigma"], row["option_type"], row["q"]),
                       axis=1)

df["residual"] = df["mid"] - df["model"]



# --------------------------------------------------------
# IMPLIED VOLATILITY & IV ERROR METRICS
# --------------------------------------------------------
opt_iv_daily = (
    opt.groupby(col_date, as_index=False)[col_iv]
       .mean()
       .rename(columns={col_date: "date", col_iv: "market_iv_raw"})
)

opt_iv_daily["market_iv"] = normalize_iv(opt_iv_daily["market_iv_raw"])

df = df.merge(opt_iv_daily[["date", "market_iv"]], on="date", how="left")

# Model-implied vol from the MODEL price (this will basically recover df["sigma"])
df["model_iv"] = df.apply(lambda row:
                          implied_vol_bs(
                              row["S"], row["K"], row["T"], row["r"],
                              row["model"], row["option_type"], row["q"]),
                          axis=1)

df["iv_error"] = df["model_iv"] - df["market_iv"]



# --------------------------------------------------------
# Save preview CSV
# --------------------------------------------------------
out_csv = "blackscholes_vs_market_daily_from_exdate.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved merged/priced series to: {out_csv}\n")

print("Preview (T decreasing):")
print(df.loc[:, ["date","exdate","T_days","T","S","K","mid","sigma_vix"]].head(50).to_string(index=False))



# --------------------------------------------------------
# TITLE helpers
# --------------------------------------------------------
first_exdate = df["exdate"].iloc[0]
initial_dte_days = int(df["T_days"].iloc[0])

title_suffix = f"initial DTE={initial_dte_days} days | expiry={first_exdate.date()}"


# --------------------------------------------------------
# PROFESSIONAL-STYLE PRICE PLOT
# --------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Output folder
save_dir = Path(r"C:\Users\samib\OneDrive - Imperial College London\Imperial\Education\Year 4\Masters Project\Code\MastersProject\Results\moneyness\long option")
#save_dir = Path(r"C:\Users\sb1922\OneDrive - Imperial College London\Imperial\Education\Year 4\Masters Project\Code\MastersProject\Results\OTM option")
save_dir.mkdir(parents=True, exist_ok=True)

# Make sure date is datetime
df["date"] = pd.to_datetime(df["date"])

# Sort by date just in case
plot_df = df.sort_values("date").copy()

# Figure + axes
fig, ax = plt.subplots(figsize=(20, 10), dpi=150)

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
    label="Black–Scholes"
)

# # Titles and labels
# ax.set_title(
#     f"Market vs Black–Scholes",
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
png_path = save_dir / "market_vs_black_scholes_long.png"
pdf_path = save_dir / "market_vs_black_scholes_long.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
#fig.savefig(pdf_path, bbox_inches="tight")

plt.show()

print(f"Saved PNG to: {png_path}")
#print(f"Saved PDF to: {pdf_path}")




plt.figure(figsize=(10,4))
plt.plot(df["date"], df["residual"])
plt.axhline(0, linestyle="--")
plt.title(f"Residuals over time | {title_suffix}")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(df["mid"], df["model"])
mn = np.nanmin([df["mid"].min(), df["model"].min()])
mx = np.nanmax([df["mid"].max(), df["model"].max()])
plt.plot([mn, mx], [mn, mx])
plt.title(f"Model vs Market Mid | {title_suffix}")
plt.xlabel("Market mid")
plt.ylabel("Model")
plt.tight_layout()
plt.show()



# --------------------------------------------------------
# METRICS — EXACT SAME STRUCTURE AS BINOMIAL FILE
# --------------------------------------------------------
df["error"] = df["model"] - df["mid"]
df["abs_error"] = df["error"].abs()
df["sq_error"] = df["error"]**2

mask = df[["mid","model","error"]].apply(np.isfinite).all(axis=1)
n_used = int(mask.sum())

RMSE = float(np.sqrt(df.loc[mask,"sq_error"].mean()))
MAE = float(df.loc[mask,"abs_error"].mean())
bias = float(df.loc[mask,"error"].mean())

# BAE
tol = 0.05 * df["mid"]
df["bae_component"] = (df["error"] - bias).abs()
BAE = float(df.loc[mask,"bae_component"].mean())

# Hit-rate
df["hit"] = (df["bae_component"] <= tol).astype(int)
hit_rate = float(df.loc[mask,"hit"].mean())

# IV metrics
iv_mask = df["model_iv"].apply(np.isfinite)
IV_MAE = float(np.nanmean(np.abs(df.loc[iv_mask, "iv_error"])))
IV_RMSE = float(np.sqrt(np.nanmean(df.loc[iv_mask, "iv_error"]**2)))
IV_BIAS = float(np.nanmean(df.loc[iv_mask, "iv_error"]))


print("\n=== Pricing Metrics on Single Option Time-Series ===")
print(f"Rows used: {n_used} / {len(df)}")
print(f"RMSE: {RMSE:.6f}")
print(f"MAE : {MAE:.6f}")
print(f"Bias (model - market): {bias:.6f}")
print(f"BAE: {BAE:.6f}")
print(f"Hit-rate: {hit_rate:.2%}")

print("\n=== Implied Volatility Metrics ===")
print(f"IV MAE : {IV_MAE:.6f}")
print(f"IV RMSE: {IV_RMSE:.6f}")
print(f"IV Bias: {IV_BIAS:.6f}")










# import seaborn as sns
# # Keep only finite errors
# err = df.loc[mask, "error"].dropna()

# plt.figure(figsize=(8, 5))
# sns.histplot(err, bins=35, kde=False)
# plt.axvline(err.mean(), color="red", linestyle="--", linewidth=1.5, label="Mean error")
# plt.axvline(0, color="black", linestyle="--", linewidth=1.0, label="Zero error")
# plt.title("Distribution of Pricing Errors")
# plt.xlabel("Error (Model - Market)")
# plt.ylabel("Frequency")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6, 6))
# stats.probplot(err, dist="norm", plot=plt)
# plt.title("QQ Plot of Pricing Errors")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.scatter(df.loc[mask, "model"], df.loc[mask, "error"], alpha=0.45)
# plt.axhline(0, color="black", linestyle="--", linewidth=1)
# plt.title("Pricing Errors vs Model Price")
# plt.xlabel("Model Price")
# plt.ylabel("Error (Model - Market)")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.scatter(df.loc[mask, "T"], df.loc[mask, "error"], alpha=0.45)
# plt.axhline(0, color="black", linestyle="--", linewidth=1)
# plt.title("Pricing Errors vs Time to Maturity")
# plt.xlabel("Time to Maturity (Years)")
# plt.ylabel("Error (Model - Market)")
# plt.tight_layout()
# plt.show()

# if {"S", "K"}.issubset(df.columns):
#     df["moneyness"] = df["S"] / df["K"]

#     plt.figure(figsize=(8, 5))
#     plt.scatter(df.loc[mask, "moneyness"], df.loc[mask, "error"], alpha=0.45)
#     plt.axhline(0, color="black", linestyle="--", linewidth=1)
#     plt.title("Pricing Errors vs Moneyness")
#     plt.xlabel("Moneyness (S/K)")
#     plt.ylabel("Error (Model - Market)")
#     plt.tight_layout()
#     plt.show()

# if "moneyness" in df.columns:
#     df["moneyness_bucket"] = pd.cut(
#         df["moneyness"],
#         bins=[0, 0.8, 0.96, 1.04, 1.2, np.inf],
#         labels=["Deep ITM", "ITM", "ATM", "OTM", "Deep OTM"]
#     )

#     plt.figure(figsize=(8, 5))
#     sns.boxplot(data=df.loc[mask], x="moneyness_bucket", y="error")
#     plt.axhline(0, color="black", linestyle="--", linewidth=1)
#     plt.title("Pricing Errors by Moneyness Bucket")
#     plt.xlabel("Moneyness Bucket")
#     plt.ylabel("Error (Model - Market)")
#     plt.tight_layout()
#     plt.show()