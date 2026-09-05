# === Monte Carlo vs Market Mid: S&P underlying + Options CSV ===
# Structure aligned with binomial.py, but using Monte Carlo pricing.
# T is computed correctly from exdate - date (days to expiry).

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

R = 0.0012         # annual risk-free rate
Q = 0.00         # dividend yield

N_SIMS      = 20000   # Monte Carlo paths
ANTITHETIC  = True      # use antithetic variates
SEED        = 42        # base seed for reproducibility (per-day seed = SEED + row_idx)

FORCE_OPTION_TYPE = "call"   # "call", "put", or None to infer from delta
#SIGMA_OVERRIDE    = 0.11     # None = use file IV; number = force constant sigma
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
    """Standard Black–Scholes price for European call/put with continuous dividend yield q."""
    if (T is None) or (T <= 0) or (sigma is None) or (np.isnan(sigma)) or (sigma <= 0) \
       or (S0 <= 0) or (K <= 0):
        # For exact expiry T<=0, fall back to intrinsic if S,K are OK
        if T is not None and T <= 0 and S0 > 0 and K > 0:
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
    """BSM implied volatility (root-finding on Black–Scholes price)."""
    if (T is None) or (T <= 0) or (price is None) or (price <= 0) \
       or (S is None) or (K is None) or (S <= 0) or (K <= 0):
        return np.nan

    def f(vol):
        return black_scholes_euro(S, K, T, r, vol, option, q) - price

    try:
        return brentq(f, 1e-9, 5.0, maxiter=200)
    except Exception:
        return np.nan


def monte_carlo_euro(S0, K, T, r, sigma, option="call", q=0.0,
                     n_sims=100_000, antithetic=True, seed=None) -> float:
    """
    Risk-neutral Monte Carlo for European options with dividend yield q.
    S_T = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z), Z ~ N(0,1)
    Returns *only* the option price (to align with binomial interface).
    """
    # Handle expiry / degenerate cases like BSM helper
    if (T is None) or (T <= 0) or (sigma is None) or (np.isnan(sigma)) or (sigma <= 0) \
       or (S0 <= 0) or (K <= 0):
        if T is not None and T <= 0 and S0 > 0 and K > 0:
            return max(S0 - K, 0.0) if option == "call" else max(K - S0, 0.0)
        return np.nan

    rng = np.random.default_rng(seed)

    if antithetic:
        m = n_sims // 2
        Z = rng.standard_normal(m)
        Z = np.concatenate([Z, -Z])  # ≈ n_sims total draws
    else:
        Z = rng.standard_normal(n_sims)

    drift = (r - q - 0.5 * sigma ** 2) * T
    diffu = sigma * sqrt(T)
    ST = S0 * np.exp(drift + diffu * Z)

    if str(option).lower() == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)

    disc = exp(-r * T)
    price = float(disc * payoffs.mean())
    return price


# ---------- Load OPTIONS ----------
opt = pd.read_csv(OPTIONS_CSV)

col_best_bid   = "best_bid"
col_best_offer = "best_offer"
col_date       = "date"
col_exdate     = "exdate"
col_strike     = "strike_price"
col_delta      = "delta"
col_iv         = "impl_volatility"

opt[col_date]   = parse_date(opt[col_date],   fmts=["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"])
opt[col_exdate] = parse_date(opt[col_exdate], fmts=["%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"])

opt["mid"] = (
    pd.to_numeric(opt[col_best_bid],   errors="coerce") +
    pd.to_numeric(opt[col_best_offer], errors="coerce")
) / 2.0

# strikes in this dataset are typically given as 1000x index level
opt["K"] = pd.to_numeric(opt[col_strike], errors="coerce") / 1000.0

# # Choose sigma per row
# if SIGMA_OVERRIDE is None:
#     opt["sigma_row"] = normalize_iv(opt[col_iv])
# else:
#     opt["sigma_row"] = SIGMA_OVERRIDE

# Choose option type (global)
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
       .rename(columns={col_date: "date", col_exdate: "exdate", "sigma_row": "sigma"})
)


# ---------- Load SPX ----------
spx = pd.read_csv(SPX_CSV)
spx["Date"] = parse_date(
    spx["Date"],
    fmts=["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"],
    dayfirst_fallback=False
)
spx["Price"] = spx["Price"].apply(to_float_clean)

spx_daily = spx[["Date", "Price"]].rename(columns={"Date": "date", "Price": "S"})


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

# ---------- Compute T (days to expiry) ----------
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


# ---------- Monte Carlo pricing ----------
def mc_price_row(row):
    # Per-day seed so paths are reproducible but independent across days
    row_seed = SEED + int(row.name)
    return monte_carlo_euro(
        S0=row["S"],
        K=row["K"],
        T=row["T"],
        r=row["r"],
        sigma=row["sigma"],
        option=row["option_type"],
        q=row["q"],
        n_sims=N_SIMS,
        antithetic=ANTITHETIC,
        seed=row_seed
    )

df["model"] = df.apply(mc_price_row, axis=1)


# =================================================================
# === Daily Market IV → align to df (as in binomial.py)
# =================================================================
opt["norm_iv"] = normalize_iv(opt[col_iv])

market_iv_daily = (
    opt.sort_values([col_date])
       .groupby(col_date, as_index=False)
       .agg({"norm_iv": "mean"})
       .rename(columns={col_date: "date"})
)

df = df.merge(market_iv_daily, on="date", how="left")


# =================================================================
# === Model IV inversion (BSM), using Monte Carlo price
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
out_csv = "montecarlo_vs_market_daily_from_exdate.csv"
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

print("\nPreview:")
print(df[["date", "exdate", "T_days", "T", "S", "K", "mid", "model", "sigma_vix"]]
      .head(40).to_string(index=False))


# ---------- PLOTS ----------
# (Hash out / comment these blocks if you don't want graphs.)

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
    label="Monte Carlo",
    color= "#4D4D4D"
)

# # Titles and labels
# ax.set_title(
#     f"Market vs Monte Carlo",
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
png_path = save_dir / "market_vs_montecarlo_yearlong_atm.png"
pdf_path = save_dir / "market_vs_montecarlo_yearlong_atm.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
#fig.savefig(pdf_path, bbox_inches="tight")

plt.show()

print(f"Saved PNG to: {png_path}")
#print(f"Saved PDF to: {pdf_path}")

plt.figure(figsize=(10, 4))
plt.plot(df["date"], df["mid"] - df["model"])
plt.axhline(0, color='black')
plt.title("Residuals (Market - MC)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(df["mid"], df["model"], s=20)
mn = float(min(df["mid"].min(), df["model"].min()))
mx = float(max(df["mid"].max(), df["model"].max()))
plt.plot([mn, mx], [mn, mx])
plt.title("Market vs Monte Carlo Model")
plt.xlabel("Market")
plt.ylabel("Monte Carlo")
plt.tight_layout()
plt.show()


# ---------- Pricing Metrics ----------
df["error"]     = df["model"] - df["mid"]
df["abs_error"] = df["error"].abs()
df["sq_error"]  = df["error"] ** 2

mask = df[["mid", "model", "error"]].apply(np.isfinite).all(axis=1)

RMSE = np.sqrt(df.loc[mask, "sq_error"].mean())
MAE  = df.loc[mask, "abs_error"].mean()
bias = df.loc[mask, "error"].mean()

tol = 0.05 * df["mid"]
df["bae_component"] = (df["error"] - bias).abs()
BAE = df.loc[mask, "bae_component"].mean()

df["hit"] = (df["bae_component"] <= tol).astype(int)
hit_rate = df.loc[mask, "hit"].mean()


# ---------- IV Metrics ----------
iv_mask = df["model_iv"].apply(np.isfinite) & df["norm_iv"].apply(np.isfinite)

IV_MAE  = float(np.nanmean(np.abs(df.loc[iv_mask, "iv_error"])))
IV_RMSE = float(np.sqrt(np.nanmean(df.loc[iv_mask, "iv_error"] ** 2)))
IV_BIAS = float(np.nanmean(df.loc[iv_mask, "iv_error"]))


print("\n=== Pricing Metrics on Single Option Time-Series ===")
print(f"Rows used: {mask.sum()} / {len(df)}")
print(f"RMSE: {RMSE:.6f}")
print(f"MAE : {MAE:.6f}")
print(f"Bias (model - market): {bias:.6f}")
print(f"BAE : {BAE:.6f}")
print(f"Hit-rate: {hit_rate:.2%}")

print("\n=== IV Metrics ===")
print(f"IV MAE : {IV_MAE:.6f}")
print(f"IV RMSE: {IV_RMSE:.6f}")
print(f"IV Bias: {IV_BIAS:.6f}")

