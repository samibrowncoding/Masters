import pandas as pd
from pathlib import Path
import numpy as np

# === 1. Load the single-option CSV ===
opt_path = Path(r"Data\yearlongatmoption.csv")
# opt_path = Path(r"Data\30day134606830.csv")
opt = pd.read_csv(opt_path)

# Quick sanity check on original columns
print("Columns:", opt.columns.tolist())
print(opt.head())

# === 2. Parse dates ===
# Your option file appears to use day-first format like '01/07/2021' = 1 July 2021
opt["date"] = pd.to_datetime(opt["date"], dayfirst=True)
opt["exdate"] = pd.to_datetime(opt["exdate"], dayfirst=True)

# === 3. Compute mid price ===
# You already have bid/ask; Unnamed: 15 also looks like mid, but let's recompute
opt["mid"] = (opt["best_bid"] + opt["best_offer"]) / 2.0

# === 4. Convert strike to index units ===
# OptionMetrics-style strike_price often 4,200,000 -> 4200.0
opt["strike"] = opt["strike_price"] / 1000.0

# === 5. Compute time-to-maturity in years ===
# T = (exdate - date) in calendar days /365.0
opt["T_days"] = (opt["exdate"] - opt["date"]).dt.days
opt["T"] = opt["T_days"] /365.0

# === 6. Keep only the core columns we need for now ===
opt_core = opt[["date", "exdate", "T_days", "T", "strike", "best_bid", "best_offer", "mid", "impl_volatility"]].copy()

# print("\n=== Cleaned single-option data (head) ===")
# print(opt_core.head())

# print("\n=== Tail to check last dates & T ===")
# print(opt_core.tail())

# print("\nUnique exdate:", opt_core["exdate"].unique())
# print("Min date:", opt_core["date"].min())
# print("Max date:", opt_core["date"].max())
# print("Min T_days:", opt_core["T_days"].min(), "Max T_days:", opt_core["T_days"].max())

# === 0. Load your cleaned single-option dataset from Step 1 ===
opt = opt_core.copy()

# ----------------------------------------------------------------------------------------
# === 1. Load + clean SPX historical index ===
# ----------------------------------------------------------------------------------------

spx_path = Path("Data/S&P 500 Historical Data 2015-2022.csv")
spx = pd.read_csv(spx_path)

print("SPX raw columns:", spx.columns.tolist())
print(spx.head())

# Parse date safely even if formats are mixed
spx["date"] = pd.to_datetime(spx["Date"], format="mixed")

# Remove commas and convert to float
def clean_price(x):
    return float(str(x).replace(",", ""))

spx["S"] = spx["Price"].apply(clean_price)

spx = spx[["date", "S"]]

# ----------------------------------------------------------------------------------------
# === 2. Load + clean VIX9D ===
# ----------------------------------------------------------------------------------------

vix_path = Path("Data/VIX9D_History.csv")
vix = pd.read_csv(vix_path)

print("VIX9D raw columns:", vix.columns.tolist())
print(vix.head())

# Parse date with mixed formats
vix["date"] = pd.to_datetime(vix["DATE"], format="mixed")

# Detect the column containing VIX values
vix_value_col = [c for c in vix.columns if c.lower() in ["close", "vix9d", "value", "vix"]][0]

vix = vix[["date", vix_value_col]].rename(columns={vix_value_col: "vix9d"})

# ----------------------------------------------------------------------------------------
# === 3. Merge all datasets ===
# ----------------------------------------------------------------------------------------

df = opt.merge(spx, on="date", how="left")
df = df.merge(vix, on="date", how="left")

# ----------------------------------------------------------------------------------------
# === 4. Final clean dataframe ===
# ----------------------------------------------------------------------------------------

final = df[["date", "S", "vix9d", "T", "strike", "mid"]].copy()

# print("\n=== Final merged dataset (head) ===")
# print(final.head())

# print("\n=== Missing SPX/VIX values ===")
# print(final.isna().sum())

def heston_charfunc(u, S0, T, r, kappa, theta, xi, rho, v0):
    """
    Risk-neutral characteristic function φ(u) of log(S_T) under Heston.
    Uses the 'little Heston trap' stable formulation.
    
    Parameters
    ----------
    u : complex or np.ndarray of complex
        Integration variable.
    S0 : float
        Spot price at time 0.
    T : float
        Time to maturity in years.
    r : float
        Constant risk-free rate.
    kappa, theta, xi, rho, v0 : floats
        Heston parameters.
    """
    u = np.asarray(u, dtype=complex)
    i = 1j

    # Log spot
    lnS0 = np.log(S0)

    # Common quantities
    a = kappa * theta
    b = kappa
    # d and g as in the original Heston paper (with little trap choice)
    d = np.sqrt((rho * xi * i * u - b)**2 + (xi**2) * (i * u + u**2))
    g = (b - rho * xi * i * u - d) / (b - rho * xi * i * u + d)

    # Avoid numerical issues when 1 - g * exp(-dT) is near zero
    exp_neg_dT = np.exp(-d * T)
    one_minus_g_exp = 1.0 - g * exp_neg_dT
    one_minus_g = 1.0 - g

    C = (r * i * u * T
         + (a / (xi**2)) * ((b - rho * xi * i * u - d) * T
         - 2.0 * np.log(one_minus_g_exp / one_minus_g)))
    D = ((b - rho * xi * i * u - d) / (xi**2)) * ((1.0 - exp_neg_dT) / one_minus_g_exp)

    return np.exp(C + D * v0 + i * u * lnS0)


def _heston_Pj(j, S0, K, T, r, kappa, theta, xi, rho, v0,
               umax=100.0, N=2000):
    """
    Compute P1 or P2 via Fourier integral using Simpson's rule.
    j = 1 or 2.
    """
    assert j in (1, 2)
    # Integration grid (avoid u = 0 exactly)
    N = int(N)
    if N % 2 == 1:
        N += 1  # Simpson requires even number of intervals
    du = umax / N
    u = np.linspace(du, umax, N)  # start at du, not 0

    i = 1j
    lnK = np.log(K)

    # Precompute φ(-i) once (used for P1)
    phi_minus_i = heston_charfunc(-i, S0, T, r, kappa, theta, xi, rho, v0)

    # Characteristic functions at needed arguments
    if j == 1:
        # f1(u) = exp(-iu lnK) * exp(-rT) * φ(u - i) / (i u φ(-i))
        phi_u = heston_charfunc(u - i, S0, T, r, kappa, theta, xi, rho, v0)
        numer = np.exp(-i * u * lnK) * np.exp(-r * T) * phi_u
        denom = i * u * phi_minus_i
    else:
        # f2(u) = exp(-iu lnK) * exp(-rT) * φ(u) / (i u)
        phi_u = heston_charfunc(u, S0, T, r, kappa, theta, xi, rho, v0)
        numer = np.exp(-i * u * lnK) * np.exp(-r * T) * phi_u
        denom = i * u

    integrand = np.real(numer / denom)

    # Simpson's rule weights
    w = np.ones_like(u)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0

    integral = (du / 3.0) * np.sum(w * integrand)

    Pj = 0.5 + (1.0 / np.pi) * integral
    return Pj


def heston_price(S0, K, T, r, kappa, theta, xi, rho, v0,
                 cp_flag="C", umax=100.0, N=2000):
    """
    Heston European option price using the semi-analytical formula.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate (constant).
    kappa, theta, xi, rho, v0 : floats
        Heston parameters.
    cp_flag : str, optional
        'C' for call, 'P' for put.
    umax : float, optional
        Upper limit of integration in u-space.
    N : int, optional
        Number of integration points (even) for Simpson's rule.

    Returns
    -------
    price : float
        Option price.
    """
    if T <= 0:
        # At or past maturity: payoff
        intrinsic = max(S0 - K, 0.0)
        if cp_flag.upper() == "P":
            intrinsic = max(K - S0, 0.0)
        return intrinsic

    P1 = _heston_Pj(1, S0, K, T, r, kappa, theta, xi, rho, v0,
                    umax=umax, N=N)
    P2 = _heston_Pj(2, S0, K, T, r, kappa, theta, xi, rho, v0,
                    umax=umax, N=N)

    call_price = S0 * P1 - K * np.exp(-r * T) * P2

    if cp_flag.upper() == "C":
        return np.real(call_price)
    else:
        # Put via put-call parity
        put_price = call_price - S0 + K * np.exp(-r * T)
        return np.real(put_price)
    
   

# # ========== Quick sanity test ==========
# S0_test = 100.0
# K_test  = 100.0
# T_test  = 0.5     # 6 months
# r_test  = 0.01

# # Some reasonable Heston parameters
# kappa_test = 2.0
# theta_test = 0.04
# xi_test    = 0.5
# rho_test   = -0.7
# v0_test    = 0.04

# price_call = heston_price(S0_test, K_test, T_test, r_test,
#                           kappa_test, theta_test, xi_test, rho_test, v0_test,
#                           cp_flag="C", umax=100.0, N=2000)

# print("Test Heston call price:", price_call)

df2 = final.copy()

# --- 1. Convert VIX9D to daily v0 ---
# VIX9D is in %, so divide by 100
df2["v0"] = (df2["vix9d"] / 100.0)**2

# --- 2. Temporary Heston parameters (reasonable guesses) ---
kappa_test = 3.0
theta_test = 0.04
xi_test    = 0.5
rho_test   = -0.7
r_test     = 0.004  # constant r as discussed

# --- 3. Compute Heston price per day ---
heston_prices = []

for idx, row in df2.iterrows():
    price = heston_price(
        row["S"],
        row["strike"],
        row["T"],
        r_test,
        kappa_test,
        theta_test,
        xi_test,
        rho_test,
        row["v0"],
        cp_flag="C",
        umax=40.0,   # lower umax for speed (temporary)
        N=800        # lower N for speed (temporary)
    )
    heston_prices.append(price)


df2["heston_price"] = heston_prices


# print("Df2")
# print(df2.head())
# print(df2.tail())

# STEP 5A — Load calibration dataset
# =========================================

# You will replace this with your real OptionMetrics merge later.
# For now, create a placeholder DataFrame structure.

# Example structure (replace when ready):
# calib = pd.DataFrame({
#     "S":      df2["S"].values,
#     "K":      df2["strike"].values,
#     "T":      df2["T"].values,
#     "mid":    df2["mid"].values,
#     "v0":     df2["v0"].values,
#     "cp_flag": ["C"] * len(df2)
# })

# print("Calibration dataset shape:", calib.shape)
# print(calib.head())

# #step 5b
# from scipy.optimize import minimize

# def heston_objective(params, calib_df):
#     kappa, theta, xi, rho = params

#     # Penalise invalid region
#     if kappa <= 0 or theta <= 0 or xi <= 0 or rho <= -1 or rho >= 1:
#         return 1e10

#     errors = []

#     for idx, row in calib_df.iterrows():
#         price = heston_price(
#             row["S"], row["K"], row["T"], r_test,
#             kappa, theta, xi, rho,
#             row["v0"],
#             cp_flag=row["cp_flag"],
#             umax=25.0,
#             N=600
#         )
#         errors.append((price - row["mid"])**2)

#     return np.mean(errors)


# bounds = [
#     (0.5, 10.0),    # kappa
#     (0.0001, 0.5),  # theta
#     (0.05, 2.0),    # xi
#     (-0.999, -0.1)  # rho
# ]

# initial_guess = [2.0, 0.04, 0.5, -0.5]

# result = minimize(
#     heston_objective,
#     initial_guess,
#     args=(calib,),
#     bounds=bounds,
#     method="L-BFGS-B"
# )

# print("\n=== Calibration result ===")
# print("Success:", result.success)
# print("Message:", result.message)
# print("Params:", result.x)
# print("Objective:", result.fun)


























# 1. Load big OptionMetrics CSV (change filename/path as needed)
om_path = Path("Data/options_2020.csv") 
om = pd.read_csv(om_path)

print("OM raw columns:", om.columns.tolist())
print(om.head())

# 2. Parse dates (mixed formats safe)
om["date"] = pd.to_datetime(om["date"], format="mixed")
om["exdate"] = pd.to_datetime(om["exdate"], format="mixed")

# 3. Filter for SPX index options
# Adjust the condition if your file uses different identifiers
if "secid" in om.columns:
    om = om[om["secid"] == 108105]
if "index_flag" in om.columns:
    om = om[om["index_flag"] == 1]










# 4. Calibration window (before your option starts on 2021-07-01)
calib_end = pd.Timestamp("2020-06-30")                  #dont forget about this
calib_start = calib_end - pd.Timedelta(days=90)

om = om[(om["date"] >= calib_start) & (om["date"] <= calib_end)]

# 5. Basic sanity filters
om = om[(om["best_bid"] > 0) & (om["best_offer"] > 0)]
om["mid"] = (om["best_bid"] + om["best_offer"]) / 2.0
om = om[om["mid"] > 0]

# 6. Strike in index units
om["K"] = om["strike_price"] / 1000.0

# 7. Merge with SPX and VIX9D (we already built spx, vix earlier)
om = om.merge(spx, on="date", how="left")   # adds 'S'
om = om.merge(vix, on="date", how="left")   # adds 'vix9d'

# Drop rows without underlying or VIX
om = om.dropna(subset=["S", "vix9d"])

# 8. Time to maturity T
om["T_days"] = (om["exdate"] - om["date"]).dt.days
om = om[om["T_days"] > 0]
om["T"] = om["T_days"] /365.0

# 9. v0 from VIX9D
om["v0"] = (om["vix9d"] / 100.0)**2

# 10. Moneyness filter: keep somewhat near-the-money options
moneyness = om["S"] / om["K"]
om = om[(moneyness > 0.5) & (moneyness < 1.5)]

# 11. Keep only what we need
calib_cols = ["S", "K", "T", "mid", "v0", "cp_flag"]
calib = om[calib_cols].copy()

# 12. Downsample for speed if huge
MAX_OPTIONS = 1000
if len(calib) > MAX_OPTIONS:
    calib = calib.sample(MAX_OPTIONS, random_state=42).reset_index(drop=True)

print("Calibration dataset shape:", calib.shape)
print(calib.head())


#step 5b
from scipy.optimize import minimize

def heston_objective(params, calib_df):
    kappa, theta, xi, rho = params

    # Penalise invalid region
    if kappa <= 0 or theta <= 0 or xi <= 0 or rho <= -1 or rho >= 1:
        return 1e10

    errors = []

    for idx, row in calib_df.iterrows():
        price = heston_price(
            row["S"], row["K"], row["T"], r_test,
            kappa, theta, xi, rho,
            row["v0"],
            cp_flag=row["cp_flag"],
            umax=20.0,
            N=200
        )
        errors.append((price - row["mid"])**2)

    return np.mean(errors)


bounds = [
    (0.5, 10.0),    # kappa
    (0.0001, 0.5),  # theta
    (0.05, 2.0),    # xi
    (-0.999, -0.1)  # rho
]

initial_guess = [2.0, 0.04, 0.5, -0.5]



import time
start_time = time.time()
iter_count = 0

def calib_callback(xk):
    global iter_count
    iter_count += 1
    elapsed = time.time() - start_time
    print(f"Iteration {iter_count} — time elapsed: {elapsed:.1f}s — params: {xk}")

result = minimize(
    heston_objective,
    initial_guess,
    args=(calib,),
    bounds=bounds,
    method="L-BFGS-B",
    callback=calib_callback
)

print("\n=== Calibration result ===")
print("Success:", result.success)
print("Message:", result.message)
print("Params:", result.x)
print("Objective:", result.fun)


# =========================================
# STEP 6 — Apply calibrated Heston parameters to your single option
# =========================================

# 1. Extract calibrated parameters
kappa_cal = result.x[0]
theta_cal = result.x[1]
xi_cal    = result.x[2]
rho_cal   = result.x[3]

print("\n=== Calibrated Parameters ===")
print(f"kappa = {kappa_cal:.6f}")
print(f"theta = {theta_cal:.6f}")
print(f"xi    = {xi_cal:.6f}")
print(f"rho   = {rho_cal:.6f}")

# 2. Price your single option with high accuracy now
# (Increase integration grid because this is the final pricing step)
heston_prices = []

for idx, row in df2.iterrows():
    price = heston_price(
        row["S"],
        row["strike"],
        row["T"],
        r_test,
        kappa_cal,
        theta_cal,
        xi_cal,
        rho_cal,
        row["v0"],
        cp_flag="C",
        umax=80.0,   # HIGHER ACCURACY FOR FINAL PRICES
        N=1500       # HIGHER ACCURACY FOR FINAL PRICES
    )
    heston_prices.append(price)

df2["heston_calibrated"] = heston_prices

print("\n=== Sample of calibrated prices ===")
print(df2[["date", "S", "mid", "heston_calibrated"]].head())
print(df2[["date", "S", "mid", "heston_calibrated"]].tail())

# 3. Compute pricing metrics
diff = df2["heston_calibrated"] - df2["mid"]

RMSE = (diff**2).mean()**0.5
MAE  = diff.abs().mean()
bias = diff.mean()
hit_rate = (np.sign(df2["heston_calibrated"] - df2["mid"]) == 
            np.sign(df2["mid"].diff().fillna(0))).mean() * 100

print("\n=== Heston Pricing Metrics (Calibrated) ===")
print(f"RMSE: {RMSE:.6f}")
print(f"MAE : {MAE:.6f}")
print(f"Bias: {bias:.6f}")
print(f"Hit-rate: {hit_rate:.2f}%")

# 4. OPTIONAL: compare day-by-day
print("\n=== Head Comparison ===")
print(df2[["date", "S", "mid", "heston_calibrated"]].head())

print("\n=== Tail Comparison ===")
print(df2[["date", "S", "mid", "heston_calibrated"]].tail())


# =========================================
# STEP 7 — Plots for Heston performance
# =========================================

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Output folder
save_dir = Path(
    r"C:\Users\samib\OneDrive - Imperial College London\Imperial\Education\Year 4\Masters Project\Code\MastersProject\Results\long option"
)
save_dir.mkdir(parents=True, exist_ok=True)

# Make sure date is datetime
df2["date"] = pd.to_datetime(df2["date"])

# Sort by date
plot_df = df2.sort_values("date").copy()

# Figure + axes
fig, ax = plt.subplots(figsize=(20, 6), dpi=150)

# Lines
ax.plot(
    plot_df["date"], plot_df["mid"],
    linewidth=3.5,
    label="Market"
)

ax.plot(
    plot_df["date"], plot_df["heston_calibrated"],
    linewidth=3.5,
    linestyle="--",
    label="Heston",
    color="#9467BD"  # muted purple
)

# Axis labels
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
png_path = save_dir / "market_vs_heston_yearlong_atm.png"
pdf_path = save_dir / "market_vs_heston_yearlong_atm.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
# fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)

plt.show()

print(f"Saved PNG to: {png_path}")
# # --- 1. Time-series comparison ---
# plt.figure(figsize=(10,5))
# plt.plot(df2["date"], df2["mid"], label="Market mid", linewidth=1.8)
# plt.plot(df2["date"], df2["heston_calibrated"], label="Heston (calibrated)", linestyle="--", linewidth=1.8)
# plt.title("Heston (Calibrated) vs Market Price", fontsize=14)
# plt.xlabel("Date", fontsize=12)
# plt.ylabel("Option Price", fontsize=12)
# plt.legend(fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# --- 2. Residuals over time ---
plt.figure(figsize=(10,4))
plt.plot(df2["date"], df2["heston_calibrated"] - df2["mid"], linewidth=1.8)
plt.axhline(0, color="black", linewidth=1)
plt.title("Residuals: Heston - Market", fontsize=13)
plt.xlabel("Date")
plt.ylabel("Residual")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 3. Scatter: Market vs Model ---
plt.figure(figsize=(6,6))
plt.scatter(df2["mid"], df2["heston_calibrated"], s=30, alpha=0.7)
mn = float(min(df2["mid"].min(), df2["heston_calibrated"].min()))
mx = float(max(df2["mid"].max(), df2["heston_calibrated"].max()))
plt.plot([mn, mx], [mn, mx], color='black', linewidth=1)
plt.title("Market vs Heston (Calibrated)", fontsize=14)
plt.xlabel("Market Price")
plt.ylabel("Heston Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================
# STEP 8 — Pricing Metrics
# =========================================

df2["error"] = df2["heston_calibrated"] - df2["mid"]
df2["abs_error"] = df2["error"].abs()
df2["sq_error"] = df2["error"]**2

mask = df2[["mid", "heston_calibrated", "error"]].apply(np.isfinite).all(axis=1)

RMSE = np.sqrt(df2.loc[mask, "sq_error"].mean())
MAE  = df2.loc[mask, "abs_error"].mean()
bias = df2.loc[mask, "error"].mean()

tol = 0.05 * df2["mid"]  # 5% tolerance band
df2["bae_component"] = (df2["error"] - bias).abs()
BAE = df2.loc[mask, "bae_component"].mean()

df2["hit"] = (df2["bae_component"] <= tol).astype(int)
hit_rate = df2.loc[mask, "hit"].mean()

print("\n=== Heston Pricing Metrics (Calibrated) ===")
print(f"RMSE: {RMSE:.6f}")
print(f"MAE : {MAE:.6f}")
print(f"Bias: {bias:.6f}")
print(f"BAE : {BAE:.6f}")
print(f"Hit-rate: {hit_rate:.2%}")


# # Only if df2["model_iv"] and df2["market_iv"] exist

# iv_mask = df2["model_iv"].apply(np.isfinite) & df2["market_iv"].apply(np.isfinite)

# IV_MAE  = float(np.nanmean(np.abs(df2.loc[iv_mask,"model_iv"] - df2.loc[iv_mask,"market_iv"])))
# IV_RMSE = float(np.sqrt(np.nanmean((df2.loc[iv_mask,"model_iv"] - df2.loc[iv_mask,"market_iv"])**2)))
# IV_BIAS = float(np.nanmean(df2.loc[iv_mask,"model_iv"] - df2.loc[iv_mask,"market_iv"]))

# print("\n=== IV Metrics ===")
# print(f"IV MAE : {IV_MAE:.6f}")
# print(f"IV RMSE: {IV_RMSE:.6f}")
# print(f"IV BIAS: {IV_BIAS:.6f}")
