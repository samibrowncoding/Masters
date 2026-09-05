import pandas as pd
from pathlib import Path
import numpy as np
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# =========================================
# STEP 1 — Load the single-option CSV (the option we price every day)
# =========================================

OPTIONS_CSV = Path(r"Data\moneyness\DeepITMoption.csv")
#OPTIONS_CSV = Path(r"Data\30day134606830.csv")
opt = pd.read_csv(OPTIONS_CSV)

print("Columns:", opt.columns.tolist())
print(opt.head())

opt["date"]   = pd.to_datetime(opt["date"],   dayfirst=True)
opt["exdate"] = pd.to_datetime(opt["exdate"], dayfirst=True)

opt["mid"]    = (opt["best_bid"] + opt["best_offer"]) / 2.0
opt["strike"] = opt["strike_price"] / 1000.0
opt["T_days"] = (opt["exdate"] - opt["date"]).dt.days
opt["T"]      = opt["T_days"] / 365.0

opt_core = opt[["date", "exdate", "T_days", "T", "strike",
                 "best_bid", "best_offer", "mid", "impl_volatility"]].copy()
opt = opt_core.copy()

# =========================================
# STEP 2 — Load + clean SPX historical index
# =========================================

spx_path = Path("Data/S&P 500 Historical Data 2015-2022.csv")
spx = pd.read_csv(spx_path)

print("SPX raw columns:", spx.columns.tolist())
print(spx.head())

spx["date"] = pd.to_datetime(spx["Date"], format="mixed")

def clean_price(x):
    return float(str(x).replace(",", ""))

spx["S"] = spx["Price"].apply(clean_price)
spx = spx[["date", "S"]]

# =========================================
# STEP 3 — Load + clean VIX9D
# =========================================

vix_path = Path("Data/VIX9D_History.csv")
vix = pd.read_csv(vix_path)

print("VIX9D raw columns:", vix.columns.tolist())
print(vix.head())

vix["date"] = pd.to_datetime(vix["DATE"], format="mixed")
vix_value_col = [c for c in vix.columns if c.lower() in ["close", "vix9d", "value", "vix"]][0]
vix = vix[["date", vix_value_col]].rename(columns={vix_value_col: "vix9d"})

# =========================================
# STEP 4 — Build df2: the single option enriched with S and v0
# =========================================

df = opt.merge(spx, on="date", how="left")
df = df.merge(vix, on="date", how="left")
final = df[["date", "S", "vix9d", "T", "strike", "mid"]].copy()

df2 = final.copy()
df2["v0"] = (df2["vix9d"] / 100.0) ** 2

r_test = 0.004   # constant risk-free rate

print("\nSingle option rows:", len(df2))
print(df2[["date", "S", "v0", "T", "strike", "mid"]].head())

# =========================================
# STEP 5 — Heston pricing engine (unchanged from original)
# =========================================

def heston_charfunc(u, S0, T, r, kappa, theta, xi, rho, v0):
    """
    Risk-neutral characteristic function φ(u) of log(S_T) under Heston.
    Uses the 'little Heston trap' stable formulation.
    """
    u = np.asarray(u, dtype=complex)
    i = 1j
    lnS0 = np.log(S0)
    a = kappa * theta
    b = kappa
    d = np.sqrt((rho * xi * i * u - b)**2 + (xi**2) * (i * u + u**2))
    g = (b - rho * xi * i * u - d) / (b - rho * xi * i * u + d)
    exp_neg_dT      = np.exp(-d * T)
    one_minus_g_exp = 1.0 - g * exp_neg_dT
    one_minus_g     = 1.0 - g
    C = (r * i * u * T
         + (a / (xi**2)) * ((b - rho * xi * i * u - d) * T
         - 2.0 * np.log(one_minus_g_exp / one_minus_g)))
    D = ((b - rho * xi * i * u - d) / (xi**2)) * ((1.0 - exp_neg_dT) / one_minus_g_exp)
    return np.exp(C + D * v0 + i * u * lnS0)


def _heston_Pj(j, S0, K, T, r, kappa, theta, xi, rho, v0, umax=100.0, N=2000):
    """Compute P1 or P2 via Fourier integral using Simpson's rule."""
    assert j in (1, 2)
    N = int(N)
    if N % 2 == 1:
        N += 1
    du   = umax / N
    u    = np.linspace(du, umax, N)
    i    = 1j
    lnK  = np.log(K)
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


def heston_price(S0, K, T, r, kappa, theta, xi, rho, v0,
                 cp_flag="C", umax=100.0, N=2000):
    """Heston European option price using the semi-analytical formula."""
    if T <= 0:
        intrinsic = max(S0 - K, 0.0)
        if cp_flag.upper() == "P":
            intrinsic = max(K - S0, 0.0)
        return intrinsic
    P1 = _heston_Pj(1, S0, K, T, r, kappa, theta, xi, rho, v0, umax=umax, N=N)
    P2 = _heston_Pj(2, S0, K, T, r, kappa, theta, xi, rho, v0, umax=umax, N=N)
    call_price = S0 * P1 - K * np.exp(-r * T) * P2
    if cp_flag.upper() == "C":
        return np.real(call_price)
    else:
        return np.real(call_price - S0 + K * np.exp(-r * T))


# =========================================
# STEP 5A — Load the full OptionMetrics universe used for calibration
# =========================================

om_path = Path("Data/options_2021.csv")
om = pd.read_csv(om_path)

print("OM raw columns:", om.columns.tolist())
print(om.head())

om["date"]   = pd.to_datetime(om["date"],   format="mixed")
om["exdate"] = pd.to_datetime(om["exdate"], format="mixed")

# Filter for SPX index options
if "secid"      in om.columns: om = om[om["secid"]      == 108105]
if "index_flag" in om.columns: om = om[om["index_flag"] == 1]

# Basic sanity filters
om = om[(om["best_bid"] > 0) & (om["best_offer"] > 0)]
om["mid"] = (om["best_bid"] + om["best_offer"]) / 2.0
om = om[om["mid"] > 0]

# Strike in index units
om["K"] = om["strike_price"] / 1000.0

# Merge with SPX and VIX9D
om = om.merge(spx, on="date", how="left")
om = om.merge(vix, on="date", how="left")
om = om.dropna(subset=["S", "vix9d"])

# Time to maturity
om["T_days"] = (om["exdate"] - om["date"]).dt.days
om = om[om["T_days"] > 0]
om["T"]  = om["T_days"] / 365.0

# v0 from VIX9D
om["v0"] = (om["vix9d"] / 100.0) ** 2

# Moneyness filter
moneyness = om["S"] / om["K"]
om = om[(moneyness > 1) & (moneyness < 1.4)]

# Ensure cp_flag column exists
if "cp_flag" not in om.columns:
    for candidate in ["optiontype", "option_type", "type", "flag"]:
        if candidate in om.columns:
            om = om.rename(columns={candidate: "cp_flag"})
            break
    else:
        om["cp_flag"] = "C"   # fallback — adjust if your data has puts

# Keep only the columns needed for calibration, retaining the 'date' column
# so we can slice a rolling window for each pricing day
calib_cols = ["date", "S", "K", "T", "mid", "v0", "cp_flag"]
om_calib   = om[calib_cols].copy()

print("\nFull calibration universe shape:", om_calib.shape)
print("Date range:", om_calib["date"].min(), "→", om_calib["date"].max())


# =========================================
# STEP 5B — Calibration objective (same as original, unchanged)
# =========================================

MAX_OPTIONS   = 600            # max options sampled per calibration day
CALIB_WINDOW  = 30             # rolling look-back in calendar days

bounds        = [(0.5, 10.0), (0.0001, 0.5), (0.05, 2.0), (-0.999, -0.1)]
initial_guess = [2.0, 0.04, 0.5, -0.5]


def heston_objective(params, calib_df):
    kappa, theta, xi, rho = params

    # Penalise invalid region
    if kappa <= 0 or theta <= 0 or xi <= 0 or rho <= -1 or rho >= 1:
        return 1e10

    errors = []
    for _, row in calib_df.iterrows():
        try:
            price = heston_price(
                row["S"], row["K"], row["T"], r_test,
                kappa, theta, xi, rho,
                row["v0"],
                cp_flag=row["cp_flag"],
                umax=20.0,
                N=200
            )
            errors.append((price - row["mid"]) ** 2)
        except Exception:
            pass   # skip any numerical failures silently

    return np.mean(errors) if errors else 1e10


# =========================================
# STEP 6 — Daily calibration + pricing loop
#
#   For each trading day in df2:
#     1. Pull options from om_calib in the CALIB_WINDOW days before that date
#     2. Run the same L-BFGS-B optimiser as before
#     3. Price the single option using that day's calibrated parameters
#     4. Warm-start the next day from today's solution (faster + smoother)
# =========================================

trading_days = sorted(df2["date"].unique())
print(f"\nPricing {len(trading_days)} trading days  "
      f"({trading_days[0].date()} → {trading_days[-1].date()})")

heston_prices    = []
kappa_list, theta_list, xi_list, rho_list = [], [], [], []

prev_params = None   # warm-start: reuse previous day's solution as next initial guess
total_start = time.time()

for day_num, pricing_date in enumerate(trading_days, 1):

    # --- 1. Slice calibration window: CALIB_WINDOW days ending the day before ---
    window_end   = pricing_date - pd.Timedelta(days=1)
    window_start = pricing_date - pd.Timedelta(days=CALIB_WINDOW)

    calib = om_calib[
        (om_calib["date"] >= window_start) &
        (om_calib["date"] <= window_end)
    ].copy()

    # Downsample for speed if huge (same cap as original)
    if len(calib) > MAX_OPTIONS:
        calib = calib.sample(MAX_OPTIONS, random_state=42).reset_index(drop=True)

    # --- 2. Calibrate ---
    if len(calib) < 10:
        # Not enough data — reuse previous parameters rather than crash
        print(f"[{day_num}/{len(trading_days)}] {pricing_date.date()} — "
              f"only {len(calib)} options in window, reusing previous params")
        params = prev_params if prev_params is not None else initial_guess
    else:
        x0 = prev_params if prev_params is not None else initial_guess

        day_start = time.time()
        result = minimize(
            heston_objective,
            x0,
            args=(calib,),
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-8}
        )
        elapsed = time.time() - day_start

        params = list(result.x)
        print(f"[{day_num}/{len(trading_days)}] {pricing_date.date()} — "
              f"calib {'OK' if result.success else 'warn'} "
              f"obj={result.fun:.4f}  "
              f"kappa={params[0]:.4f} theta={params[1]:.5f} "
              f"xi={params[2]:.4f} rho={params[3]:.4f}  ({elapsed:.1f}s)")

    prev_params = params
    kappa_cal, theta_cal, xi_cal, rho_cal = params

    # --- 3. Price the single option for this day ---
    row = df2[df2["date"] == pricing_date].iloc[0]

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
        umax=80.0,    # higher accuracy for final prices (same as original)
        N=1500        # higher accuracy for final prices (same as original)
    )
    heston_prices.append(price)
    kappa_list.append(kappa_cal)
    theta_list.append(theta_cal)
    xi_list.append(xi_cal)
    rho_list.append(rho_cal)

total_elapsed = time.time() - total_start
print(f"\nTotal time: {total_elapsed/60:.1f} min")

# Write results back onto df2 — same column name as the original
df2["heston_calibrated"] = heston_prices
df2["kappa"] = kappa_list
df2["theta"] = theta_list
df2["xi"]    = xi_list
df2["rho"]   = rho_list

print("\n=== Sample of calibrated prices ===")
print(df2[["date", "S", "mid", "heston_calibrated"]].head())
print(df2[["date", "S", "mid", "heston_calibrated"]].tail())

# Save to CSV for later use
df2.to_csv("Data/heston_daily_results.csv", index=False)
print("\nSaved → Data/heston_daily_results.csv")


# =========================================
# STEP 7 — Plots for Heston performance (same structure as original + param plot)
# =========================================

# --- 1. Time-series comparison ---
plt.figure(figsize=(10, 5))
plt.plot(df2["date"], df2["mid"],               label="Market mid",          linewidth=1.8)
plt.plot(df2["date"], df2["heston_calibrated"], label="Heston (daily calib)", linestyle="--", linewidth=1.8)
plt.title("Heston (Daily Calibrated) vs Market Price", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Option Price", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 2. Residuals over time ---
plt.figure(figsize=(10, 4))
plt.plot(df2["date"], df2["heston_calibrated"] - df2["mid"], linewidth=1.8)
plt.axhline(0, color="black", linewidth=1)
plt.title("Residuals: Heston - Market", fontsize=13)
plt.xlabel("Date")
plt.ylabel("Residual")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 3. Scatter: Market vs Model ---
plt.figure(figsize=(6, 6))
plt.scatter(df2["mid"], df2["heston_calibrated"], s=30, alpha=0.7)
mn = float(min(df2["mid"].min(), df2["heston_calibrated"].min()))
mx = float(max(df2["mid"].max(), df2["heston_calibrated"].max()))
plt.plot([mn, mx], [mn, mx], color="black", linewidth=1)
plt.title("Market vs Heston (Daily Calibrated)", fontsize=14)
plt.xlabel("Market Price")
plt.ylabel("Heston Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 4. Calibrated parameters over time (new) ---
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for ax, col, label, colour in zip(
        axes,
        ["kappa", "theta", "xi", "rho"],
        ["κ  (mean reversion speed)", "θ  (long-run variance)",
         "ξ  (vol of vol)", "ρ  (spot-vol correlation)"],
        ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    ax.plot(df2["date"], df2[col], color=colour, linewidth=1.5)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True, alpha=0.3)
axes[0].set_title("Calibrated Heston Parameters Over Time", fontsize=13)
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.show()


# =========================================
# STEP 8 — Pricing Metrics (identical to original)
# =========================================

df2["error"]     = df2["heston_calibrated"] - df2["mid"]
df2["abs_error"] = df2["error"].abs()
df2["sq_error"]  = df2["error"] ** 2

mask = df2[["mid", "heston_calibrated", "error"]].apply(np.isfinite).all(axis=1)

RMSE = np.sqrt(df2.loc[mask, "sq_error"].mean())
MAE  = df2.loc[mask, "abs_error"].mean()
bias = df2.loc[mask, "error"].mean()

tol = 0.05 * df2["mid"]   # 5 % tolerance band
df2["bae_component"] = (df2["error"] - bias).abs()
BAE = df2.loc[mask, "bae_component"].mean()

df2["hit"] = (df2["bae_component"] <= tol).astype(int)
hit_rate = df2.loc[mask, "hit"].mean()

print("\n=== Heston Pricing Metrics (Daily Calibrated) ===")
print(f"RMSE:     {RMSE:.6f}")
print(f"MAE:      {MAE:.6f}")
print(f"Bias:     {bias:.6f}")
print(f"BAE:      {BAE:.6f}")
print(f"Hit-rate: {hit_rate:.2%}")