# === Monte Carlo vs Market Mid: S&P underlying + Options CSV ===
# Requires: montecarlo.py in the same folder (we import monte_carlo_euro)
# Dates: options = DD/MM/YYYY, S&P = MM/DD/YYYY
# Strike scaling: strike_price / 1000

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Bring in your MC engine ---
from Models.montecarlogood import monte_carlo_euro  # uses antithetic MC with CI, seed, etc.

# ---------------------- USER INPUTS ---------------------- #
OPTIONS_CSV = Path(r"Data/30day134606830.csv")
SPX_CSV     = Path(r"Data/S&P 500 Historical Data 2015-2022.csv")

R = 0.04           # annual risk-free (your input)
Q = 0.00           # dividend yield (your input)
T_DAYS = 30        # time to expiry in days (your input)

# MC knobs
N_SIMS = 200_000
ANTITHETIC = True
SEED = 42

# Force option type?  "call" or "put" or None to infer from delta sign
FORCE_OPTION_TYPE = None

# Force a single vol? set e.g. 0.20, else None to use file IV per-day
SIGMA_OVERRIDE = None
# -------------------------------------------------------- #

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
    iv = pd.to_numeric(iv_series, errors="coerce")
    if iv.notna().sum() == 0:
        return iv
    med = np.nanmedian(iv.values)
    if med > 1.0:  # likely given in percent (e.g., 20 for 20%)
        iv = iv / 100.0
    return iv

def infer_option_type_from_delta(delta_series: pd.Series) -> str:
    med = np.nanmedian(pd.to_numeric(delta_series, errors="coerce"))
    return "call" if med >= 0 else "put"

def main():
    # ---------- Load & parse OPTIONS (DD/MM/YYYY) ----------
    opt = pd.read_csv(OPTIONS_CSV)

    col_best_bid   = "best_bid"
    col_best_offer = "best_offer"
    col_date       = "date"
    col_strike     = "strike_price"
    col_delta      = "delta"
    col_iv         = "impl_volatility"

    # Explicit: DD/MM/YYYY for the options file
    opt[col_date] = pd.to_datetime(opt[col_date], format="%d/%m/%Y", errors="coerce")

    # Market mid
    opt["mid"] = (pd.to_numeric(opt[col_best_bid], errors="coerce") +
                  pd.to_numeric(opt[col_best_offer], errors="coerce")) / 2.0

    # Strike scaling (OptionMetrics style → ÷1000)
    opt["K"] = pd.to_numeric(opt[col_strike], errors="coerce") / 1000.0

    # Vol per-day
    if SIGMA_OVERRIDE is None:
        opt["sigma"] = normalize_iv(opt[col_iv])
    else:
        opt["sigma"] = SIGMA_OVERRIDE

    # Option type
    if FORCE_OPTION_TYPE in ("call", "put"):
        opt_type = FORCE_OPTION_TYPE
    else:
        opt_type = infer_option_type_from_delta(opt.get(col_delta, pd.Series(dtype=float)))

    # Aggregate to one row per day (average mid & IV if duplicates)
    opt_daily = (opt.groupby(col_date, as_index=False)
                    .agg({"mid":"mean", "K":"first", "sigma":"mean"})
                    .rename(columns={col_date: "date"}))

    # ---------- Load & parse S&P (MM/DD/YYYY) ----------
    spx = pd.read_csv(SPX_CSV)
    spx["Date"]  = pd.to_datetime(spx["Date"], format="%m/%d/%Y", errors="coerce")
    spx["Price"] = spx["Price"].apply(to_float_clean)
    spx_daily = spx[["Date","Price"]].dropna(subset=["Date"]).rename(columns={"Date":"date","Price":"S"})

    # ---------- Align on dates ----------
    df = pd.merge(opt_daily, spx_daily, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Params
    df["r"] = R
    df["q"] = Q
    df["T"] = T_DAYS / 365.0
    print("ooo asaaa")
    print(df["T"])
    df["option_type"] = opt_type

    # ---------- Price with Monte Carlo ----------
    def mc_price_row(row):
        # Risk-neutral GBM uses r and q; our MC function expects r only.
        # If you want to include a continuous dividend yield q, you can replace r->(r - q)
        # in the drift inside montecarlo.py. For now, we approximate by passing r-q here:
        eff_r = row["r"] - row["q"]
        res = monte_carlo_euro(
            S0=row["S"], K=row["K"], T=row["T"], r=eff_r, sigma=row["sigma"],
            option=row["option_type"], n_sims=N_SIMS, antithetic=ANTITHETIC, seed=SEED
        )
        return res.price

    df["model_mc"] = df.apply(mc_price_row, axis=1)
    df["residual"] = df["mid"] - df["model_mc"]

    # ---------- Save ----------
    out_csv = "mc_vs_market_daily.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(df.head(8))

    # ---------- Plots (same set as binomial) ----------
    # 1) Time series: market mid vs model
    plt.figure(figsize=(10,5))
    plt.plot(df["date"], df["mid"], label="Market mid")
    plt.plot(df["date"], df["model_mc"], label="MC model")
    plt.title("Daily Option Price: Market Mid vs Monte Carlo")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Residuals over time
    plt.figure(figsize=(10,4))
    plt.plot(df["date"], df["residual"], label="Residual (mid - MC)")
    plt.axhline(0, linestyle="--")
    plt.title("Pricing Residuals Over Time (Monte Carlo)")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()

    # 3) Scatter: model vs market mid
    plt.figure(figsize=(6,6))
    plt.scatter(df["mid"], df["model_mc"])
    mn = np.nanmin([df["mid"].min(), df["model_mc"].min()])
    mx = np.nanmax([df["mid"].max(), df["model_mc"].max()])
    plt.plot([mn, mx], [mn, mx])
    plt.title("MC Model vs Market Mid")
    plt.xlabel("Market mid")
    plt.ylabel("MC model")
    plt.tight_layout()
    plt.show()

    # ---------- (Kept) Original MC demo plots — commented out ----------
    # from montecarlo import (
    #     mc_simulate_trace, plot_terminal_ST, plot_payoff_distribution,
    #     plot_convergence, plot_Z_qq, compare_antithetic_vs_plain,
    #     plot_mc_calls_and_puts
    # )
    # # Example: uncomment to view the generic MC diagnostics (single-parameter, not time series)
    # S0, T, r, sigma = 100.0, 0.25, 0.05, 0.20
    # trace = mc_simulate_trace(S0, 100.0, T, r, sigma, option="call", n_sims=200_000, antithetic=True, seed=7)
    # plot_terminal_ST(trace, S0, T, r, sigma)
    # plot_payoff_distribution(trace, option="call")
    # plot_convergence(S0, 100.0, T, r, sigma, option="call")
    # plot_Z_qq(trace)
    # compare_antithetic_vs_plain(S0, 100.0, T, r, sigma, option="call")
    # plot_mc_calls_and_puts(S0, T, r, sigma, n_sims=150_000, antithetic=True, seed=11)

if __name__ == "__main__":
    main()
