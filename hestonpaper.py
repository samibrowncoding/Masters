import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize


# ============================================================
# 1. Dataclasses for parameters
# ============================================================

@dataclass
class HestonParamsRealWorld:
    kappa: float
    theta: float
    xi: float
    mu: float
    rho: float
    lam: float


@dataclass
class HestonParamsRiskNeutral:
    kappa: float
    theta: float
    xi: float
    rho: float


# ============================================================
# 2. Load your data
# ============================================================

def load_merged_spx_vix(
    spx_path: str = "Data/S&P 500 Historical Data 2015-2022.csv",
    vix_path: str = "Data/VIX9D_History.csv",
) -> pd.DataFrame:
    """
    Load SPX and 9-day VIX, clean, and merge on date.
    Returns a DataFrame with columns: Date, Price_clean, CLOSE (VIX), sorted by Date.
    """

    # --- SPX ---
    spx = pd.read_csv(spx_path)
    # Investing.com format: "10/03/2025" = mm/dd/yyyy
    spx["Date"] = pd.to_datetime(spx["Date"], format="%m/%d/%Y")
    # strip commas like "6,715.79" -> 6715.79
    spx["Price_clean"] = (
        spx["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    spx = spx[["Date", "Price_clean"]]

    # --- 9-day VIX ---
    vix = pd.read_csv(vix_path)
    vix["DATE"] = pd.to_datetime(vix["DATE"], format="%m/%d/%Y")
    # CLOSE is the index level
    vix = vix[["DATE", "CLOSE"]]

    # --- Merge ---
    df = pd.merge(
        spx,
        vix,
        left_on="Date",
        right_on="DATE",
        how="inner",
    )
    df = df.drop(columns=["DATE"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def load_option_series(opt_path: str = "data/30day134606830.csv") -> pd.DataFrame:
    """
    Load your single option file, parse dates as dd/mm/yyyy (OptionMetrics export),
    and compute the mid price.
    """
    opt = pd.read_csv(opt_path)

    # OptionMetrics uses European format in your file: "01/07/2021" = 1 July 2021
    opt["date_dt"] = pd.to_datetime(opt["date"], dayfirst=True)
    opt["exdate_dt"] = pd.to_datetime(opt["exdate"], dayfirst=True)

    # Mid price from bid/ask
    opt["mid"] = (opt["best_bid"] + opt["best_offer"]) / 2.0

    # Keep the fields we care about
    opt = opt[
        [
            "optionid",
            "date_dt",
            "exdate_dt",
            "strike_price",
            "best_bid",
            "best_offer",
            "mid",
        ]
    ].sort_values("date_dt")

    return opt


# ============================================================
# 3. Heston likelihood (real-world measure)
# ============================================================

def heston_neg_loglik(
    params: np.ndarray,
    X: np.ndarray,
    V: np.ndarray,
    dt: float,
) -> float:
    """
    Negative log-likelihood for the discretised Heston model under the real-world measure,
    following the Tomurcuk-style MLE (bivariate normal transition of (X_t, V_t)).

    params = [kappa, theta, xi, mu, rho, lam]
    X, V   = arrays of length N (daily observations).
    dt     = 1/252 for daily trading days.
    """
    kappa, theta, xi, mu, rho, lam = params

    # Basic parameter constraints to avoid nonsense regions
    if xi <= 0 or theta <= 0 or kappa <= 0 or abs(rho) >= 1:
        return 1e10

    eps = 1e-8
    V = np.clip(V, eps, None)

    # One-step transitions
    X_t = X[:-1]
    X_tp1 = X[1:]
    V_t = V[:-1]
    V_tp1 = V[1:]

    # Discretised dynamics (Euler)
    # X_{t+dt} = X_t + [mu + (lam - 0.5) V_t] dt + sqrt(V_t dt) * Z_X
    # V_{t+dt} = V_t + kappa(θ - V_t) dt + xi sqrt(V_t dt) * Z_V
    mu_X = X_t + (mu + (lam - 0.5) * V_t) * dt
    var_X = V_t * dt

    mu_V = V_t + kappa * (theta - V_t) * dt
    var_V = (xi**2) * V_t * dt

    sigma_X = np.sqrt(var_X)
    sigma_V = np.sqrt(var_V)

    z1 = X_tp1 - mu_X
    z2 = V_tp1 - mu_V

    one_minus_rho2 = 1.0 - rho**2

    term1 = np.log(2.0 * np.pi)
    term2 = np.log(xi)
    term3 = np.log(V_t)
    term4 = np.log(dt)
    term5 = 0.5 * np.log(one_minus_rho2)

    quad = (
        (z1**2) / var_X
        - 2.0 * rho * z1 * z2 / (sigma_X * sigma_V)
        + (z2**2) / var_V
    )

    ll_terms = term1 + term2 + term3 + term4 + term5 + quad / (2.0 * one_minus_rho2)

    neg_ll = float(np.sum(ll_terms))
    return neg_ll


def calibrate_heston_mle(
    X: np.ndarray,
    V: np.ndarray,
    dt: float = 1.0 / 252.0,
    initial_guess: np.ndarray | None = None,
) -> HestonParamsRealWorld:
    """
    Calibrate Heston params by MLE on (X_t, V_t) time series.
    Returns a HestonParamsRealWorld dataclass.
    """
    if initial_guess is None:
        initial_guess = np.array(
            [
                3.0,   # kappa
                0.04,  # theta
                0.5,   # xi
                0.05,  # mu
                -0.5,  # rho
                0.0,   # lam
            ],
            dtype=float,
        )

    bounds = [
        (1e-4, 20.0),      # kappa
        (1e-4, 2.0),       # theta
        (1e-4, 5.0),       # xi
        (-1.0, 1.0),       # mu
        (-0.999, 0.999),   # rho
        (-10.0, 10.0),     # lam
    ]

    res = minimize(
        heston_neg_loglik,
        initial_guess,
        args=(X, V, dt),
        bounds=bounds,
        method="L-BFGS-B",
    )

    if not res.success:
        print("WARNING: Heston MLE did not fully converge:", res.message)

    kappa, theta, xi, mu, rho, lam = res.x

    return HestonParamsRealWorld(
        kappa=float(kappa),
        theta=float(theta),
        xi=float(xi),
        mu=float(mu),
        rho=float(rho),
        lam=float(lam),
    )


# ============================================================
# 4. Risk-neutral Heston pricer (semi-analytic, corrected)
# ============================================================

import math
import cmath

def _heston_cf(
    phi: float,
    T: float,
    S0: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    j: int,
) -> complex:
    """
    Canonical Heston characteristic function for P_j (j=1,2).

    This follows the standard representation (Gatheral-style):
        C(T,φ), D(T,φ) with
        a = kappa * theta
        b_j = kappa - rho*xi for j=1, and kappa for j=2
        u_1 = 0.5, u_2 = -0.5
    """
    # Parameters u and b depend on j
    if j == 1:
        u = 0.5
        b = kappa - rho * xi
    elif j == 2:
        u = -0.5
        b = kappa
    else:
        raise ValueError("j must be 1 or 2")

    a = kappa * theta
    phi_c = complex(phi)

    # d(φ) term
    d = cmath.sqrt(
        (rho * xi * 1j * phi_c - b) ** 2
        - xi**2 * (2 * u * 1j * phi_c - phi_c**2)
    )

    # g(φ) term
    g = (b - rho * xi * 1j * phi_c + d) / (b - rho * xi * 1j * phi_c - d)

    # C(T,φ) and D(T,φ)
    exp_dT = cmath.exp(d * T)
    one_minus_gexp = 1.0 - g * exp_dT
    one_minus_g = 1.0 - g

    C = (
        r * 1j * phi_c * T
        + a / (xi**2)
        * ((b - rho * xi * 1j * phi_c + d) * T - 2.0 * cmath.log(one_minus_gexp / one_minus_g))
    )

    D = (
        (b - rho * xi * 1j * phi_c + d)
        / (xi**2)
        * ((1.0 - exp_dT) / one_minus_gexp)
    )

    # Characteristic function
    return cmath.exp(C + D * v0 + 1j * phi_c * math.log(S0))


def _heston_Pj(
    j: int,
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    phi_max: float = 100.0,
    n_phi: int = 2000,
) -> float:
    """
    Probability P_j using numerical integration of the Heston characteristic function.
    """
    if T <= 0.0:
        # At maturity, probabilities collapse: P1=P2=1 if in-the-money, else 0
        return 1.0 if S0 > K else 0.0

    phis = np.linspace(1e-5, phi_max, n_phi)
    logK = math.log(K)

    integrand = []
    for phi in phis:
        cf_val = _heston_cf(phi, T, S0, r, v0, kappa, theta, xi, rho, j)
        integrand.append(
            (cmath.exp(-1j * phi * logK) * cf_val / (1j * phi)).real
        )

    integral = np.trapz(integrand, phis)
    P = 0.5 + (1.0 / math.pi) * integral
    return float(P)


def heston_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    params_q: HestonParamsRiskNeutral,
    phi_max: float = 100.0,
    n_phi: int = 2000,
) -> float:
    """
    Risk-neutral Heston European call price.

    We also enforce a final sanity check: price >= 0 and
    not below discounted intrinsic value.
    """
    kappa = params_q.kappa
    theta = params_q.theta
    xi = params_q.xi
    rho = params_q.rho

    if T <= 0.0:
        return max(S0 - K, 0.0)

    P1 = _heston_Pj(1, S0, K, T, r, v0, kappa, theta, xi, rho, phi_max, n_phi)
    P2 = _heston_Pj(2, S0, K, T, r, v0, kappa, theta, xi, rho, phi_max, n_phi)

    price = S0 * P1 - K * math.exp(-r * T) * P2

    # Sanity clip: never negative & not below discounted intrinsic value
    intrinsic_disc = max(S0 - K * math.exp(-r * T), 0.0)
    price = max(price, intrinsic_disc, 0.0)

    return float(price)


# ============================================================
# 5. Putting it all together for your option
# ============================================================

def main():
    # -----------------------------
    # Load merged SPX + 9-day VIX
    # -----------------------------
    df = load_merged_spx_vix()
    # df has: Date, Price_clean, CLOSE (VIX index)

    # Build log-price and variance
    X_all = np.log(df["Price_clean"].values)
    sigma_all = df["CLOSE"].values / 100.0
    V_all = sigma_all**2
    dates_all = df["Date"].values

    # -----------------------------
    # Calibration window (global)
    # Here we copy the paper's spirit:
    # 2015-10-02 to 2021-06-30
    # -----------------------------
    calib_end = pd.Timestamp("2021-06-30")
    mask_calib = df["Date"] <= calib_end
    X_cal = X_all[mask_calib.values]
    V_cal = V_all[mask_calib.values]

    dt = 1.0 / 252.0

    print(f"Calibration window: {df['Date'][mask_calib].min().date()} "
          f"to {df['Date'][mask_calib].max().date()} "
          f"(N={len(X_cal)})")

    params_p = calibrate_heston_mle(X_cal, V_cal, dt=dt)
    print("\n=== Calibrated real-world Heston parameters (global) ===")
    print(params_p)

    # Risk-neutral parameters: drop mu and lam
    params_q = HestonParamsRiskNeutral(
        kappa=params_p.kappa,
        theta=params_p.theta,
        xi=params_p.xi,
        rho=params_p.rho,
    )

    # -----------------------------
    # Load your option time series
    # -----------------------------
    opt = load_option_series()
    start = opt["date_dt"].min()
    end = opt["date_dt"].max()
    expiry = opt["exdate_dt"].iloc[0]

    print(f"\nOption life from {start.date()} to {end.date()} "
          f"(N={len(opt)}) with expiry {expiry.date()}")

    # Option strike: OptionMetrics uses scaled strike, typically /1000 for SPX
    raw_strike = opt["strike_price"].iloc[0]
    K = raw_strike / 1000.0

    # Merge option days with SPX+VIX to get S_t and VIX_t on those dates
    df_life = df.merge(
        opt[["date_dt", "mid"]],
        left_on="Date",
        right_on="date_dt",
        how="inner",
    ).sort_values("Date")

    # Keep only within option life
    mask_life = (df_life["Date"] >= start) & (df_life["Date"] <= end)
    df_life = df_life[mask_life].copy().reset_index(drop=True)

    # Check we have same number of days
    print(f"Joined option/SPX/VIX rows over life: {len(df_life)}")

    # -----------------------------
    # Price the same option each day
    # -----------------------------
    # Risk-free rate (constant – change if you have a curve)
    r = 0.01

    model_prices = []
    maturities = []

    for i, row in df_life.iterrows():
        date = row["Date"]
        S_t = row["Price_clean"]
        vix_t = row["CLOSE"]

        # Time to maturity in years: calendar days / 252 (trading-year)
        tau_days = (expiry - date).days
        # if tau_days <= 0, option is at or past expiry -> price ~ intrinsic; we stop
        if tau_days <= 0:
            T = 0.0
        else:
            T = tau_days / 252.0

        maturities.append(T)

        # v0 from 9-day VIX
        sigma_t = vix_t / 100.0
        v0 = sigma_t**2

        if T > 0.0:
            price = heston_call_price(S_t, K, T, r, v0, params_q)
        else:
            # At expiry, price is intrinsic
            price = max(S_t - K, 0.0)

        model_prices.append(price)

    df_life["T"] = maturities
    df_life["heston_price"] = model_prices

    # -----------------------------
    # Basic error metrics
    # -----------------------------
    market = df_life["mid"].values
    model = df_life["heston_price"].values

    rmse = np.sqrt(np.mean((model - market) ** 2))
    mae = np.mean(np.abs(model - market))
    bias = np.mean(model - market)

    print("\n=== Heston pricing metrics on your single option time-series ===")
    print(f"Rows used: {len(df_life)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"Bias (model - market): {bias:.6f}")

    # Show a few rows so you can see the fit
    print("\nHead of results (Date, S, VIX9D, T, market_mid, heston_price):")
    print(
        df_life[
            ["Date", "Price_clean", "CLOSE", "T", "mid", "heston_price"]
        ].head(10)
    )


if __name__ == "__main__":
    main()
