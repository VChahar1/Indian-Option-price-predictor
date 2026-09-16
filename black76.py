"""Black-76 pricing and implied volatility on forwards.

Working on forwards (extracted from put-call parity) rather than spot means
dividends and the funding rate are handled implicitly — the right choice for
EOD index options where neither is cleanly observable for free.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black76_price(F: float, K: float, T: float, sigma: float,
                  D: float, is_call: bool) -> float:
    """Discounted Black-76 price. D is the discount factor to expiry."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return D * intrinsic
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if is_call:
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price: float, F: float, K: float, T: float,
                D: float, is_call: bool) -> float:
    """Implied vol via Brent bracketing. Returns NaN when no vol reproduces
    the price (below intrinsic / stale settle), which is common in EOD data
    and should be filtered, not fudged."""
    if T <= 0 or price <= 0 or F <= 0 or K <= 0:
        return np.nan
    intrinsic = D * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    if price <= intrinsic + 1e-10:
        return np.nan

    def f(sigma: float) -> float:
        return black76_price(F, K, T, sigma, D, is_call) - price

    lo, hi = 1e-4, 5.0
    try:
        if f(lo) > 0 or f(hi) < 0:
            return np.nan
        return brentq(f, lo, hi, xtol=1e-8, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


def bs_vega(F: float, K: float, T: float, sigma: float, D: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    return D * F * norm.pdf(d1) * np.sqrt(T)
