"""Build implied-vol surfaces from normalized EOD F&O parquet.

Per (date, expiry):
  1. Liquidity filter on the option chain
  2. Implied forward F and discount factor D via put-call parity regression
  3. Black-76 implied vols per strike
  4. Raw-SVI fit of total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2+s^2))
  5. Static-arbitrage diagnostics (butterfly g(k) >= 0, calendar monotonicity)

Outputs under $NSE_DATA_ROOT/parquet/surfaces/:
  iv_points/year=YYYY/DATE.parquet   per-strike IVs (k, iv, w, oi, vega, ...)
  svi_params/year=YYYY/DATE.parquet  per-expiry SVI params + fit diagnostics

Usage:
  python3 build_surfaces.py --date 2026-07-17            # one day
  python3 build_surfaces.py --all                        # every day on disk
  python3 build_surfaces.py --all --symbol BANKNIFTY
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from black76 import bs_vega, implied_vol
from config import DATA_ROOT, PARQUET_EOD_DIR

SURF_DIR = DATA_ROOT / "parquet" / "surfaces"

MIN_OI = 100            # contracts; drop dust strikes
MIN_PARITY_PAIRS = 5    # strikes with both liquid legs needed for the forward
MIN_FIT_POINTS = 8      # IV points needed to attempt an SVI fit
MONEYNESS_BAND = 0.25   # keep |ln(K/F)| below this
DAYS_IN_YEAR = 365.0


# ---------------------------------------------------------------- forwards --

def parity_forward(chain: pd.DataFrame) -> tuple[float, float, int] | None:
    """Implied forward via C - P = D*F - D*K regressed across strikes.
    Returns (F, D, n_pairs) or None."""
    wide = chain.pivot_table(index="strike", columns="opt_type",
                             values="settle", aggfunc="first")
    if not {"CE", "PE"}.issubset(wide.columns):
        return None
    wide = wide.dropna()
    spot = chain["underlying_close"].iloc[0]
    if np.isfinite(spot):  # parity is most reliable near ATM
        wide = wide[(wide.index > 0.85 * spot) & (wide.index < 1.15 * spot)]
    if len(wide) < MIN_PARITY_PAIRS:
        return None
    K = wide.index.to_numpy(float)
    y = (wide["CE"] - wide["PE"]).to_numpy(float)
    slope, intercept = np.polyfit(K, y, 1)
    D, F = -slope, intercept / -slope if slope < 0 else (np.nan, np.nan)
    if not (0.7 < D <= 1.02) or not np.isfinite(F) or F <= 0:
        return None
    return float(F), float(min(D, 1.0)), len(wide)


# --------------------------------------------------------------------- svi --

def svi_w(k: np.ndarray, p: np.ndarray) -> np.ndarray:
    a, b, rho, m, s = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def fit_svi(k: np.ndarray, w: np.ndarray, weights: np.ndarray) -> dict | None:
    """Weighted raw-SVI fit with parameter bounds keeping it sane."""
    w_max = w.max()
    x0 = np.array([0.5 * w.min(), 0.1, -0.3, 0.0, 0.1])
    lb = np.array([1e-6, 1e-6, -0.999, -1.0, 1e-4])
    ub = np.array([w_max, 1.0, 0.999, 1.0, 1.0])

    def resid(p):
        return np.sqrt(weights) * (svi_w(k, p) - w)

    try:
        res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=2000)
    except Exception:
        return None
    if not res.success and res.cost > 1e-2:
        return None
    p = res.x
    fitted = svi_w(k, p)
    rmse_iv = float(np.sqrt(np.mean((np.sqrt(fitted) - np.sqrt(w)) ** 2)))
    return {"a": p[0], "b": p[1], "rho": p[2], "m": p[3], "sigma_svi": p[4],
            "fit_rmse_iv_totvar": rmse_iv, "n_points": len(k)}


def butterfly_ok(p: np.ndarray, k_grid: np.ndarray) -> bool:
    """Gatheral's g(k) >= 0 check for absence of butterfly arbitrage."""
    a, b, rho, m, s = p
    w = svi_w(k_grid, p)
    root = np.sqrt((k_grid - m) ** 2 + s ** 2)
    w1 = b * (rho + (k_grid - m) / root)          # dw/dk
    w2 = b * s ** 2 / root ** 3                    # d2w/dk2
    with np.errstate(divide="ignore", invalid="ignore"):
        g = ((1 - k_grid * w1 / (2 * w)) ** 2
             - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2)
    return bool(np.nanmin(g) >= -1e-6)


# ------------------------------------------------------------------- daily --

def build_day(df_day: pd.DataFrame, symbol: str, trade_date) -> tuple[pd.DataFrame, pd.DataFrame]:
    opts = df_day[(df_day["symbol"] == symbol)
                  & (df_day["instrument"].isin(["IDO", "STO"]))
                  & (df_day["volume"] > 0)
                  & (df_day["oi"] >= MIN_OI)
                  & (df_day["settle"] > 0)].copy()

    iv_rows, param_rows = [], []
    for expiry, chain in opts.groupby("expiry"):
        T = (expiry - trade_date).days / DAYS_IN_YEAR
        if T <= 1 / DAYS_IN_YEAR:
            continue
        fwd = parity_forward(chain)
        if fwd is None:
            continue
        F, D, n_pairs = fwd

        for _, r in chain.iterrows():
            k = np.log(r["strike"] / F)
            if abs(k) > MONEYNESS_BAND:
                continue
            # OTM side only: cleaner prices, and parity makes ITM redundant
            is_call = r["opt_type"] == "CE"
            if (is_call and k < 0) or (not is_call and k > 0):
                continue
            iv = implied_vol(r["settle"], F, r["strike"], T, D, is_call)
            if not np.isfinite(iv) or not (0.03 < iv < 2.0):
                continue
            iv_rows.append(dict(trade_date=trade_date, symbol=symbol,
                                expiry=expiry, T=T, F=F, D=D,
                                strike=r["strike"], opt_type=r["opt_type"],
                                k=k, iv=iv, w=iv * iv * T, oi=r["oi"],
                                volume=r["volume"], settle=r["settle"],
                                vega=bs_vega(F, r["strike"], T, iv, D)))

        pts = [x for x in iv_rows if x["expiry"] == expiry]
        if len(pts) < MIN_FIT_POINTS:
            continue
        kk = np.array([x["k"] for x in pts])
        ww = np.array([x["w"] for x in pts])
        wt = np.array([x["vega"] for x in pts])
        wt = wt / wt.sum()
        fit = fit_svi(kk, ww, wt)
        if fit is None:
            continue
        p = np.array([fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma_svi"]])
        grid = np.linspace(kk.min(), kk.max(), 101)
        atm_iv = float(np.sqrt(svi_w(np.array([0.0]), p)[0] / T))
        param_rows.append(dict(trade_date=trade_date, symbol=symbol,
                               expiry=expiry, T=T, F=F, D=D,
                               n_parity_pairs=n_pairs, atm_iv=atm_iv,
                               butterfly_ok=butterfly_ok(p, grid), **fit))

    params = pd.DataFrame(param_rows)
    # calendar check: total ATM variance should be non-decreasing in T
    if len(params) > 1:
        srt = params.sort_values("T")
        w_atm = (srt["atm_iv"] ** 2 * srt["T"]).to_numpy()
        cal = np.concatenate([[True], np.diff(w_atm) >= -1e-6])
        params.loc[srt.index, "calendar_ok"] = cal
    elif len(params) == 1:
        params["calendar_ok"] = True
    return pd.DataFrame(iv_rows), params


# --------------------------------------------------------------------- cli --

def run_date(path: Path, symbol: str) -> None:
    df = pd.read_parquet(path)
    trade_date = df["trade_date"].dropna().iloc[0]
    iv, params = build_day(df, symbol, trade_date)
    if params.empty:
        print(f"[surfaces] {trade_date.date()} {symbol}: no fittable expiries",
              file=sys.stderr)
        return
    d = trade_date.date()
    for name, frame in (("iv_points", iv), ("svi_params", params)):
        out_dir = SURF_DIR / name / f"year={d.year}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # one file per (day, symbol) so multiple symbols don't clobber
        frame.to_parquet(out_dir / f"{d.isoformat()}_{symbol}.parquet", index=False)
    flags = params[~(params["butterfly_ok"] & params["calendar_ok"])]
    note = f", {len(flags)} expiry(ies) flagged for arbitrage" if len(flags) else ""
    print(f"[surfaces] {d} {symbol}: {len(params)} expiries fit, "
          f"{len(iv)} IV points{note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--date", help="YYYY-MM-DD")
    ap.add_argument("--all", action="store_true", help="process every EOD file")
    args = ap.parse_args()

    if args.all:
        files = sorted(PARQUET_EOD_DIR.glob("year=*/*.parquet"))
    elif args.date:
        files = sorted(PARQUET_EOD_DIR.glob(f"year=*/{args.date}.parquet"))
    else:
        sys.exit("need --date or --all")
    if not files:
        sys.exit("no matching EOD parquet files found")
    for f in files:
        try:
            run_date(f, args.symbol)
        except Exception as e:
            print(f"[surfaces] {f.name}: ERROR {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
