"""Daily prediction sheet — the bot's output.

For the latest date on disk (or --date), emits one row per liquid contract:

  settle, iv                the market
  iv_fit, resid_vol         SVI fair-value IV and the contract's deviation
                            from it (vol points; +ve = rich vs the smile)
  resid_z                   residual standardized by that expiry's residual
                            cross-section (crude rich/cheap score)
  delta, vega, theta_day    Black-76 Greeks at the observed IV
  pred_1d, pred_5d          predicted settles: sticky-strike IV (the
                            backtest-winning model), forward unchanged,
                            time rolled forward -> pure carry/decay path

Backtested accuracy of the prediction columns (h=5, conditioning on the
realized forward): median vega-normalized error ~0.7 vol pts on NIFTY.
The unconditional error is dominated by underlying moves, which nothing
here forecasts. Residuals are model diagnostics, not trade signals.

Output: CSV in $NSE_DATA_ROOT/reports/ plus a console summary.

Usage:
  python3 predict_today.py                      # latest date, NIFTY
  python3 predict_today.py --symbol BANKNIFTY --date 2026-07-17
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest_prices import b76
from build_surfaces import SURF_DIR, svi_w
from config import DATA_ROOT

REPORT_DIR = DATA_ROOT / "reports"
DAYS_IN_YEAR = 365.0
WING_CUT = 0.10  # |k| beyond this is a wing: shown, but excluded from rich/cheap


def latest_files(symbol: str, date: str | None):
    pat_p = f"svi_params/year=*/*_{symbol}.parquet"
    pat_i = f"iv_points/year=*/*_{symbol}.parquet"
    pf = sorted(SURF_DIR.glob(pat_p))
    if not pf:
        sys.exit(f"no surfaces for {symbol} — run build_surfaces.py first")
    if date:
        pf = [p for p in pf if p.name.startswith(date)]
        if not pf:
            sys.exit(f"no surface file for {symbol} on {date}")
    params = pd.read_parquet(pf[-1])
    d = pd.to_datetime(params["trade_date"].iloc[0])
    ifile = [p for p in sorted(SURF_DIR.glob(pat_i))
             if p.name == pf[-1].name]
    pts = pd.read_parquet(ifile[-1])
    return d, params, pts


def greeks(F, K, T, sig, D, is_call):
    v = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    delta = np.where(is_call, D * norm.cdf(d1), -D * norm.cdf(-d1))
    vega = D * F * norm.pdf(d1) * np.sqrt(T) / 100  # per vol point
    p_now = b76(F, K, T, sig, D, is_call)
    p_1d = b76(F, K, np.maximum(T - 1 / DAYS_IN_YEAR, 1e-4), sig, D, is_call)
    theta_day = p_1d - p_now
    return delta, vega, theta_day


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--date", help="YYYY-MM-DD (default: latest on disk)")
    args = ap.parse_args()
    sym = args.symbol

    d, params, pts = latest_files(sym, args.date)
    clean = params[params["butterfly_ok"]
                   & params.get("calendar_ok", True)].set_index("expiry")
    pts = pts[pts["expiry"].isin(clean.index)].copy()
    if pts.empty:
        sys.exit(f"{d.date()}: no clean expiries to report")

    # fitted smile IV per contract + prediction columns
    iv_fit = np.full(len(pts), np.nan)
    for exp_, sub in pts.groupby("expiry"):
        r = clean.loc[exp_]
        p5 = np.array([r["a"], r["b"], r["rho"], r["m"], r["sigma_svi"]])
        idx = pts.index.get_indexer(sub.index)
        iv_fit[idx] = np.sqrt(np.clip(svi_w(sub["k"].to_numpy(), p5),
                                      1e-6, None) / r["T"])
    pts["iv_fit"] = iv_fit
    pts["resid_vol"] = pts["iv"] - pts["iv_fit"]
    pts["resid_z"] = (pts.groupby("expiry")["resid_vol"]
                      .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)))

    is_call = (pts["opt_type"] == "CE").to_numpy()
    F, K = pts["F"].to_numpy(), pts["strike"].to_numpy()
    T, iv, D = pts["T"].to_numpy(), pts["iv"].to_numpy(), pts["D"].to_numpy()
    pts["delta"], pts["vega_pt"], pts["theta_day"] = greeks(F, K, T, iv, D, is_call)
    for h, col in ((1, "pred_1d"), (5, "pred_5d")):
        T2 = np.maximum(T - h / DAYS_IN_YEAR, 1e-4)
        pts[col] = b76(F, K, T2, iv, D, is_call)   # sticky strike, F unchanged

    cols = ["expiry", "strike", "opt_type", "k", "settle", "iv", "iv_fit",
            "resid_vol", "resid_z", "delta", "vega_pt", "theta_day",
            "pred_1d", "pred_5d", "oi", "volume"]
    sheet = pts[cols].sort_values(["expiry", "strike"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"pricesheet_{sym}_{d.date()}.csv"
    sheet.to_csv(out, index=False, float_format="%.4f")

    # console summary
    print(f"== {sym} price sheet for {d.date()} — "
          f"{len(sheet)} contracts, {sheet['expiry'].nunique()} expiries ==")
    for exp_, r in clean.iterrows():
        print(f"  {pd.Timestamp(exp_).date()}  T={r['T']*365:3.0f}d  "
              f"F={r['F']:.0f}  ATM IV={r['atm_iv']:.1%}  "
              f"fit RMSE={r['fit_rmse_iv_totvar']:.1e}")
    core = sheet[sheet["k"].abs() <= WING_CUT]
    show = ["expiry", "strike", "opt_type", "settle", "iv", "iv_fit",
            "resid_vol", "resid_z"]
    print("\n-- richest vs fitted smile (excl. wings) --")
    print(core.nlargest(5, "resid_z")[show].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n-- cheapest vs fitted smile (excl. wings) --")
    print(core.nsmallest(5, "resid_z")[show].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nfull sheet: {out}")
    print("note: residuals are fit diagnostics; EOD settles include stale "
          "prices, and no transaction costs are considered here.")


if __name__ == "__main__":
    main()
