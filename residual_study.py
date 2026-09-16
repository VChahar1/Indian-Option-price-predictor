"""Do rich/cheap residuals vs the fitted smile mean-revert?

For contracts liquid at both t and t+h, define resid = iv - iv_fit (vol pts,
fit from that day's SVI). Three questions, each with an honesty control:

1. REVERSION: Fama-MacBeth cross-sectional regressions of resid_{t+h} on
   resid_t, daily slopes averaged with a HAC t-stat. beta = 1 means residuals
   are permanent features of the strike; beta = 0 means full reversion
   within h days.
2. WHO MOVES: regress (iv_{t+h} - iv_t) on resid_t   -> contract converges?
              regress (fit_{t+h} - fit_t) on resid_t -> smile chases?
   (slopes sum to beta - 1 by construction)
3. STALENESS CONTROL: reversion beta by same-day volume tercile. If
   reversion lives only in the low-volume tercile, it is stale prints
   snapping back, not a tradable effect.

Plus a quintile sort: convergence capture in vega-rupees per contract for
top/bottom resid_z quintiles — an upper bound on gross economics, since EOD
data has no bid/ask and no costs.

Usage:
  python3 residual_study.py --symbol NIFTY --horizons 1 2 5 10
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from build_surfaces import SURF_DIR, svi_w
from config import DATA_ROOT

WING_CUT = 0.10
MIN_TENOR_D, MAX_TENOR_D = 5, 60
MIN_XS = 30          # contracts needed for a daily cross-section


# ------------------------------------------------------------------- panel --

def load_points_with_fit(symbol: str) -> pd.DataFrame:
    pfiles = sorted(SURF_DIR.glob(f"svi_params/year=*/*_{symbol}.parquet"))
    ifiles = sorted(SURF_DIR.glob(f"iv_points/year=*/*_{symbol}.parquet"))
    if not pfiles:
        sys.exit(f"no surfaces for {symbol}")
    params = pd.concat((pd.read_parquet(f) for f in pfiles), ignore_index=True)
    pts = pd.concat((pd.read_parquet(f) for f in ifiles), ignore_index=True)
    for df in (params, pts):
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["expiry"] = pd.to_datetime(df["expiry"])
    params = params[params["butterfly_ok"] & params.get("calendar_ok", True)]
    key = ["trade_date", "expiry"]
    pts = pts.merge(params[key + ["a", "b", "rho", "m", "sigma_svi", "T"]]
                    .rename(columns={"T": "T_fit"}), on=key, how="inner")
    p5 = pts[["a", "b", "rho", "m", "sigma_svi"]].to_numpy()
    w = svi_w(pts["k"].to_numpy(), p5.T)
    pts["iv_fit"] = np.sqrt(np.clip(w, 1e-6, None) / pts["T_fit"])
    pts["resid"] = pts["iv"] - pts["iv_fit"]
    pts = pts[(pts["k"].abs() <= WING_CUT)
              & (pts["T"] * 365 >= MIN_TENOR_D)
              & (pts["T"] * 365 <= MAX_TENOR_D)]
    return pts


def paired_panel(pts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    dates = sorted(pts["trade_date"].unique())
    nxt = {dates[i]: dates[i + horizon] for i in range(len(dates) - horizon)}
    a = pts.copy()
    a["date_h"] = a["trade_date"].map(nxt)
    b = pts[["trade_date", "expiry", "strike", "opt_type",
             "iv", "iv_fit", "resid"]].rename(
        columns={"trade_date": "date_h", "iv": "iv_h",
                 "iv_fit": "fit_h", "resid": "resid_h"})
    m = a.merge(b, on=["date_h", "expiry", "strike", "opt_type"], how="inner")
    m["d_iv"] = m["iv_h"] - m["iv"]
    m["d_fit"] = m["fit_h"] - m["iv_fit"]
    return m


# ------------------------------------------------------------ fama-macbeth --

def fm_slope(panel: pd.DataFrame, ycol: str, horizon: int,
             min_xs: int = MIN_XS) -> tuple[float, float, int]:
    """Daily cross-sectional OLS slope of ycol on resid; HAC mean t-stat."""
    slopes = []
    for _, g in panel.groupby("trade_date"):
        if len(g) < min_xs:
            continue
        x, y = g["resid"].to_numpy(), g[ycol].to_numpy()
        vx = np.var(x)
        if vx < 1e-12:
            continue
        slopes.append(np.cov(x, y, bias=True)[0, 1] / vx)
    s = np.array(slopes)
    n = len(s)
    if n < 30:
        return np.nan, np.nan, n
    mean = s.mean()
    # Newey-West on the daily slope series, lag = horizon
    d = s - mean
    var = np.mean(d ** 2)
    for lag in range(1, horizon + 1):
        w = 1 - lag / (horizon + 1)
        var += 2 * w * np.mean(d[lag:] * d[:-lag])
    t = mean / np.sqrt(max(var, 1e-12) / n)
    return float(mean), float(t), n


# ----------------------------------------------------------------- reports --

def run(symbol: str, horizons: list[int]) -> None:
    pts = load_points_with_fit(symbol)
    print(f"== {symbol} residual study ==")
    print(f"universe: |k|<={WING_CUT}, tenor {MIN_TENOR_D}-{MAX_TENOR_D}d, "
          f"{pts['trade_date'].nunique()} days, "
          f"median |resid| = {pts['resid'].abs().median()*100:.2f} vol pts")

    rows = []
    for h in horizons:
        panel = paired_panel(pts, h)
        beta, t_b, n = fm_slope(panel, "resid_h", h)
        g_c, t_c, _ = fm_slope(panel, "d_iv", h)
        g_f, t_f, _ = fm_slope(panel, "d_fit", h)
        rows.append(dict(h=h, days=n, pairs=len(panel),
                         beta=beta, t_beta=t_b,
                         contract_slope=g_c, t_contract=t_c,
                         smile_slope=g_f, t_smile=t_f))
    tbl = pd.DataFrame(rows).set_index("h")
    print("\n-- Fama-MacBeth: resid_{t+h} ~ resid_t --")
    print(tbl.to_string(float_format=lambda x: f"{x:.3f}"))
    print("(beta=1: permanent strike feature; beta=0: full reversion.\n"
          " contract_slope < 0: contract IV converges to smile;\n"
          " smile_slope > 0: smile chases the contract.)")

    # staleness control at the main horizon
    h = horizons[len(horizons) // 2]
    panel = paired_panel(pts, h)
    panel["vol_terc"] = (panel.groupby("trade_date")["volume"]
                         .transform(lambda x: pd.qcut(x.rank(method="first"),
                                                      3, labels=["low", "mid", "high"])))
    print(f"\n-- reversion beta by volume tercile (h={h}) --")
    for terc in ["low", "mid", "high"]:
        b, t, n = fm_slope(panel[panel["vol_terc"] == terc], "resid_h", h,
                           min_xs=max(MIN_XS // 3, 10))
        print(f"  {terc:>4}: beta={b:.3f}  t={t:.2f}  days={n}")
    print("(reversion only in 'low' => stale prints, not signal)")

    # quintile convergence capture, vega-rupees, gross of everything
    panel["q"] = (panel.groupby("trade_date")["resid"]
                  .transform(lambda x: pd.qcut(x.rank(method="first"), 5,
                                               labels=False)))
    panel["capture_rs"] = (-np.sign(panel["resid"])
                           * (panel["resid_h"] - panel["resid"])
                           * panel["vega"])
    q = (panel.groupby("q")
         .agg(mean_resid=("resid", "mean"),
              capture_rs=("capture_rs", "mean"),
              n=("capture_rs", "size")))
    q.index = ["cheapest", "q2", "q3", "q4", "richest"]
    print(f"\n-- convergence capture by resid quintile (h={h}, "
          f"vega-rupees per contract, GROSS) --")
    print(q.to_string(float_format=lambda x: f"{x:.3f}"))
    print("caveat: no bid/ask, no costs, no fills — an upper bound on gross "
          "convergence, not P&L. NSE option spreads alone are typically "
          "rupees-per-contract at these strikes.")

    out = DATA_ROOT / "parquet" / "backtests" / f"{symbol}_residual_study.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl.reset_index().to_parquet(out, index=False)
    print(f"\nsummary saved to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 5, 10])
    args = ap.parse_args()
    run(args.symbol, args.horizons)


if __name__ == "__main__":
    main()
