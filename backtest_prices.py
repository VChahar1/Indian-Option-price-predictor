"""Price-space backtest: predict option prices h trading days ahead, score
against realized settles.

For every liquid contract on day t (from iv_points), predict its settle on
day t+h under competing surface-dynamics assumptions, conditioning on the
realized forward F' at t+h — so the score isolates *vol-surface* prediction
skill from unforecastable underlying moves:

  no_change      : P' = P_t                      (the naive benchmark)
  sticky_strike  : contract keeps its own IV; reprice at (F', T')
  sticky_money   : today's SVI smile evaluated at k' = ln(K/F'); reprice
  sm_shifted     : sticky_money IV scaled by an AR(1) mean-reversion
                   forecast of 30d ATM IV (expanding fit, bounded shift)

Scores: rupee RMSE/MAE, vega-normalized error (= implied-vol error, in vol
points), win rate vs no_change, bucketed by tenor and moneyness.

Usage:
  python3 backtest_prices.py --symbol NIFTY --horizon 5
"""
from __future__ import annotations

import argparse
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

from build_surfaces import SURF_DIR, svi_w
from config import DATA_ROOT, PARQUET_EOD_DIR
from forecast import load_cm_iv

MIN_T2_DAYS = 2        # skip contracts expiring within the horizon
SHIFT_BOUNDS = (0.75, 1.33)
AR_BURN_IN = 120       # days of iv30 history before sm_shifted activates
DAYS_IN_YEAR = 365.0


# ------------------------------------------------------------- vector B-76 --

def b76(F, K, T, sig, D, is_call):
    F, K, T, sig, D = (np.asarray(x, float) for x in (F, K, T, sig, D))
    v = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    call = D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)


# ------------------------------------------------------------------- loads --

def load_settles(symbol: str) -> pd.DataFrame:
    q = f"""
      SELECT trade_date, expiry, strike, opt_type, settle, volume
      FROM read_parquet('{PARQUET_EOD_DIR}/year=*/*.parquet')
      WHERE symbol = '{symbol}' AND instrument IN ('IDO','STO')
        AND settle > 0
    """
    df = duckdb.sql(q).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


def load_surfaces(symbol: str):
    files = sorted(SURF_DIR.glob(f"svi_params/year=*/*_{symbol}.parquet"))
    if not files:
        sys.exit(f"no svi_params for {symbol}")
    params = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    params["trade_date"] = pd.to_datetime(params["trade_date"])
    params["expiry"] = pd.to_datetime(params["expiry"])
    pts_files = sorted(SURF_DIR.glob(f"iv_points/year=*/*_{symbol}.parquet"))
    pts = pd.concat((pd.read_parquet(f) for f in pts_files), ignore_index=True)
    pts["trade_date"] = pd.to_datetime(pts["trade_date"])
    pts["expiry"] = pd.to_datetime(pts["expiry"])
    return params, pts


# ------------------------------------------------------------- AR(1) shift --

def iv_shift_series(iv30: pd.Series, dates: list, horizon: int,
                    refit_every: int = 21) -> dict:
    """Expanding direct-h-step AR fit on log iv30 -> multiplicative shift
    factor per date. Returns {} entries only after burn-in."""
    log_iv = np.log(iv30.reindex(pd.DatetimeIndex(dates)).ffill(limit=3))
    x = log_iv.to_numpy()
    shifts, beta = {}, None
    for i, d in enumerate(dates):
        if i < AR_BURN_IN or not np.isfinite(x[i]):
            continue
        if beta is None or i % refit_every == 0:
            xs, ys = x[: i - horizon], x[horizon: i]
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.sum() < 60:
                continue
            beta = np.polyfit(xs[ok], ys[ok], 1)
        pred = beta[0] * x[i] + beta[1]
        shifts[d] = float(np.clip(np.exp(pred - x[i]), *SHIFT_BOUNDS))
    return shifts


# ---------------------------------------------------------------- backtest --

def run(symbol: str, horizon: int) -> pd.DataFrame:
    params, pts = load_surfaces(symbol)
    settles = load_settles(symbol)
    iv30 = load_cm_iv(symbol)

    dates = sorted(params["trade_date"].unique())
    shifts = iv_shift_series(iv30, dates, horizon)
    p_by_date = {d: g for d, g in params.groupby("trade_date")}
    pts_by_date = {d: g for d, g in pts.groupby("trade_date")}
    set_by_date = {d: g for d, g in settles.groupby("trade_date")}

    recs = []
    for i in range(len(dates) - horizon):
        t, t2 = dates[i], dates[i + horizon]
        g_t = pts_by_date.get(t)
        prm_t = p_by_date.get(t)
        prm_t2 = p_by_date.get(t2)
        st_t2 = set_by_date.get(t2)
        if any(x is None for x in (g_t, prm_t, prm_t2, st_t2)):
            continue
        clean = prm_t[prm_t["butterfly_ok"]
                      & prm_t.get("calendar_ok", True)].set_index("expiry")
        fwd2 = prm_t2.set_index("expiry")[["F", "D"]]

        m = g_t.merge(st_t2[["expiry", "strike", "opt_type", "settle", "volume"]],
                      on=["expiry", "strike", "opt_type"],
                      suffixes=("", "_t2"))
        m = m[(m["volume_t2"] > 0) & m["expiry"].isin(clean.index)
              & m["expiry"].isin(fwd2.index)].copy()
        if m.empty:
            continue
        m["T2"] = (m["expiry"] - pd.Timestamp(t2)).dt.days / DAYS_IN_YEAR
        m = m[m["T2"] * 365 >= MIN_T2_DAYS]
        if m.empty:
            continue
        m["F2"] = m["expiry"].map(fwd2["F"])
        m["D2"] = m["expiry"].map(fwd2["D"])
        is_call = (m["opt_type"] == "CE").to_numpy()

        # sticky strike: contract's own IV today, repriced at (F2, T2)
        m["p_ss"] = b76(m["F2"], m["strike"], m["T2"], m["iv"], m["D2"], is_call)

        # sticky moneyness: today's smile at k' = ln(K/F2)
        iv_sm = np.full(len(m), np.nan)
        for exp_, sub in m.groupby("expiry"):
            r = clean.loc[exp_]
            p5 = np.array([r["a"], r["b"], r["rho"], r["m"], r["sigma_svi"]])
            k2 = np.log(sub["strike"].to_numpy(float) / sub["F2"].to_numpy(float))
            iv_sm[m.index.get_indexer(sub.index)] = np.sqrt(
                np.clip(svi_w(k2, p5), 1e-6, None) / r["T"])
        m["iv_sm"] = iv_sm
        m["p_sm"] = b76(m["F2"], m["strike"], m["T2"], m["iv_sm"], m["D2"], is_call)

        s = shifts.get(t, np.nan)
        m["p_shift"] = (b76(m["F2"], m["strike"], m["T2"], m["iv_sm"] * s,
                            m["D2"], is_call) if np.isfinite(s) else np.nan)

        m["p_nc"] = m["settle"]
        m["actual"] = m["settle_t2"]
        m["t"] = t
        recs.append(m[["t", "expiry", "strike", "opt_type", "k", "T2", "vega",
                       "actual", "p_nc", "p_ss", "p_sm", "p_shift"]])
    if not recs:
        sys.exit("no evaluable contract-days — check surfaces coverage")
    return pd.concat(recs, ignore_index=True)


# ----------------------------------------------------------------- reports --

MODELS = {"no_change": "p_nc", "sticky_strike": "p_ss",
          "sticky_money": "p_sm", "sm_shifted": "p_shift"}


def report(res: pd.DataFrame, symbol: str, horizon: int) -> None:
    print(f"\n== {symbol} price backtest  h={horizon}d  "
          f"{res['t'].min().date()} -> {res['t'].max().date()}  "
          f"contract-days: {len(res)} ==")
    e_nc = (res["p_nc"] - res["actual"]).abs()
    rows = []
    for name, col in MODELS.items():
        ok = res[col].notna()
        e = (res.loc[ok, col] - res.loc[ok, "actual"])
        vega_err = (e.abs() / res.loc[ok, "vega"].clip(lower=1)).median()
        rows.append(dict(
            model=name, n=int(ok.sum()),
            rmse_rs=float(np.sqrt((e ** 2).mean())),
            mae_rs=float(e.abs().mean()),
            med_iv_err=float(vega_err),
            win_vs_nc=float((e.abs() < e_nc[ok]).mean()) if name != "no_change" else np.nan))
    print(pd.DataFrame(rows).set_index("model")
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("(med_iv_err = median |price error|/vega, i.e. implied-vol error "
          "in vol points; win_vs_nc = share of contracts beating no_change)")

    # bucketed view for the best structural model vs naive
    res = res.copy()
    res["tenor"] = pd.cut(res["T2"] * 365, [0, 10, 40, 400],
                          labels=["<=10d", "11-40d", ">40d"])
    res["money"] = pd.cut(res["k"].abs(), [0, 0.03, 0.10, 1],
                          labels=["ATM", "near", "wing"])
    for col, name in [("p_sm", "sticky_money")]:
        g = (res.assign(ae=lambda d: (d[col] - d["actual"]).abs(),
                        ae_nc=lambda d: (d["p_nc"] - d["actual"]).abs())
             .groupby(["tenor", "money"], observed=True)
             .apply(lambda d: pd.Series({
                 "mae_rs": d["ae"].mean(), "mae_nc_rs": d["ae_nc"].mean(),
                 "improve": 1 - d["ae"].mean() / d["ae_nc"].mean(),
                 "n": len(d)}), include_groups=False))
        print(f"\n-- {name} vs no_change by bucket --")
        print(g.to_string(float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()
    res = run(args.symbol, args.horizon)
    report(res, args.symbol, args.horizon)
    out = DATA_ROOT / "parquet" / "backtests"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{args.symbol}_prices_h{args.horizon}.parquet"
    res.to_parquet(f, index=False)
    print(f"\ndetail saved to {f}")


if __name__ == "__main__":
    main()
