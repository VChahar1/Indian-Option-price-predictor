"""Forecasting baselines with a walk-forward evaluation harness.

Builds a daily feature panel from the fitted surfaces + EOD futures data,
then evaluates honest baseline models for two targets:

  RV   : realized vol of the underlying over the next h trading days
  IV30 : 30d constant-maturity ATM implied vol, h days ahead

Models (fit by OLS, refit periodically, expanding window, no leakage):
  persistence : x_{t+h} = x_t                     (the one to beat)
  ar1         : AR(1) on the target
  har         : HAR (daily/weekly/monthly RV components)
  har_x       : HAR + current IV30 level as extra predictor

Metrics: RMSE, MAE, QLIKE (variance loss), and a Diebold-Mariano test of
each model against persistence (HAC-adjusted, lag h-1). Nothing here is a
trading signal by itself — it's the benchmark your NN has to beat.

Usage:
  python3 forecast.py --symbol NIFTY --horizon 5
  python3 forecast.py --symbol BANKNIFTY --horizon 21 --min-train 250
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from build_surfaces import SURF_DIR
from config import DATA_ROOT, PARQUET_EOD_DIR

TRADING_DAYS = 252
CM_DAYS = 30  # constant-maturity tenor for the IV series


# ---------------------------------------------------------------- features --

def load_underlying(symbol: str) -> pd.Series:
    files = sorted(PARQUET_EOD_DIR.glob("year=*/*.parquet"))
    closes = []
    for f in files:
        df = pd.read_parquet(f, columns=["trade_date", "symbol", "instrument",
                                         "underlying_close"])
        row = df[(df["symbol"] == symbol) & df["underlying_close"].notna()]
        if not row.empty:
            closes.append((row["trade_date"].iloc[0],
                           float(row["underlying_close"].iloc[0])))
    s = (pd.Series(dict(closes)).sort_index())
    s.index = pd.to_datetime(s.index)
    return s.rename("close")


def load_cm_iv(symbol: str) -> pd.Series:
    """30d constant-maturity ATM IV: linear interpolation of total variance
    in T across clean (unflagged) expiries."""
    files = sorted(SURF_DIR.glob(f"svi_params/year=*/*_{symbol}.parquet"))
    if not files:
        sys.exit(f"no svi_params for {symbol} — run build_surfaces.py first")
    p = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    p = p[p["butterfly_ok"] & p.get("calendar_ok", True)]
    T0 = CM_DAYS / 365.0
    out = {}
    for d, g in p.groupby("trade_date"):
        g = g.sort_values("T")
        T, w = g["T"].to_numpy(), (g["atm_iv"] ** 2 * g["T"]).to_numpy()
        if len(g) == 0 or T.min() > 3 * T0:
            continue
        if len(g) == 1 or T0 <= T.min():
            iv = g["atm_iv"].iloc[0]
        elif T0 >= T.max():
            iv = g["atm_iv"].iloc[-1]
        else:
            w0 = np.interp(T0, T, w)
            iv = np.sqrt(w0 / T0)
        out[pd.Timestamp(d)] = float(iv)
    return pd.Series(out).sort_index().rename("iv30")


def build_panel(symbol: str, horizon: int) -> pd.DataFrame:
    close = load_underlying(symbol)
    r = np.log(close).diff()
    rv_1 = np.sqrt(TRADING_DAYS * r ** 2)                       # daily proxy
    rv_d = rv_1
    rv_w = np.sqrt(TRADING_DAYS * (r ** 2).rolling(5).mean())
    rv_m = np.sqrt(TRADING_DAYS * (r ** 2).rolling(22).mean())
    # trailing horizon-matched RV: the honest random-walk benchmark
    rv_h = np.sqrt(TRADING_DAYS * (r ** 2).rolling(horizon).mean())
    # forward realized vol over the NEXT horizon days (the RV target)
    fwd_var = (r ** 2).shift(-1).rolling(horizon).mean().shift(-(horizon - 1))
    rv_fwd = np.sqrt(TRADING_DAYS * fwd_var)

    iv30 = load_cm_iv(symbol)
    panel = pd.concat({"ret": r, "rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m,
                       "rv_h": rv_h, "rv_fwd": rv_fwd, "iv30": iv30}, axis=1)
    panel["iv30_fwd"] = panel["iv30"].shift(-horizon)           # the IV target
    return panel.dropna(subset=["rv_d", "rv_w", "rv_m", "rv_h", "iv30"])


# ------------------------------------------------------------------ models --

def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


MODEL_FEATURES = {
    "ar1": lambda df, tgt: df[[tgt.replace("_fwd", "") if tgt == "rv_fwd"
                               else "iv30"]].rename(columns=lambda c: "x")
                              .assign(x=lambda d: d["x"]),
    "har": lambda df, tgt: df[["rv_d", "rv_w", "rv_m"]],
    "har_x": lambda df, tgt: df[["rv_d", "rv_w", "rv_m", "iv30"]],
}


def features_for(model: str, df: pd.DataFrame, target: str) -> np.ndarray:
    if model == "ar1":
        base = "rv_h" if target == "rv_fwd" else "iv30"
        return df[[base]].to_numpy()
    return MODEL_FEATURES[model](df, target).to_numpy()


# ------------------------------------------------------------ walk-forward --

def walk_forward(panel: pd.DataFrame, target: str, horizon: int,
                 min_train: int, refit_every: int = 21) -> pd.DataFrame:
    df = panel.dropna(subset=[target])
    n = len(df)
    if n < min_train + 30:
        sys.exit(f"only {n} usable rows for target {target}; "
                 f"lower --min-train or collect more history")
    models = ["persistence", "ar1", "har", "har_x"]
    preds = {m: np.full(n, np.nan) for m in models}
    betas: dict[str, np.ndarray] = {}

    for i in range(min_train, n):
        if (i - min_train) % refit_every == 0:
            # training rows whose forward-looking target is fully realized by i
            tr = df.iloc[: i - horizon]
            y = tr[target].to_numpy()
            for m in models:
                if m == "persistence":
                    continue
                betas[m] = ols(features_for(m, tr, target), y)
        row = df.iloc[[i]]
        preds["persistence"][i] = (row["rv_h"] if target == "rv_fwd"
                                   else row["iv30"]).iloc[0]
        for m in models:
            if m != "persistence":
                preds[m][i] = predict(betas[m], features_for(m, row, target))[0]

    out = pd.DataFrame(preds, index=df.index)
    out["actual"] = df[target]
    out["date"] = df.index
    return out.iloc[min_train:]


# ----------------------------------------------------------------- metrics --

def qlike(actual: np.ndarray, pred: np.ndarray) -> float:
    a2, p2 = actual ** 2, np.clip(pred, 1e-4, None) ** 2
    return float(np.mean(a2 / p2 - np.log(a2 / p2) - 1))


def dm_test(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano on squared-error differentials, HAC lag h-1.
    Negative stat => model beats benchmark."""
    d = e_model ** 2 - e_bench ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
        var += 2 * (1 - lag / h) * cov
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def report(res: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    rows = []
    a = res["actual"].to_numpy()
    e_bench = res["persistence"].to_numpy() - a
    for m in ["persistence", "ar1", "har", "har_x"]:
        p = res[m].to_numpy()
        e = p - a
        row = dict(model=m,
                   rmse=float(np.sqrt(np.nanmean(e ** 2))),
                   mae=float(np.nanmean(np.abs(e))),
                   qlike=qlike(a, p))
        if m != "persistence":
            row["dm_stat"], row["dm_pval"] = dm_test(e, e_bench, horizon)
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("model")
    print(f"\n== target: {target}  horizon: {horizon}d  "
          f"oos obs: {len(res)}  ({res['date'].min().date()} -> "
          f"{res['date'].max().date()}) ==")
    print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))
    print("(dm_stat < 0 with dm_pval < 0.05 => significantly beats persistence)")
    return tbl


# --------------------------------------------------------------------- cli --

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=250)
    args = ap.parse_args()

    panel = build_panel(args.symbol, args.horizon)
    print(f"[forecast] panel: {len(panel)} days "
          f"({panel.index.min().date()} -> {panel.index.max().date()})")

    out_dir = DATA_ROOT / "parquet" / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in ["rv_fwd", "iv30_fwd"]:
        res = walk_forward(panel, target, args.horizon, args.min_train)
        report(res, target, args.horizon)
        res.to_parquet(out_dir / f"{args.symbol}_{target}_h{args.horizon}.parquet",
                       index=False)
    print(f"\nforecast paths saved to {out_dir}/")


if __name__ == "__main__":
    main()
