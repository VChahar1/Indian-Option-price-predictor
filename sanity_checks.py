"Sanity checks over the fitted vol surfaces."

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_surfaces import SURF_DIR, svi_w
from config import DATA_ROOT

REPORT_DIR = DATA_ROOT / "reports"
FRONT_MIN_DAYS = 5


def load(name: str, symbol: str) -> pd.DataFrame:
    files = sorted(SURF_DIR.glob(f"{name}/year=*/*_{symbol}.parquet"))
    if not files:
        sys.exit(f"no {name} files for {symbol} — run build_surfaces.py first")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def front_series(params: pd.DataFrame) -> pd.DataFrame:
    """Per day: the nearest expiry with T*365 >= FRONT_MIN_DAYS."""
    ok = params[params["T"] * 365 >= FRONT_MIN_DAYS]
    idx = ok.groupby("trade_date")["T"].idxmin()
    return ok.loc[idx].sort_values("trade_date")


def month_series(params: pd.DataFrame) -> pd.DataFrame:
    """Per day: expiry closest to 30 calendar days out."""
    p = params.copy()
    p["dist"] = (p["T"] * 365 - 30).abs()
    idx = p.groupby("trade_date")["dist"].idxmin()
    return p.loc[idx].sort_values("trade_date")


def try_fetch_vix(start, end) -> pd.Series | None:
    try:
        import yfinance as yf

        vix = yf.download("^INDIAVIX", start=start, end=end, progress=False)
        if vix is None or vix.empty:
            return None
        close = vix["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close / 100.0
    except Exception as e:
        print(f"[sanity] India VIX unavailable ({e}); skipping VIX overlay")
        return None


def main() -> None:
    """Check local volatility surfaces and their available reference series."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY")
    args = ap.parse_args()
    sym = args.symbol
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    params = load("svi_params", sym)
    iv_pts = load("iv_points", sym)
    params["trade_date"] = pd.to_datetime(params["trade_date"])
    iv_pts["trade_date"] = pd.to_datetime(iv_pts["trade_date"])

    days = params["trade_date"].dt.date.nunique()
    span = (params["trade_date"].max() - params["trade_date"].min()).days
    bdays = int(
        np.busday_count(
            params["trade_date"].min().date(), params["trade_date"].max().date()
        )
    )
    n_bad_fly = int((~params["butterfly_ok"]).sum())
    n_bad_cal = int(
        (~params.get("calendar_ok", pd.Series(True, index=params.index))).sum()
    )

    print(f"== {sym} surface diagnostics ==")
    print(
        f"days with fits           : {days} "
        f"(~{bdays} business days in span of {span} calendar days; "
        f"gap = holidays + filtered days)"
    )
    print(f"expiry fits total        : {len(params)}")
    print(f"median fit RMSE (totvar) : {params['fit_rmse_iv_totvar'].median():.2e}")
    print(
        f"worst 1% fit RMSE        : {params['fit_rmse_iv_totvar'].quantile(0.99):.2e}"
    )
    print(f"butterfly flags          : {n_bad_fly} ({n_bad_fly / len(params):.1%})")
    print(f"calendar flags           : {n_bad_cal} ({n_bad_cal / len(params):.1%})")

    fr = front_series(params)
    mo = month_series(params)

    fm = fr.merge(mo, on="trade_date", suffixes=("_f", "_m"))
    fm = fm[fm["T_m"] > fm["T_f"] + 3 / 365]
    if len(fm) > 20:
        carry = (np.log(fm["F_m"] / fm["F_f"]) / (fm["T_m"] - fm["T_f"])).median()
        print(f"median implied carry     : {carry:.2%}  (sanity: ~5-8% for NIFTY)")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(fr["trade_date"], fr["atm_iv"], lw=0.9, label="front ATM IV")
    ax.plot(mo["trade_date"], mo["atm_iv"], lw=0.9, label="~1M ATM IV")
    vix = try_fetch_vix(params["trade_date"].min(), params["trade_date"].max())
    if vix is not None:
        ax.plot(vix.index, vix.values, lw=0.9, ls="--", label="India VIX")
    ax.set_title(f"{sym} ATM implied vol history")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"atm_iv_history_{sym}.png", dpi=140)

    if vix is not None:
        joined = fr.set_index(fr["trade_date"].dt.normalize())["atm_iv"].to_frame()
        joined["vix"] = vix.reindex(joined.index)
        joined = joined.dropna()
        if len(joined) > 20:
            corr = joined["atm_iv"].corr(joined["vix"])
            print(f"front ATM IV vs VIX corr : {corr:.3f}  (expect > 0.9)")
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.scatter(joined["vix"], joined["atm_iv"], s=6, alpha=0.5)
            lim = [joined.min().min() * 0.9, joined.max().max() * 1.05]
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set_xlabel("India VIX")
            ax.set_ylabel("front ATM IV")
            ax.set_title(f"{sym}: ATM IV vs VIX (corr {corr:.2f})")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(REPORT_DIR / f"atm_iv_vs_vix_{sym}.png", dpi=140)

    last = params["trade_date"].max()
    day_p = params[params["trade_date"] == last].sort_values("T").head(4)
    day_i = iv_pts[iv_pts["trade_date"] == last]
    fig, ax = plt.subplots(figsize=(9, 5))
    for _, r in day_p.iterrows():
        pts = day_i[day_i["expiry"] == r["expiry"]]
        p = np.array([r["a"], r["b"], r["rho"], r["m"], r["sigma_svi"]])
        kk = np.linspace(pts["k"].min(), pts["k"].max(), 200)
        lbl = f"{pd.Timestamp(r['expiry']).date()} (T={r['T'] * 365:.0f}d)"
        (line,) = ax.plot(kk, np.sqrt(svi_w(kk, p) / r["T"]), lw=1.2, label=lbl)
        ax.scatter(pts["k"], pts["iv"], s=10, color=line.get_color(), alpha=0.6)
    ax.set_xlabel("log-moneyness ln(K/F)")
    ax.set_ylabel("implied vol")
    ax.set_title(f"{sym} smiles, {last.date()} — SVI fits vs data")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"smile_latest_{sym}.png", dpi=140)

    day_all = params[params["trade_date"] == last].sort_values("T")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(day_all["T"] * 365, day_all["atm_iv"], "o-", lw=1)
    ax.set_xlabel("days to expiry")
    ax.set_ylabel("ATM IV")
    ax.set_title(f"{sym} ATM term structure, {last.date()}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"term_structure_{sym}.png", dpi=140)

    fig, ax = plt.subplots(figsize=(11, 4))
    daily_rmse = params.groupby("trade_date")["fit_rmse_iv_totvar"].median()
    ax.plot(daily_rmse.index, daily_rmse.values, lw=0.8)
    bad = params[~params["butterfly_ok"]]
    if not bad.empty:
        ax.scatter(
            bad["trade_date"],
            bad["fit_rmse_iv_totvar"],
            s=12,
            color="crimson",
            label="butterfly flag",
            zorder=3,
        )
        ax.legend()
    ax.set_yscale("log")
    ax.set_title(f"{sym} daily median SVI fit RMSE (log scale)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"fit_quality_{sym}.png", dpi=140)

    print(f"\nplots written to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
