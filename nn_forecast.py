"""Neural-net forecaster, evaluated head-to-head against the baselines.

Same walk-forward protocol as forecast.py (expanding window, periodic refit,
strict leakage control), same targets, same sample — so the comparison table
at the end is a fair fight. Adds surface-derived features the linear
baselines don't see:

  rho_front  : front-expiry SVI skew parameter
  term_slope : ATM total-variance slope between ~1M and the front expiry
  vrp        : iv30 minus trailing horizon RV (variance-risk-premium proxy)
  ret_5      : trailing 5-day return (leverage effect)

Model: small MLP (1 hidden layer) predicting log(target/persistence-anchor)
— a stationary residual target, so predicting zero recovers persistence and
regime shifts cannot cause unbounded level extrapolation. Full-batch Adam,
chronological 85/15 early-stopping split, 3-seed ensemble averaged.
Features standardized with training-window statistics only.

Usage:
  python3 nn_forecast.py --symbol BANKNIFTY --horizon 5
  python3 nn_forecast.py --symbol NIFTY --horizon 5 --target rv_fwd
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from build_surfaces import SURF_DIR
from config import DATA_ROOT
from forecast import (build_panel, dm_test, features_for, ols, predict,
                      qlike, walk_forward)

FRONT_MIN_DAYS = 5


# ---------------------------------------------------------------- features --

def load_surface_features(symbol: str) -> pd.DataFrame:
    files = sorted(SURF_DIR.glob(f"svi_params/year=*/*_{symbol}.parquet"))
    p = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    p = p[p["butterfly_ok"] & p.get("calendar_ok", True)]
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    rows = {}
    for d, g in p.groupby("trade_date"):
        g = g[g["T"] * 365 >= FRONT_MIN_DAYS].sort_values("T")
        if g.empty:
            continue
        front = g.iloc[0]
        feat = {"rho_front": front["rho"]}
        # slope of ATM total variance per year between front and ~1M point
        cand = g[g["T"] > front["T"] + 3 / 365]
        if not cand.empty:
            nxt = cand.iloc[(cand["T"] * 365 - 30).abs().argmin()]
            w1 = front["atm_iv"] ** 2 * front["T"]
            w2 = nxt["atm_iv"] ** 2 * nxt["T"]
            feat["term_slope"] = (w2 - w1) / (nxt["T"] - front["T"])
        rows[d] = feat
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def build_nn_panel(symbol: str, horizon: int) -> pd.DataFrame:
    panel = build_panel(symbol, horizon)
    surf = load_surface_features(symbol)
    panel = panel.join(surf)
    panel["vrp"] = panel["iv30"] - panel["rv_h"]
    panel["ret_5"] = panel["ret"].rolling(5).sum()
    panel["term_slope"] = panel["term_slope"].ffill(limit=5)
    panel["rho_front"] = panel["rho_front"].ffill(limit=5)
    return panel


NN_FEATURES = ["rv_d", "rv_w", "rv_m", "rv_h", "iv30",
               "rho_front", "term_slope", "vrp", "ret_5"]
LOG_FEATURES = {"rv_d", "rv_w", "rv_m", "rv_h", "iv30"}


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for c in NN_FEATURES:
        x = df[c].to_numpy(float)
        if c in LOG_FEATURES:
            x = np.log(np.clip(x, 1e-4, None))
        cols.append(x)
    return np.column_stack(cols)


# ------------------------------------------------------------------- model --

class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(X: np.ndarray, y: np.ndarray, seed: int,
              max_epochs: int = 500, patience: int = 30) -> tuple[MLP, np.ndarray, np.ndarray]:
    """Returns (model, mu, sd) with feature standardization from train stats."""
    torch.manual_seed(seed)
    n_val = max(int(0.15 * len(X)), 10)
    Xtr, ytr = X[:-n_val], y[:-n_val]
    Xva, yva = X[-n_val:], y[-n_val:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xtr_t, Xva_t = to_t((Xtr - mu) / sd), to_t((Xva - mu) / sd)
    ytr_t, yva_t = to_t(ytr), to_t(yva)

    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t), yva_t).item()
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, mu, sd


class NNEnsemble:
    def __init__(self, n_members: int = 3):
        self.n_members = n_members
        self.members: list[tuple[MLP, np.ndarray, np.ndarray]] = []

    def fit(self, X: np.ndarray, y_log: np.ndarray) -> None:
        self.members = [train_one(X, y_log, seed) for seed in range(self.n_members)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for model, mu, sd in self.members:
                xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
                preds.append(model(xt).numpy())
        # bounded log-ratio correction to the persistence anchor
        return np.clip(np.mean(preds, axis=0), -1.5, 1.5)


# ------------------------------------------------------------ walk-forward --

def walk_forward_nn(panel: pd.DataFrame, target: str, horizon: int,
                    min_train: int, refit_every: int = 21) -> pd.Series:
    df = panel.dropna(subset=[target] + NN_FEATURES)
    n = len(df)
    if n < min_train + 30:
        sys.exit(f"only {n} usable rows after feature dropna; "
                 f"lower --min-train or check surface feature coverage")
    anchor_col = "rv_h" if target == "rv_fwd" else "iv30"
    preds = np.full(n, np.nan)
    ens = NNEnsemble()
    for i in range(min_train, n):
        if (i - min_train) % refit_every == 0:
            tr = df.iloc[: i - horizon]
            X = design_matrix(tr)
            ratio = (np.clip(tr[target].to_numpy(float), 1e-4, None)
                     / np.clip(tr[anchor_col].to_numpy(float), 1e-4, None))
            y_log = np.log(ratio)
            ens.fit(X, y_log)
            print(f"[nn] refit at {df.index[i].date()} on {len(tr)} rows",
                  file=sys.stderr)
        row = df.iloc[[i]]
        anchor = float(np.clip(row[anchor_col].iloc[0], 1e-4, None))
        preds[i] = anchor * np.exp(ens.predict(design_matrix(row))[0])
    return pd.Series(preds, index=df.index, name="nn").iloc[min_train:]


# --------------------------------------------------------------------- cli --

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=250)
    ap.add_argument("--target", default="rv_fwd", choices=["rv_fwd", "iv30_fwd"])
    args = ap.parse_args()

    panel = build_nn_panel(args.symbol, args.horizon)
    print(f"[nn] panel: {len(panel)} days, "
          f"surface features present on "
          f"{panel['rho_front'].notna().mean():.0%} of days")

    base = walk_forward(panel, args.target, args.horizon, args.min_train)
    nn_pred = walk_forward_nn(panel, args.target, args.horizon, args.min_train)

    res = base.join(nn_pred, how="inner").dropna(subset=["nn"])
    a = res["actual"].to_numpy()
    e_pers = res["persistence"].to_numpy() - a
    e_harx = res["har_x"].to_numpy() - a

    rows = []
    for m in ["persistence", "ar1", "har", "har_x", "nn"]:
        e = res[m].to_numpy() - a
        row = dict(model=m, rmse=float(np.sqrt(np.nanmean(e ** 2))),
                   mae=float(np.nanmean(np.abs(e))),
                   qlike=qlike(a, res[m].to_numpy()))
        if m != "persistence":
            row["dm_vs_pers"], row["p_pers"] = dm_test(e, e_pers, args.horizon)
        if m == "nn":
            row["dm_vs_harx"], row["p_harx"] = dm_test(e, e_harx, args.horizon)
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("model")
    print(f"\n== {args.symbol}  target: {args.target}  h: {args.horizon}d  "
          f"oos: {len(res)}  ({res['date'].min().date()} -> "
          f"{res['date'].max().date()}) ==")
    print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))
    print("(negative dm with small p => row model beats column benchmark)")

    out_dir = DATA_ROOT / "parquet" / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_dir / f"{args.symbol}_{args.target}_h{args.horizon}_nn.parquet",
                   index=False)


if __name__ == "__main__":
    main()