"""Normalize a raw UDiFF F&O bhavcopy zip into a long-format parquet file.

Output schema (one row per contract per day):
    trade_date, symbol, instrument (IDF/IDO/STF/STO), expiry, strike, opt_type,
    open, high, low, close, last, settle, prev_close, underlying_close,
    volume, oi, oi_change, turnover, trades

Idempotent: re-running for the same date overwrites the same parquet file.
Usage:
    python3 normalize.py /path/to/BhavCopy_..._20260717_F_0000.csv.zip
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from config import PARQUET_EOD_DIR, ensure_dirs

# UDiFF (ISO 20022-style) column names -> our schema. Names occasionally get
# tweaked; anything missing is filled with NA rather than crashing.
COLUMN_MAP = {
    "TradDt": "trade_date",
    "TckrSymb": "symbol",
    "FinInstrmTp": "instrument",   # IDF/IDO = index fut/opt, STF/STO = stock
    "XpryDt": "expiry",
    "StrkPric": "strike",
    "OptnTp": "opt_type",          # CE / PE, empty for futures
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "LastPric": "last",
    "SttlmPric": "settle",
    "PrvsClsgPric": "prev_close",
    "UndrlygPric": "underlying_close",
    "TtlTradgVol": "volume",
    "OpnIntrst": "oi",
    "ChngInOpnIntrst": "oi_change",
    "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "trades",
    "NewBrdLotQty": "lot_size",
}

NUMERIC_COLS = [
    "strike", "open", "high", "low", "close", "last", "settle", "prev_close",
    "underlying_close", "volume", "oi", "oi_change", "turnover", "trades",
    "lot_size",
]


def normalize_zip(zip_path: Path) -> Path:
    ensure_dirs()
    df = pd.read_csv(zip_path, compression="zip", low_memory=False)

    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        print(f"[normalize] warning: columns absent in source: {missing}",
              file=sys.stderr)

    out = pd.DataFrame(
        {new: df[old] if old in df.columns else pd.NA
         for old, new in COLUMN_MAP.items()}
    )

    # Keep only F&O rows (defensive; the FO file should contain nothing else).
    out = out[out["instrument"].isin(["IDF", "IDO", "STF", "STO"])].copy()

    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # Futures rows: blank option type / zero strike -> proper NA.
    out.loc[out["instrument"].isin(["IDF", "STF"]), ["opt_type", "strike"]] = pd.NA

    dates = out["trade_date"].dropna().dt.date.unique()
    if len(dates) != 1:
        raise ValueError(f"{zip_path.name}: expected one trade date, got {dates}")
    d = dates[0]

    dest_dir = PARQUET_EOD_DIR / f"year={d.year}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{d.isoformat()}.parquet"
    out.to_parquet(dest, index=False)
    print(f"[normalize] {zip_path.name} -> {dest} ({len(out)} rows)")
    return dest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    normalize_zip(Path(sys.argv[1]))
