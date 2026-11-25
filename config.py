"""Shared configuration for the NSE F&O collector."""

import os
from pathlib import Path


DATA_ROOT = Path(os.environ.get("NSE_DATA_ROOT", Path.home() / "nse-data"))

RAW_BHAVCOPY_DIR = DATA_ROOT / "raw" / "bhavcopy"
RAW_CHAIN_DIR = DATA_ROOT / "raw" / "chain_snapshots"
PARQUET_EOD_DIR = DATA_ROOT / "parquet" / "fo_eod"
LOG_DIR = DATA_ROOT / "logs"


BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"
)

NSE_HOME = "https://www.nseindia.com"
CHAIN_URL_INDICES = NSE_HOME + "/api/option-chain-indices?symbol={symbol}"


CHAIN_SYMBOLS = ["NIFTY", "BANKNIFTY"]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

REQUEST_TIMEOUT = 30
BACKFILL_SLEEP = 2.0


def ensure_dirs() -> None:
    for p in (RAW_BHAVCOPY_DIR, RAW_CHAIN_DIR, PARQUET_EOD_DIR, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)
