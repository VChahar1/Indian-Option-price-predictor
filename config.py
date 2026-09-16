"""Shared configuration for the NSE F&O collector."""
import os
from pathlib import Path

# Root of all collected data. Override with NSE_DATA_ROOT env var.
DATA_ROOT = Path(os.environ.get("NSE_DATA_ROOT", Path.home() / "nse-data"))

RAW_BHAVCOPY_DIR = DATA_ROOT / "raw" / "bhavcopy"
RAW_CHAIN_DIR = DATA_ROOT / "raw" / "chain_snapshots"
PARQUET_EOD_DIR = DATA_ROOT / "parquet" / "fo_eod"
LOG_DIR = DATA_ROOT / "logs"

# UDiFF F&O bhavcopy (format in force since 2024-07-08).
# {d} is the trade date as YYYYMMDD.
BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"
)

NSE_HOME = "https://www.nseindia.com"
CHAIN_URL_INDICES = NSE_HOME + "/api/option-chain-indices?symbol={symbol}"

# Index symbols to snapshot intraday. Add e.g. "FINNIFTY", "MIDCPNIFTY" if wanted.
CHAIN_SYMBOLS = ["NIFTY", "BANKNIFTY"]

# Browser-like headers; NSE rejects default python UAs.
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

REQUEST_TIMEOUT = 30  # seconds
BACKFILL_SLEEP = 2.0  # seconds between archive requests; be polite


def ensure_dirs() -> None:
    for p in (RAW_BHAVCOPY_DIR, RAW_CHAIN_DIR, PARQUET_EOD_DIR, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)
