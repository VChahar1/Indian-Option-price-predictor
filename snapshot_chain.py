"Snapshot NSE index option chains during market hours."

from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
import time
from zoneinfo import ZoneInfo

import requests

from config import (
    CHAIN_SYMBOLS,
    CHAIN_URL_INDICES,
    HEADERS,
    NSE_HOME,
    RAW_CHAIN_DIR,
    REQUEST_TIMEOUT,
    ensure_dirs,
)

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)


def market_is_open(now: dt.datetime) -> bool:
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def warmed_session() -> requests.Session:
    """NSE's API needs cookies set by the main site first."""
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(NSE_HOME, timeout=REQUEST_TIMEOUT)
    time.sleep(0.5)
    return s


def fetch_chain(session: requests.Session, symbol: str) -> dict | None:
    url = CHAIN_URL_INDICES.format(symbol=symbol)
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    if r.ok:
        try:
            return r.json()
        except json.JSONDecodeError:
            pass
    print(
        f"[chain] {symbol}: HTTP {r.status_code}, len {len(r.content)}", file=sys.stderr
    )
    return None


def main() -> None:
    now = dt.datetime.now(IST)
    if not market_is_open(now):
        return

    ensure_dirs()
    day_dir = RAW_CHAIN_DIR / now.date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%H%M%S")

    session = warmed_session()
    for symbol in CHAIN_SYMBOLS:
        data = fetch_chain(session, symbol)
        if data is None:
            session = warmed_session()
            data = fetch_chain(session, symbol)
        if data is None:
            print(f"[chain] {symbol} {stamp}: failed, skipping", file=sys.stderr)
            continue
        dest = day_dir / f"{symbol}_{stamp}.json.gz"
        with gzip.open(dest, "wt", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"[chain] {symbol} {stamp}: saved {dest.name}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
