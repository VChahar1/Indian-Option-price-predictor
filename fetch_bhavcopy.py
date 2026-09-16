"""Download the NSE F&O UDiFF bhavcopy and normalize it to parquet.

Usage:
    python3 fetch_bhavcopy.py                       # today's file (IST)
    python3 fetch_bhavcopy.py --date 2026-07-17     # a specific date
    python3 fetch_bhavcopy.py --backfill 2024-07-08 2026-07-17

404 responses are treated as "no trading that day / not yet published" and
skipped quietly, so it's safe to run daily and over weekends/holidays.
Already-downloaded days are skipped (idempotent), so a backfill can be
resumed after an interruption by re-running the same command.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from zoneinfo import ZoneInfo

import requests

from config import (BACKFILL_SLEEP, BHAVCOPY_URL, HEADERS, RAW_BHAVCOPY_DIR,
                    REQUEST_TIMEOUT, ensure_dirs)
from normalize import normalize_zip

IST = ZoneInfo("Asia/Kolkata")


def fetch_day(session: requests.Session, day: dt.date) -> bool:
    """Fetch one day's bhavcopy. Returns True if a file was obtained."""
    if day.weekday() >= 5:  # Sat/Sun
        return False
    dest_dir = RAW_BHAVCOPY_DIR / str(day.year)
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = BHAVCOPY_URL.format(d=day.strftime("%Y%m%d"))
    dest = dest_dir / url.rsplit("/", 1)[-1]

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[fetch] {day} already present, skipping download")
        normalize_if_needed(dest, day)
        return True

    for attempt in (1, 2, 3):
        try:
            r = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            print(f"[fetch] {day} attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(5 * attempt)
            continue
        if r.status_code == 404:
            print(f"[fetch] {day}: 404 (holiday or not yet published)")
            return False
        if r.ok and r.content[:2] == b"PK":  # sanity: it's really a zip
            dest.write_bytes(r.content)
            print(f"[fetch] {day}: saved {dest.name} ({len(r.content)//1024} KB)")
            normalize_zip(dest)
            return True
        print(f"[fetch] {day} attempt {attempt}: HTTP {r.status_code}",
              file=sys.stderr)
        time.sleep(5 * attempt)
    print(f"[fetch] {day}: giving up after 3 attempts", file=sys.stderr)
    return False


def normalize_if_needed(zip_path, day: dt.date) -> None:
    from config import PARQUET_EOD_DIR
    pq = PARQUET_EOD_DIR / f"year={day.year}" / f"{day.isoformat()}.parquet"
    if not pq.exists():
        normalize_zip(zip_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: today in IST)")
    ap.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                    help="inclusive YYYY-MM-DD range, fetched oldest-first")
    args = ap.parse_args()

    ensure_dirs()
    session = requests.Session()

    if args.backfill:
        start = dt.date.fromisoformat(args.backfill[0])
        end = dt.date.fromisoformat(args.backfill[1])
        day, got = start, 0
        while day <= end:
            if fetch_day(session, day):
                got += 1
            time.sleep(BACKFILL_SLEEP)
            day += dt.timedelta(days=1)
        print(f"[fetch] backfill done: {got} trading days collected")
    else:
        day = (dt.date.fromisoformat(args.date) if args.date
               else dt.datetime.now(IST).date())
        fetch_day(session, day)


if __name__ == "__main__":
    main()
