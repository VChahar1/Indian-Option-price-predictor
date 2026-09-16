# nse-collector

Free-data collection layer for the personal Indian F&O prediction bot.
Pulls the daily NSE F&O bhavcopy (UDiFF format), normalizes it to parquet,
and snapshots the NIFTY/BANKNIFTY option chains intraday.

## Layout

```
config.py            paths, URLs, headers (override root via NSE_DATA_ROOT)
fetch_bhavcopy.py    daily EOD download + backfill mode
normalize.py         UDiFF zip -> long-format parquet (one file per day)
snapshot_chain.py    intraday option-chain JSON snapshots (market-hours guarded)
systemd/             user units: 2 services + 2 timers
```

Data lands under `~/nse-data/` by default:

```
raw/bhavcopy/2026/BhavCopy_..._20260717_F_0000.csv.zip   # originals, keep forever
raw/chain_snapshots/2026-07-17/NIFTY_093215.json.gz
parquet/fo_eod/year=2026/2026-07-17.parquet
```

## Install

```bash
mkdir -p ~/nse-collector && cp -r <extracted files> ~/nse-collector/
pip install --user -r ~/nse-collector/requirements.txt

# systemd user units
mkdir -p ~/.config/systemd/user
cp ~/nse-collector/systemd/* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now nse-bhavcopy.timer nse-chain.timer

# let user timers run without an active login session
loginctl enable-linger $USER
```

Check status / logs:

```bash
systemctl --user list-timers
journalctl --user -u nse-bhavcopy.service -e
journalctl --user -u nse-chain.service -e
```

## Backfill

The UDiFF format starts 2024-07-08; earlier files use the old naming scheme.

```bash
python3 fetch_bhavcopy.py --backfill 2024-07-08 $(date +%F)
```

Safe to interrupt and re-run: already-downloaded days are skipped, 404s
(holidays/weekends) are ignored, and there is a 2 s politeness delay between
requests.

## Querying

```python
import duckdb
con = duckdb.connect()
con.sql("""
    SELECT trade_date, expiry, strike, opt_type, settle, oi
    FROM read_parquet('~/nse-data/parquet/fo_eod/*/*.parquet')
    WHERE symbol = 'NIFTY' AND instrument = 'IDO'
      AND volume > 0
    ORDER BY trade_date, expiry, strike
""").df()
```

## Notes and known sharp edges

- **Chain snapshots are best-effort.** NSE changes its anti-bot behavior
  without notice. If snapshots start failing (check journalctl), the usual
  fixes are a newer User-Agent string in `config.py` or a longer warmup.
  The EOD bhavcopy path is independent and much more stable.
- **Timers are DST-proof by design**: the chain timer fires over a wide
  local-time window and `snapshot_chain.py` itself checks 09:15–15:30 IST,
  exiting instantly when closed.
- **The bhavcopy timer fires at 16:00 and 20:00** local time; publication
  time varies, and the fetch is idempotent so the second run is a free retry.
- Raw files are never modified. If the schema changes, delete
  `parquet/` and re-run `normalize.py` over the raw zips.
- If NSE tweaks UDiFF column names, `normalize.py` warns about missing
  columns instead of crashing; check stderr in journalctl occasionally.
