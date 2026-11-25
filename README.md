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
```

Data lands under `~/nse-data/` by default:

```
raw/bhavcopy/2026/BhavCopy_..._20260717_F_0000.csv.zip   # originals, keep forever
raw/chain_snapshots/2026-07-17/NIFTY_093215.json.gz
parquet/fo_eod/year=2026/2026-07-17.parquet
```

## Install

```bash
git clone https://github.com/VChahar1/Indian-Option-price-predictor.git
cd Indian-Option-price-predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_bhavcopy.py --help
```

This snapshot does not include `systemd/` service or timer files. Run the collectors manually or configure your own scheduler; copying units from that missing directory will not work.

The pricing and surface scripts use the scientific Python dependencies in `requirements.txt`. The neural forecast script additionally needs a suitable PyTorch installation (`python -m pip install torch`); the optional VIX overlay uses `yfinance` when available.

No automated test suite is included. The file `sanity_checks.py` is a data-dependent surface diagnostic, rather than a unit-test runner: build local surface data before running it. The setup cleanup checked command help, Black-76 pricing parity and a price-to-implied-volatility round trip; it did not validate forecast accuracy or download the market-data history.

## Backfill

The downloader uses the UDiFF filename template in `config.py`; older archive formats may require a different template.

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
  without notice. If snapshots start failing (check the collector output), the usual
  fixes are a newer User-Agent string in `config.py` or a longer warmup.
  The EOD bhavcopy path is independent and much more stable.
- `snapshot_chain.py` checks weekdays and the configured 09:15–15:30 IST time window. It does not maintain an exchange-holiday calendar. Scheduling is left to the caller; timer units are not included.
- Raw files are never modified. If the schema changes, delete
  `parquet/` and re-run `normalize.py` over the raw zips.
- If NSE tweaks UDiFF column names, `normalize.py` warns about missing
  columns instead of crashing; check stderr in journalctl occasionally.
