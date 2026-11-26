"""Fetch public Energy-Charts data (Fraunhofer ISE, CC BY 4.0) for Germany / DE-LU.

Raw JSON responses are cached under data/raw/ (git-ignored). The processed hourly
table is written to data/processed/hourly_de.csv.gz by scripts/build_dataset.py.

Usage: python scripts/fetch_data.py [--start 2023-11-01] [--end 2026-11-01]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.energy-charts.info"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

ENDPOINTS = {
    "public_power": "public_power?country=de&start={s}&end={e}",
    "price": "price?bzn=DE-LU&start={s}&end={e}",
    "co2eq": "co2eq?country=de&start={s}&end={e}",
    "fc_load": "public_power_forecast?country=de&production_type=load&forecast_type=day-ahead&start={s}&end={e}",
    "fc_solar": "public_power_forecast?country=de&production_type=solar&forecast_type=day-ahead&start={s}&end={e}",
    "fc_wind_onshore": "public_power_forecast?country=de&production_type=wind_onshore&forecast_type=day-ahead&start={s}&end={e}",
    "fc_wind_offshore": "public_power_forecast?country=de&production_type=wind_offshore&forecast_type=day-ahead&start={s}&end={e}",
}


def fetch(name: str, start: str, end: str, sleep: float = 0.4) -> dict:
    path = RAW / f"{name}_{start}_{end}.json"
    if path.exists():
        return json.loads(path.read_text())
    url = f"{API}/" + ENDPOINTS[name].format(s=start, e=end)
    for attempt in range(5):
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            path.write_text(r.text)
            time.sleep(sleep)
            return r.json()
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{url} -> HTTP {r.status_code}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-11-01")
    ap.add_argument("--end", default="2026-11-01")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    months = pd.date_range(a.start, a.end, freq="MS")
    for m0, m1 in zip(months[:-1], months[1:]):
        # Energy-Charts 'end' is inclusive of the whole end day -> stop one day early
        s, e = m0.strftime("%Y-%m-%d"), (m1 - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for name in ENDPOINTS:
            fetch(name, s, e)
        print("fetched", s, e, flush=True)


if __name__ == "__main__":
    main()
