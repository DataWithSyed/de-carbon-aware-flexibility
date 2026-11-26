"""Parse cached Energy-Charts JSON into one hourly UTC table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Energy-Charts public_power names -> column names used here
GEN_COLUMNS = {
    "Hydro Run-of-River": "hydro_ror",
    "Biomass": "biomass",
    "Fossil brown coal / lignite": "lignite",
    "Fossil hard coal": "hard_coal",
    "Fossil oil": "oil",
    "Fossil coal-derived gas": "coal_gas",
    "Fossil gas": "gas",
    "Geothermal": "geothermal",
    "Hydro water reservoir": "hydro_res",
    "Hydro pumped storage": "pumped_gen",
    "Others": "others",
    "Waste": "waste",
    "Wind offshore": "wind_off",
    "Wind onshore": "wind_on",
    "Solar": "solar",
    "Nuclear": "nuclear",
}
OTHER_COLUMNS = {
    "Hydro pumped storage consumption": "pumped_cons",
    "Cross border electricity trading": "net_import",
    "Load": "load",
    "Residual load": "residual_load",
}


def _series(unix: list, values: list, name: str) -> pd.Series:
    idx = pd.to_datetime(np.asarray(unix, dtype="int64"), unit="s", utc=True)
    s = pd.Series(np.asarray([np.nan if v is None else v for v in values], dtype=float), index=idx, name=name)
    return s[~s.index.duplicated()]


def parse_public_power(d: dict) -> pd.DataFrame:
    cols = {}
    for p in d["production_types"]:
        name = GEN_COLUMNS.get(p["name"]) or OTHER_COLUMNS.get(p["name"])
        if name:
            cols[name] = _series(d["unix_seconds"], p["data"], name)
    return pd.DataFrame(cols)


def parse_simple(d: dict, key: str, name: str) -> pd.Series:
    return _series(d["unix_seconds"], d[key], name)


def to_hourly(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Average sub-hourly MW (or EUR/MWh, g/kWh) values to hourly means (UTC)."""
    return x.resample("1h").mean()


def load_raw(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    parts: dict[str, list] = {k: [] for k in
                              ["public_power", "price", "co2eq", "fc_load", "fc_solar", "fc_wind_onshore", "fc_wind_offshore"]}
    for f in sorted(raw_dir.glob("*.json")):
        name = f.name.rsplit("_", 2)[0]
        if name not in parts:
            continue
        d = json.loads(f.read_text())
        if name == "public_power":
            parts[name].append(parse_public_power(d))
        elif name == "price":
            parts[name].append(parse_simple(d, "price", "price"))
        elif name == "co2eq":
            parts[name].append(parse_simple(d, "co2eq", "ec_co2eq"))
        else:
            parts[name].append(parse_simple(d, "forecast_values", name))
    frames = []
    for name, lst in parts.items():
        if not lst:
            raise FileNotFoundError(f"no cached files for {name} in {raw_dir}")
        x = pd.concat(lst).sort_index()
        x = x[~x.index.duplicated()]
        frames.append(to_hourly(x))
    df = pd.concat(frames, axis=1).sort_index()
    for c in GEN_COLUMNS.values():
        if c not in df:
            df[c] = 0.0
    return df
