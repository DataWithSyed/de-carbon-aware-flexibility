"""Marginal emission factors by regression of hourly changes (Hawkes 2010; Siler-Evans et al. 2012).

MEF_b = slope of  dE_t = a_b + MEF_b * dX_t  within bin b (season x local hour),
where E is hourly direct CO2 (t) and X is either residual load or load (MWh).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def add_local_calendar(df: pd.DataFrame, tz: str = "Europe/Berlin") -> pd.DataFrame:
    out = df.copy()
    loc = out.index.tz_convert(tz)
    out["date"] = loc.date
    out["hour"] = loc.hour
    out["month"] = loc.month
    out["dow"] = loc.dayofweek
    out["season"] = [SEASON[m] for m in loc.month]
    return out


def fit_mef(co2_t: pd.Series, x_mwh: pd.Series, season: pd.Series, hour: pd.Series,
            min_obs: int = 30) -> pd.DataFrame:
    """OLS slope per (season, hour) on first differences of consecutive hours (t/MWh)."""
    d = pd.DataFrame({"dE": co2_t.diff(), "dX": x_mwh.diff(), "season": season, "hour": hour})
    # only consecutive hourly steps
    step = pd.Series(co2_t.index, index=co2_t.index).diff() == pd.Timedelta(hours=1)
    d = d[step.values].dropna()
    rows = []
    for (s, h), g in d.groupby(["season", "hour"]):
        if len(g) < min_obs:
            continue
        X = np.column_stack([np.ones(len(g)), g["dX"].to_numpy()])
        y = g["dE"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        n = len(g)
        sxx = ((g["dX"] - g["dX"].mean()) ** 2).sum()
        se = float(np.sqrt(resid @ resid / (n - 2) / sxx)) if sxx > 0 else np.nan
        r2 = 1 - (resid @ resid) / (((y - y.mean()) ** 2).sum()) if n > 2 else np.nan
        rows.append({"season": s, "hour": h, "mef_t_per_mwh": beta[1], "se": se, "r2": r2, "n": n})
    return pd.DataFrame(rows)


def lookup(mef: pd.DataFrame, season: pd.Series, hour: pd.Series) -> pd.Series:
    key = mef.set_index(["season", "hour"])["mef_t_per_mwh"]
    idx = pd.MultiIndex.from_arrays([season.to_numpy(), hour.to_numpy()])
    return pd.Series(key.reindex(idx).to_numpy(), index=season.index, name="mef")
