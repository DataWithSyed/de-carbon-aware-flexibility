"""Day-ahead forecasts of the hourly average emission factor (AEF).

Issue time: day D-1 18:00 Europe/Berlin. Allowed information for day D:
Energy-Charts day-ahead forecasts of load, solar, wind on/offshore for D, the
DE-LU day-ahead price for D (auction results published ~12:45 D-1), AEF of
day D-1 up to hour 16 (local), and AEF of days D-2 and D-7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["fc_load", "fc_solar", "fc_wind", "fc_residual", "fc_res_share", "price",
            "aef_d2", "aef_d7", "aef_d1_am", "aef_d1_h16", "hour_sin", "hour_cos",
            "doy_sin", "doy_cos", "weekend"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns aef, price, fc_* and local calendar (date, hour, dow)."""
    x = df.copy()
    x["fc_wind"] = x["fc_wind_onshore"].fillna(0) + x["fc_wind_offshore"].fillna(0)
    x["fc_residual"] = x["fc_load"] - x["fc_solar"].fillna(0) - x["fc_wind"]
    x["fc_res_share"] = (x["fc_solar"].fillna(0) + x["fc_wind"]) / x["fc_load"]
    piv = x.pivot_table(index="date", columns="hour", values="aef", aggfunc="mean")
    dates = pd.Index(sorted(piv.index))
    piv = piv.reindex(dates)
    prev = {k: piv.shift(k) for k in (1, 2, 7)}
    am = piv.loc[:, [h for h in piv.columns if h <= 11]].mean(axis=1).shift(1)
    h16 = piv.get(16, pd.Series(np.nan, index=dates)).shift(1)
    key = list(zip(x["date"], x["hour"]))

    def take(p):
        st = p.stack(future_stack=True)
        return st.reindex(key).to_numpy()

    x["aef_d2"] = take(prev[2])
    x["aef_d7"] = take(prev[7])
    x["aef_d1_am"] = am.reindex(x["date"]).to_numpy()
    x["aef_d1_h16"] = h16.reindex(x["date"]).to_numpy()
    x["hour_sin"] = np.sin(2 * np.pi * x["hour"] / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x["hour"] / 24)
    doy = pd.to_datetime(x["date"]).dt.dayofyear.to_numpy()
    x["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    x["weekend"] = (x["dow"] >= 5).astype(float)
    return x


def make_models(seed: int = 0) -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "price_only": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gbm": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=40, l2_regularization=1.0, random_state=seed),
    }


def rolling_backtest(x: pd.DataFrame, test_start: str, test_end: str, seed: int = 0) -> pd.DataFrame:
    """Expanding-window, refit at the start of each local month of the test period."""
    x = x.dropna(subset=FEATURES + ["aef"]).copy()
    x["d"] = pd.to_datetime(x["date"])
    months = pd.date_range(test_start, test_end, freq="MS")
    out = []
    for m0, m1 in zip(months[:-1], months[1:]):
        tr = x[x["d"] < m0]
        te = x[(x["d"] >= m0) & (x["d"] < m1)].copy()
        if te.empty:
            continue
        for name, model in make_models(seed).items():
            cols = ["price", "hour_sin", "hour_cos"] if name == "price_only" else FEATURES
            model.fit(tr[cols], tr["aef"])
            te[f"pred_{name}"] = model.predict(te[cols])
        te["pred_naive_d2"] = te["aef_d2"]
        te["pred_naive_d7"] = te["aef_d7"]
        out.append(te)
    return pd.concat(out)


def metrics(bt: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        e = bt[f"pred_{m}"] - bt["aef"]
        # within-day rank agreement: what matters for shifting load
        sp = bt.groupby("date").apply(lambda g: g[f"pred_{m}"].corr(g["aef"], method="spearman"),
                                      include_groups=False)
        rows.append({"model": m, "mae": e.abs().mean(), "rmse": float(np.sqrt((e ** 2).mean())),
                     "bias": e.mean(), "within_day_spearman": sp.mean()})
    return pd.DataFrame(rows)
