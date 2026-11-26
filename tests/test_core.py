import numpy as np
import pandas as pd
import pytest

from carbonflex import bess, data, intensity, mef, schedule, forecast


def test_factor_values_match_ipcc_defaults():
    f = intensity.FACTORS
    assert f["lignite"].kg_co2_per_tj == 101_000
    assert f["hard_coal"].kg_co2_per_tj == 94_600
    assert f["gas"].kg_co2_per_tj == 56_100
    # 101,000 kg/TJ * 0.0036 TJ/MWh / 0.38 = 956.8 kg/MWh
    assert f["lignite"].t_per_mwh == pytest.approx(0.9568, abs=1e-4)
    assert f["gas"].t_per_mwh == pytest.approx(0.40392, abs=1e-5)
    assert f["biomass"].t_per_mwh == 0.0


def _gen_frame(**kw):
    cols = {c: [0.0] for c in intensity.GEN}
    for k, v in kw.items():
        cols[k] = [v]
    return pd.DataFrame(cols)


def test_intensity_pure_and_mixed():
    assert intensity.intensity_g_per_kwh(_gen_frame(wind_on=100)).iloc[0] == 0
    g = intensity.intensity_g_per_kwh(_gen_frame(gas=100)).iloc[0]
    assert g == pytest.approx(403.92, abs=0.01)
    mixed = intensity.intensity_g_per_kwh(_gen_frame(gas=100, solar=100)).iloc[0]
    assert mixed == pytest.approx(g / 2)
    # negative values (e.g. tiny own-consumption readings) are clipped
    assert intensity.intensity_g_per_kwh(_gen_frame(gas=100, solar=-5)).iloc[0] == pytest.approx(g)


def test_to_hourly_averages_quarter_hours():
    idx = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    s = pd.Series([1, 2, 3, 4, 10, 10, 10, 10], index=idx, dtype=float)
    h = data.to_hourly(s)
    assert list(h) == [2.5, 10.0]


def test_parse_public_power_maps_names():
    d = {"unix_seconds": [0, 900], "production_types": [
        {"name": "Fossil gas", "data": [1.0, None]}, {"name": "Load", "data": [5.0, 6.0]},
        {"name": "Renewable share of load", "data": [1, 2]}]}
    df = data.parse_public_power(d)
    assert set(df.columns) == {"gas", "load"}
    assert np.isnan(df["gas"].iloc[1])


def test_fill_lowest_is_optimal_vs_bruteforce():
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = rng.normal(size=8)
        x = schedule.fill_lowest(s, energy=3.0, pmax=1.0)
        assert x.sum() == pytest.approx(3.0)
        best = np.sort(s)[:3].sum()
        assert s @ x == pytest.approx(best)


def test_fill_lowest_fraction_mask_and_capacity():
    s = np.array([5, 1, 2, 0, 3.0])
    mask = np.array([1, 1, 1, 0, 1], bool)
    x = schedule.fill_lowest(s, energy=1.5, pmax=1.0, mask=mask)
    assert x.tolist() == [0, 1.0, 0.5, 0, 0]
    with pytest.raises(ValueError):
        schedule.fill_lowest(s, energy=5, pmax=1.0, mask=mask)
    a = schedule.asap(5, 1.5, 1.0, mask=np.array([0, 1, 1, 1, 1], bool))
    assert a.tolist() == [0, 1.0, 0.5, 0, 0]


def test_bess_lp_buys_low_sells_high_and_conserves_energy():
    p = np.array([10, 10, 100, 100.0])
    r = bess.bess_day(p, p_mw=1, e_mwh=1, eta_rt=1.0, soc0=0.0)
    assert r["revenue"] == pytest.approx(90.0, abs=1e-6)
    assert r["charge"].sum() == pytest.approx(r["discharge"].sum())
    r2 = bess.bess_day(p, p_mw=1, e_mwh=1, eta_rt=0.81, soc0=0.0)
    # discharge = 0.81 * charge with round-trip losses
    assert r2["discharge"].sum() == pytest.approx(0.81 * r2["charge"].sum(), rel=1e-6)


def test_bess_carbon_price_can_stop_dirty_arbitrage():
    p = np.array([10, 10, 30, 30.0])
    ef = np.array([1.0, 1.0, 0.0, 0.0])  # cheap hours are dirty, expensive hours clean
    r0 = bess.bess_day(p, ef=ef, carbon_price=0.0, p_mw=1, e_mwh=1, eta_rt=1.0, soc0=0.0)
    assert bess.attributed_emissions_t(r0["charge"], r0["discharge"], ef) == pytest.approx(1.0)
    r1 = bess.bess_day(p, ef=ef, carbon_price=50.0, p_mw=1, e_mwh=1, eta_rt=1.0, soc0=0.0)
    assert r1["throughput_mwh"] == pytest.approx(0.0, abs=1e-9)


def test_mef_recovers_known_slope():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=24 * 120, freq="1h", tz="UTC")
    x = pd.Series(np.cumsum(rng.normal(0, 500, len(idx))) + 40_000, index=idx)
    e = 0.7 * x + rng.normal(0, 20, len(idx))
    cal = mef.add_local_calendar(pd.DataFrame(index=idx))
    out = mef.fit_mef(pd.Series(e, index=idx), x, cal["season"], cal["hour"])
    assert out["mef_t_per_mwh"].between(0.67, 0.73).all()
    lk = mef.lookup(out, cal["season"], cal["hour"])
    assert lk.notna().mean() > 0.99


def test_features_use_only_past_aef():
    idx = pd.date_range("2025-03-01", periods=24 * 20, freq="1h", tz="Europe/Berlin").tz_convert("UTC")
    df = pd.DataFrame(index=idx)
    df["aef"] = np.arange(len(idx), dtype=float)
    for c in ["fc_load", "fc_solar", "fc_wind_onshore", "fc_wind_offshore", "price"]:
        df[c] = 1.0
    df = mef.add_local_calendar(df)
    x = forecast.build_features(df)
    # every lagged AEF feature must be strictly older than D-1 17:00 local
    for col in ["aef_d2", "aef_d7", "aef_d1_am", "aef_d1_h16"]:
        assert (x[col].dropna() < x.loc[x[col].notna(), "aef"]).all()
    row = x.iloc[24 * 10 + 5]
    assert row["aef_d2"] == row["aef"] - 48


def test_rolling_backtest_trains_only_on_past_months():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", "2025-04-30", freq="D")
    rows = []
    for d in dates:
        for h in range(24):
            rows.append({"date": d.date(), "hour": h, "dow": d.dayofweek})
    x = pd.DataFrame(rows)
    for c in forecast.FEATURES:
        if c not in x:
            x[c] = rng.normal(size=len(x))
    x["aef"] = 300 + 50 * x["price"] + rng.normal(0, 5, len(x))
    bt = forecast.rolling_backtest(x, "2025-03-01", "2025-05-01")
    assert pd.to_datetime(bt["date"]).min() == pd.Timestamp("2025-03-01")
    assert len(bt) == 61 * 24
    m = forecast.metrics(bt, ["ridge", "gbm", "naive_d2"]).set_index("model")
    assert m.loc["ridge", "mae"] < 10 < m.loc["naive_d2", "mae"]
