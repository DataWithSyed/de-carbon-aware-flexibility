"""Full analysis: AEF reconstruction + validation, MEF, day-ahead AEF forecasts,
carbon-aware flexible-load scheduling and BESS emissions. Writes results/ and figures/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from carbonflex import bess, forecast, intensity, mef, schedule  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
TRAIN_END = "2025-11-01"   # MEF estimation: Nov 2023 - Oct 2025
TEST = ("2025-11-01", "2026-11-01")
LAMBDAS = [0, 25, 50, 100, 200, 400, 1000]
summary: dict = {}

df = pd.read_csv(ROOT / "data/processed/hourly_de.csv.gz", index_col="utc", parse_dates=["utc"])
df.index = pd.to_datetime(df.index, utc=True)
df["co2_t"] = intensity.emissions_t(df)
df["gen_mwh"] = intensity.generation_mwh(df)
df["aef"] = intensity.intensity_g_per_kwh(df)
df = mef.add_local_calendar(df)
df["year"] = np.where(pd.to_datetime(df["date"]) >= TRAIN_END, "Nov25-Oct26",
                      np.where(pd.to_datetime(df["date"]) >= "2024-11-01", "Nov24-Oct25", "Nov23-Oct24"))
intensity.factor_table().to_csv(RES / "emission_factors.csv", index=False)

# ---------------- 1. AEF reconstruction vs Energy-Charts co2eq ----------------
v = df.dropna(subset=["aef", "ec_co2eq"])
rows = []
for y, g in v.groupby("year"):
    rows.append({"year": y, "hours": len(g),
                 "aef_gen_weighted": g["co2_t"].sum() * 1000 / g["gen_mwh"].sum(),
                 "aef_mean": g["aef"].mean(), "ec_co2eq_mean": g["ec_co2eq"].mean(),
                 "pearson": g["aef"].corr(g["ec_co2eq"]),
                 "within_day_spearman": g.groupby("date").apply(
                     lambda d: d["aef"].corr(d["ec_co2eq"], method="spearman"), include_groups=False).mean(),
                 "direct_co2_mt": g["co2_t"].sum() / 1e6, "gen_twh": g["gen_mwh"].sum() / 1e6,
                 "fossil_share": g[["lignite", "hard_coal", "gas", "oil", "coal_gas"]].clip(lower=0).sum().sum() / g["gen_mwh"].sum()})
val = pd.DataFrame(rows)
val.to_csv(RES / "aef_validation.csv", index=False)
summary["validation"] = val.round(3).to_dict(orient="records")
summary["hours_total"] = int(len(df)); summary["hours_missing_aef"] = int(df["aef"].isna().sum())

# ---------------- 2. Marginal emission factors ----------------
tr = df[pd.to_datetime(df["date"]) < TRAIN_END]
te = df[pd.to_datetime(df["date"]) >= TEST[0]]
mef_tr = mef.fit_mef(tr["co2_t"], tr["residual_load"], tr["season"], tr["hour"])
mef_te = mef.fit_mef(te["co2_t"], te["residual_load"], te["season"], te["hour"])
mef_load = mef.fit_mef(tr["co2_t"], tr["load"], tr["season"], tr["hour"])
mef_tr.assign(period="train").to_csv(RES / "mef_residual_load_train.csv", index=False)
mef_te.assign(period="test").to_csv(RES / "mef_residual_load_test.csv", index=False)
mef_load.to_csv(RES / "mef_load_train.csv", index=False)
df["mef_signal"] = mef.lookup(mef_tr, df["season"], df["hour"])   # ex-ante signal (train fit)
df["mef_eval"] = mef.lookup(mef_te, df["season"], df["hour"])     # evaluation factor (test fit)
summary["mef"] = {
    "train_mean_t_per_mwh": round(mef_tr["mef_t_per_mwh"].mean(), 3),
    "train_median_r2": round(mef_tr["r2"].median(), 3),
    "test_mean_t_per_mwh": round(mef_te["mef_t_per_mwh"].mean(), 3),
    "load_regressor_train_mean": round(mef_load["mef_t_per_mwh"].mean(), 3),
    "load_regressor_median_r2": round(mef_load["r2"].median(), 3),
    "train_vs_test_bin_corr": round(float(np.corrcoef(mef_tr.set_index(["season", "hour"]).loc[mef_te.set_index(["season", "hour"]).index, "mef_t_per_mwh"], mef_te["mef_t_per_mwh"])[0, 1]), 3),
    "test_aef_mean_t_per_mwh": round(te["aef"].mean() / 1000, 3),
}
# who answers a change in residual load? (pooled OLS slopes of hourly first differences, test year)
comp = pd.DataFrame({
    "lignite": te["lignite"], "hard_coal": te["hard_coal"], "gas": te["gas"],
    "other_fossil": te["oil"] + te["coal_gas"] + te["others"] + te["waste"],
    "biomass": te["biomass"], "hydro_and_pumped_net": te["hydro_ror"] + te["hydro_res"] + te["pumped_gen"] + te["pumped_cons"],
    "net_import": te["net_import"], "wind_solar": te["wind_on"] + te["wind_off"] + te["solar"],
})
dR = te["residual_load"].diff()
consec = pd.Series(te.index, index=te.index).diff() == pd.Timedelta(hours=1)
resp = {}
for c in comp:
    dc = comp[c].diff()
    ok = consec & dR.notna() & dc.notna()
    resp[c] = float(np.polyfit(dR[ok], dc[ok], 1)[0])
summary["residual_load_response_shares"] = {k: round(v, 3) for k, v in resp.items()}
fossil_share = resp["lignite"] + resp["hard_coal"] + resp["gas"] + resp["other_fossil"]
summary["mef"]["import_adjusted_mean_t_per_mwh"] = {
    f"imports_at_{n}": round(summary["mef"]["test_mean_t_per_mwh"] + resp["net_import"] * f, 3)
    for n, f in [("0", 0.0), ("gas_0.404", 0.40392), ("lignite_0.957", 0.95684)]}
summary["mef"]["domestic_fossil_response_share"] = round(fossil_share, 3)
mef_aef_corr = df.loc[te.index, ["mef_eval", "aef"]].dropna()
summary["mef"]["test_hourly_corr_mef_vs_aef"] = round(mef_aef_corr["mef_eval"].corr(mef_aef_corr["aef"]), 3)

# ---------------- 3. Day-ahead AEF forecasts ----------------
x = forecast.build_features(df)
bt = forecast.rolling_backtest(x, *TEST)
models = ["naive_d7", "naive_d2", "price_only", "ridge", "gbm"]
met = forecast.metrics(bt, models)
met.to_csv(RES / "forecast_metrics.csv", index=False)
summary["forecast"] = met.round(3).to_dict(orient="records")
summary["forecast_test_hours"] = int(len(bt))
df = df.join(bt[["pred_gbm", "pred_ridge", "pred_naive_d2"]])

# ---------------- 4. Flexible-load scheduling ----------------
CASES = {"daily_flex_6MWh": dict(energy=6.0, pmax=1.0, h0=0, h1=24),
         "daytime_4MWh_08-18": dict(energy=4.0, pmax=1.0, h0=8, h1=18)}
need = ["price", "aef", "pred_gbm", "mef_signal", "mef_eval", "ec_co2eq"]
days = [g for _, g in df[pd.to_datetime(df["date"]) >= TEST[0]].groupby("date") if g[need].notna().all().all()]
summary["schedule_days"] = len(days)
sched_rows = []
for case, c in CASES.items():
    for strat in ["asap", "price", "carbon_fc", "carbon_oracle", "mef"] + [f"blend_{l}" for l in LAMBDAS[1:]]:
        cost = co2_aef = co2_mef = co2_ec = energy = 0.0
        for g in days:
            mask = ((g["hour"] >= c["h0"]) & (g["hour"] < c["h1"])).to_numpy()
            p = g["price"].to_numpy()
            if strat == "asap":
                xx = schedule.asap(len(g), c["energy"], c["pmax"], mask)
            else:
                sig = {"price": p, "carbon_fc": g["pred_gbm"].to_numpy(), "carbon_oracle": g["aef"].to_numpy(),
                       "mef": g["mef_signal"].to_numpy()}.get(strat)
                if sig is None:
                    lam = float(strat.split("_")[1])
                    sig = p + lam * g["pred_gbm"].to_numpy() / 1000.0
                xx = schedule.fill_lowest(sig, c["energy"], c["pmax"], mask)
            cost += p @ xx; energy += xx.sum()
            co2_aef += g["aef"].to_numpy() @ xx / 1000
            co2_mef += g["mef_eval"].to_numpy() @ xx
            co2_ec += g["ec_co2eq"].to_numpy() @ xx / 1000
        sched_rows.append({"case": case, "strategy": strat, "energy_mwh": energy, "cost_eur": cost,
                           "eur_per_mwh": cost / energy, "co2_aef_t": co2_aef, "co2_mef_t": co2_mef,
                           "co2_ec_t": co2_ec, "aef_kg_per_mwh": co2_aef * 1000 / energy})
sch = pd.DataFrame(sched_rows)
for case in CASES:
    m = sch["case"] == case
    base = sch[m & (sch["strategy"] == "price")].iloc[0]
    asap_ = sch[m & (sch["strategy"] == "asap")].iloc[0]
    sch.loc[m, "d_cost_vs_price_eur"] = sch.loc[m, "cost_eur"] - base["cost_eur"]
    sch.loc[m, "d_co2_aef_vs_price_t"] = sch.loc[m, "co2_aef_t"] - base["co2_aef_t"]
    sch.loc[m, "d_co2_mef_vs_price_t"] = sch.loc[m, "co2_mef_t"] - base["co2_mef_t"]
    sch.loc[m, "d_co2_aef_vs_asap_pct"] = 100 * (sch.loc[m, "co2_aef_t"] / asap_["co2_aef_t"] - 1)
    sch.loc[m, "d_co2_mef_vs_asap_pct"] = 100 * (sch.loc[m, "co2_mef_t"] / asap_["co2_mef_t"] - 1)
    saved = -sch.loc[m, "d_co2_aef_vs_price_t"]
    sch.loc[m, "abatement_eur_per_t_aef"] = sch.loc[m, "d_cost_vs_price_eur"] / saved.where(saved > 1e-6)
sch.to_csv(RES / "schedule_results.csv", index=False)
summary["schedule"] = sch.round(2).to_dict(orient="records")

# day-level bootstrap CI for carbon_fc vs price, AEF accounting (daily_flex case)
def day_delta(case, s1, s2, acc):
    c = CASES[case]; out = []
    for g in days:
        mask = ((g["hour"] >= c["h0"]) & (g["hour"] < c["h1"])).to_numpy()
        sig = {"price": g["price"], "carbon_fc": g["pred_gbm"], "mef": g["mef_signal"]}
        x1 = schedule.fill_lowest(sig[s1].to_numpy(), c["energy"], c["pmax"], mask)
        x2 = schedule.fill_lowest(sig[s2].to_numpy(), c["energy"], c["pmax"], mask)
        f = g["aef"].to_numpy() / 1000 if acc == "aef" else g["mef_eval"].to_numpy()
        out.append(f @ (x1 - x2))
    return np.array(out)

rng = np.random.default_rng(2026)
ci = {}
for case in CASES:
    for s1, acc in [("carbon_fc", "aef"), ("carbon_fc", "mef"), ("mef", "mef")]:
        d = day_delta(case, s1, "price", acc)
        boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        ci[f"{case}|{s1}-price|{acc}"] = {"mean_t_per_day": round(d.mean(), 4),
                                         "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}
summary["schedule_bootstrap"] = ci

# ---------------- 5. BESS emissions ----------------
bess_rows = []
for sig_name in ["aef_fc", "mef"]:
    for lam in [0, 50, 100, 200, 400]:
        if sig_name == "mef" and lam == 0:
            continue
        rev = thr = e_aef = e_mef = e_ec = 0.0
        for g in days:
            ef_sig = g["pred_gbm"].to_numpy() / 1000 if sig_name == "aef_fc" else g["mef_signal"].to_numpy()
            r = bess.bess_day(g["price"].to_numpy(), ef=ef_sig, carbon_price=lam)
            rev += r["revenue"]; thr += r["throughput_mwh"]
            e_aef += bess.attributed_emissions_t(r["charge"], r["discharge"], g["aef"].to_numpy() / 1000)
            e_mef += bess.attributed_emissions_t(r["charge"], r["discharge"], g["mef_eval"].to_numpy())
            e_ec += bess.attributed_emissions_t(r["charge"], r["discharge"], g["ec_co2eq"].to_numpy() / 1000)
        bess_rows.append({"signal": "none" if lam == 0 else sig_name, "carbon_price": lam, "revenue_eur": rev,
                          "discharge_mwh": thr, "co2_aef_t": e_aef, "co2_mef_t": e_mef, "co2_ec_t": e_ec})
bdf = pd.DataFrame(bess_rows)
bdf.to_csv(RES / "bess_emissions.csv", index=False)
summary["bess"] = bdf.round(2).to_dict(orient="records")

(RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

# ---------------- figures ----------------
tt = df[pd.to_datetime(df["date"]) >= TEST[0]]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
hm = tt.pivot_table(index="month", columns="hour", values="aef", aggfunc="mean")
hm = hm.reindex([11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
im = ax.imshow(hm.values, aspect="auto", cmap="RdYlGn_r")
ax.set_yticks(range(12)); ax.set_yticklabels(["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"])
ax.set_xlabel("local hour"); ax.set_title("Average emission factor, g CO2/kWh (Nov 2025-Oct 2026)")
fig.colorbar(im, ax=ax)
ax = axes[0, 1]
for s, col in zip(["DJF", "JJA"], ["tab:blue", "tab:orange"]):
    a = tt[tt["season"] == s].groupby("hour")["aef"].mean() / 1000
    m_ = mef_te[mef_te["season"] == s].set_index("hour")
    ax.plot(a.index, a.values, color=col, lw=2, label=f"AEF {s}")
    ax.errorbar(m_.index, m_["mef_t_per_mwh"], yerr=1.96 * m_["se"], color=col, ls="--", marker="o", ms=3,
                label=f"MEF {s} (d residual load, 95 % CI)")
ax.set_xlabel("local hour"); ax.set_ylabel("t CO2 / MWh"); ax.set_title("Average vs marginal emission factor (test year)")
ax.legend(fontsize=8)
ax = axes[1, 0]
s = sch[sch["case"] == "daily_flex_6MWh"].set_index("strategy")
bl = s.loc[["price"] + [f"blend_{l}" for l in LAMBDAS[1:]]]
ax.plot(bl["eur_per_mwh"], bl["aef_kg_per_mwh"], "o-", color="tab:blue", label="price + internal carbon price (EUR/t)")
for k in bl.index:
    lab = "0" if k == "price" else k.split("_")[1]
    ax.annotate(lab, (bl.loc[k, "eur_per_mwh"], bl.loc[k, "aef_kg_per_mwh"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
for k, col in [("carbon_fc", "tab:red"), ("carbon_oracle", "tab:green")]:
    ax.scatter(s.loc[k, "eur_per_mwh"], s.loc[k, "aef_kg_per_mwh"], color=col, s=50, zorder=3,
               label={"carbon_fc": "min forecast AEF (GBM)", "carbon_oracle": "min actual AEF (oracle)"}[k])
ax.text(0.98, 0.97, f"off-scale: ASAP {s.loc['asap', 'eur_per_mwh']:.1f} EUR/MWh, {s.loc['asap', 'aef_kg_per_mwh']:.0f} kg/MWh\n"
        f"MEF-signal {s.loc['mef', 'eur_per_mwh']:.1f} EUR/MWh, {s.loc['mef', 'aef_kg_per_mwh']:.0f} kg/MWh",
        transform=ax.transAxes, ha="right", va="top", fontsize=8)
ax.set_xlabel("energy cost, EUR/MWh (realised DA)"); ax.set_ylabel("AEF-accounted CO2, kg/MWh")
ax.set_title("6 MWh/day flexible load: cost vs CO2 (test year)"); ax.legend(fontsize=8, loc="lower left")
ax = axes[1, 1]
wk = tt.loc["2026-03-02":"2026-03-09"]
ax.plot(wk.index, wk["aef"], label="actual AEF", color="k")
ax.plot(wk.index, wk["pred_gbm"], label="GBM day-ahead", color="tab:red")
ax.plot(wk.index, wk["pred_naive_d2"], label="naive D-2", color="tab:gray", alpha=0.7)
ax.set_ylabel("g CO2/kWh"); ax.set_title("Day-ahead AEF forecast, example week (Mar 2026)"); ax.legend(fontsize=8)
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(FIG / "overview.png", dpi=130)
print(json.dumps({k: summary[k] for k in ["validation", "mef", "forecast"]}, indent=1, default=float))
print(sch[["case", "strategy", "eur_per_mwh", "aef_kg_per_mwh", "co2_mef_t", "d_co2_aef_vs_asap_pct", "d_co2_mef_vs_asap_pct"]].round(2).to_string())
print(json.dumps(ci, indent=1))
print(bdf.round(1).to_string())
