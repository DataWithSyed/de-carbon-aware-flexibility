"""Bottom-up, production-based CO2 intensity of German public net generation.

Direct (stack) CO2 only, per MWh of net generation. Fuel factors are IPCC 2006
Guidelines Vol. 2 Ch. 2 Table 2.2 defaults (energy industries, kg CO2/TJ fuel).
Net electrical efficiencies are fleet-average ASSUMPTIONS (labelled below), not
plant data. Biomass CO2 is biogenic and reported as 0 here (direct fossil CO2).
Imports/exports are ignored (production-based, not consumption-based).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TJ_PER_MWH = 0.0036  # 1 MWh = 3.6 GJ = 0.0036 TJ


@dataclass(frozen=True)
class Factor:
    kg_co2_per_tj: float  # fuel emission factor (IPCC 2006 V2 Ch2 Table 2.2 default) or 0
    efficiency: float     # ASSUMED net electrical efficiency (1.0 for non-combustion)
    source: str

    @property
    def t_per_mwh(self) -> float:
        if self.kg_co2_per_tj == 0:
            return 0.0
        return self.kg_co2_per_tj * TJ_PER_MWH / self.efficiency / 1000.0


IPCC = "IPCC 2006 GL V2 Ch2 Table 2.2"
FACTORS: dict[str, Factor] = {
    "lignite": Factor(101_000, 0.38, f"{IPCC} lignite; ASSUMED eta 38 %"),
    "hard_coal": Factor(94_600, 0.42, f"{IPCC} other bituminous coal; ASSUMED eta 42 %"),
    "gas": Factor(56_100, 0.50, f"{IPCC} natural gas; ASSUMED fleet eta 50 % (CCGT/CHP/OCGT mix)"),
    "oil": Factor(77_400, 0.38, f"{IPCC} residual fuel oil; ASSUMED eta 38 %"),
    # coal-derived gas = mix of coke-oven (44,400) and blast-furnace gas (260,000) -> PROXY: hard-coal factor
    "coal_gas": Factor(94_600, 0.42, "PROXY: hard-coal factor (coke-oven/blast-furnace mix unknown)"),
    "waste": Factor(0.5 * 91_700, 0.25, f"{IPCC} MSW non-biomass 91,700 x ASSUMED 50 % fossil share; ASSUMED eta 25 %"),
    "others": Factor(56_100, 0.50, "PROXY: natural-gas factor"),
    "biomass": Factor(0, 1.0, "biogenic CO2 excluded (direct fossil CO2 only)"),
}
ZERO = ["hydro_ror", "hydro_res", "pumped_gen", "geothermal", "wind_on", "wind_off", "solar", "nuclear"]
GEN = list(FACTORS) + ZERO


def factor_table() -> pd.DataFrame:
    rows = [{"type": k, "kg_co2_per_tj": f.kg_co2_per_tj, "efficiency": f.efficiency,
             "t_co2_per_mwh": round(f.t_per_mwh, 4), "source": f.source} for k, f in FACTORS.items()]
    rows += [{"type": k, "kg_co2_per_tj": 0, "efficiency": 1.0, "t_co2_per_mwh": 0.0,
              "source": "no direct combustion"} for k in ZERO]
    return pd.DataFrame(rows)


def emissions_t(df: pd.DataFrame, factors: dict[str, Factor] | None = None) -> pd.Series:
    """Hourly direct CO2 (t) from hourly mean MW (= MWh in the hour)."""
    factors = factors or FACTORS
    out = sum(df[k].clip(lower=0).fillna(0) * f.t_per_mwh for k, f in factors.items())
    return out.rename("co2_t")


def generation_mwh(df: pd.DataFrame) -> pd.Series:
    return sum(df[k].clip(lower=0).fillna(0) for k in GEN).rename("gen_mwh")


def intensity_g_per_kwh(df: pd.DataFrame, factors: dict[str, Factor] | None = None) -> pd.Series:
    """Average emission factor (g CO2 / kWh = kg/MWh) of domestic net public generation."""
    g = generation_mwh(df)
    return (emissions_t(df, factors) * 1000.0 / g.where(g > 0)).rename("aef")
