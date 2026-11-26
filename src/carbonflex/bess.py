"""Daily BESS arbitrage LP with an optional internal carbon price (paper study)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def bess_day(price: np.ndarray, ef: np.ndarray | None = None, carbon_price: float = 0.0,
             p_mw: float = 1.0, e_mwh: float = 2.0, eta_rt: float = 0.88, soc0: float = 0.5,
             wear_eur_per_mwh: float = 0.0) -> dict:
    """max sum p_t (d_t - c_t) - lambda * sum ef_t (c_t - d_t) - wear * sum d_t.

    ef in t/MWh (signal used for the carbon term). SoC_end = SoC_start = soc0*E.
    Variables x = [c(0..n-1), d(0..n-1)].
    """
    price = np.asarray(price, float)
    n = len(price)
    ef = np.zeros(n) if ef is None else np.asarray(ef, float)
    eta = np.sqrt(eta_rt)
    # minimise cost: c*(p + lam*ef) - d*(p + lam*ef) + wear*d
    sig = price + carbon_price * ef
    cvec = np.concatenate([sig, -sig + wear_eur_per_mwh])
    L = np.tril(np.ones((n, n)))
    # soc_t = soc0*E + eta*cumsum(c) - cumsum(d)/eta  in [0, E]
    A_soc = np.hstack([eta * L, -L / eta])
    s0 = soc0 * e_mwh
    A_ub = np.vstack([A_soc, -A_soc])
    b_ub = np.concatenate([np.full(n, e_mwh - s0), np.full(n, s0)])
    A_eq = np.concatenate([eta * np.ones(n), -np.ones(n) / eta])[None, :]
    res = linprog(cvec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[0.0],
                  bounds=[(0, p_mw)] * (2 * n), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    c, d = res.x[:n], res.x[n:]
    return {"charge": c, "discharge": d, "revenue": float(price @ (d - c)),
            "throughput_mwh": float(d.sum())}


def attributed_emissions_t(charge: np.ndarray, discharge: np.ndarray, ef: np.ndarray) -> float:
    """Net system CO2 attributed to the BESS under factor ef (t/MWh): charging adds, discharging displaces."""
    return float(np.asarray(ef) @ (np.asarray(charge) - np.asarray(discharge)))
