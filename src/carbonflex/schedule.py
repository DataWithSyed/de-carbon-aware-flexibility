"""Flexible-load scheduling on price and carbon signals (one local day at a time)."""
from __future__ import annotations

import numpy as np


def fill_lowest(signal: np.ndarray, energy: float, pmax: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Energy-limited load: put power pmax into the lowest-signal allowed hours (fractional last hour).

    Exact optimum of  min sum(signal*x)  s.t. sum(x)=energy, 0<=x<=pmax*mask.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    cap = pmax * mask.sum()
    if energy > cap + 1e-9:
        raise ValueError(f"energy {energy} exceeds window capacity {cap}")
    x = np.zeros(n)
    order = [i for i in np.argsort(signal, kind="stable") if mask[i]]
    left = energy
    for i in order:
        if left <= 1e-12:
            break
        x[i] = min(pmax, left)
        left -= x[i]
    return x


def asap(n: int, energy: float, pmax: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Uncontrolled baseline: full power from the first allowed hour until energy is delivered."""
    mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    x = np.zeros(n)
    left = energy
    for i in range(n):
        if mask[i] and left > 1e-12:
            x[i] = min(pmax, left)
            left -= x[i]
    return x
