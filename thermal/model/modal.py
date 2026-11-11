"""Exponential-mode fit of HotSpot step responses and the periodic steady state of a two-phase power wave.

HotSpot prints temperatures to 0.01 K, so raw step responses carry rounding noise that would add up over many periods.
A step response s(t) is fitted as sum_j c_j (1 - exp(-t / tau_j)) over 36 fixed time constants, constrained to end at
the run's steady state; the periodic steady state of any on / off wave then follows in closed form (the model is linear).
Fit residual rms 0.003 K, equal to the rounding noise of the 0.01 K output (0.01 / sqrt(12)); max 0.012 K
(out/bank.json ripple.floorplans.*.modal_fit_residual_rms_max_K).
"""
import numpy as np

from .geom import DT_MS

MODE_TAUS = np.geomspace(0.2, 20000.0, 36)          # ms


def modal_fit(s, s_inf, cols, nmax):
    """Least squares on rows 1..nmax of s (row n = rise at n x 0.1 ms, row 0 = 0) with sum_j c_j = s_inf (heavy row)
    and a small ridge. Returns c (taus x columns) and the residual rms / max (K) on the fitted columns."""
    t = np.arange(1, nmax + 1) * DT_MS
    A = 1.0 - np.exp(-t[:, None] / MODE_TAUS[None, :])
    w, lam = 1e3, 1e-3
    Aa = np.vstack([A, w * np.ones((1, len(MODE_TAUS))), lam * np.eye(len(MODE_TAUS))])
    B = np.vstack([s[1:nmax + 1][:, cols], w * s_inf[None, cols], np.zeros((len(MODE_TAUS), len(cols)))])
    c = np.zeros((len(MODE_TAUS), s.shape[1]))
    with np.errstate(all="ignore"):                          # numpy 2.0 + Accelerate raise spurious matmul warnings
        c[:, cols] = np.linalg.lstsq(Aa, B, rcond=None)[0]
        full = 1.0 - np.exp(-(np.arange(1, len(s)) * DT_MS)[:, None] / MODE_TAUS[None, :])
        res = full @ c[:, cols] - s[1:, cols]
    return c, float(np.sqrt((res ** 2).mean())), float(np.abs(res).max())


def pss_t(c, a, b, t):
    """Periodic steady state at times t (ms, 0 <= t < a + b) of the zero-mean wave (1 - d on for a ms, -d off for b ms,
    d = a / (a + b)) through modes c (taus x columns); a and b need not be multiples of the 0.1 ms row."""
    d = a / (a + b)
    tau = MODE_TAUS[:, None]
    p, q = np.exp(-a / tau), np.exp(-b / tau)
    x0 = c * ((1 - d) * (1 - p) * q - d * (1 - q)) / (1 - p * q)
    xa = c * (1 - d) * (1 - p) + x0 * p
    hi = t < a
    y = np.empty((len(t), c.shape[1]))
    e1 = np.exp(-t[hi][:, None] / tau.T)
    e2 = np.exp(-(t[~hi] - a)[:, None] / tau.T)
    with np.errstate(all="ignore"):
        y[hi] = (1 - d) * c.sum(axis=0)[None] + e1 @ (x0 - (1 - d) * c)
        y[~hi] = -d * c.sum(axis=0)[None] + e2 @ (xa + d * c)
    return y
