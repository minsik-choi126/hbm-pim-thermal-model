"""The heat factor beta (one definition for every PIM distribution, memory split and readout).

beta is the memory power that raises the peak DRAM temperature as much as the PIM power does, divided by the PIM power.
Both are added at the same operating point; the memory power is placed like the model's memory heat (share x in the base
die, 1 - x over the DRAM dies). It is evaluated at the operating point of GPU+PIM at the H100 threshold with no power
constraint that the threshold model gives with this beta (fixed-point iteration, 3-7 steps). One value per PIM
distribution and split; the same value is used for H100, H200 and B200 (their own operating points move it by 0.2 % at
x = 0 in direct runs, 0.5 % from the family interpolation).

Threshold model (analysis repository, abs_threshold.dT_lin), GPU+PIM minus GPU-only:
    dT(e) = a (P_G(e) - P_G0) + b (M(e) - M0) + b beta P_P(e),    M = m0 + phi (P_M - m0)
with the measured slopes a, b and the operating points over the in-bank read energy e of
data/inputs/beta_operating_points.json. The threshold e_x is the first e where dT reaches 0 (linear interpolation on the
0.01 pJ/bit grid). With beta defined this way the condition compares GPU and memory power only, and with the GPU power
held it reproduces the direct HotSpot threshold (analysis repository, beta_definition.py: within -1.8 to +0.6 % with
the GPU power free, 6e-5 with it held, against -30 % for the local slope at a fixed point).
"""
import json

import numpy as np

from .. import paths
from . import geom as G

STACKS = G.STACKS
READ_MAX = ("cell", "coarse", "gpu")                 # maximum; bottom / base: hottest stack
GPUS = ("H100", "H200", "B200")
GPU_MAIN = "H100"

_OPS = json.load(open(paths.inputs("beta_operating_points.json")))
EF = np.round(_OPS["e_pJ_bit"]["start"] + _OPS["e_pJ_bit"]["step"] * np.arange(_OPS["e_pJ_bit"]["n"]), 6)
E_START = json.load(open(paths.inputs("beta_operating_point.json")))["e_pJ_bit"]   # start of the iteration (1.836)
SOURCE = _OPS["source"]
DEFINITION = ("memory power that raises the peak DRAM temperature as much as the PIM power does at the same operating point "
              "(memory power split like the model's memory heat), divided by the PIM power; evaluated at the GPU+PIM "
              "operating point of the H100 threshold with no power constraint that this beta gives (fixed point)")


def temps(V, Pw, ro):
    """Readout ro at many points: Pw = {source: array of W} (GPU in W, the others in W per stack)."""
    acc = None
    for s, w in Pw.items():
        w = np.atleast_1d(np.asarray(w, dtype=float))
        t = w[:, None] * V[s][ro][None, :]
        acc = t if acc is None else acc + t
    if ro in READ_MAX:
        return acc.max(axis=1)
    hot = np.argmax(acc, axis=1)
    return acc[np.arange(len(hot)), hot]


def temp(V, x, PG, M, PP=0.0, pim=None, phy=False, ro="cell"):
    """Rise of readout ro at GPU power PG, memory heat M (share x in the base die, 1 - x over the DRAM dies) and PIM
    power PP spread by pim = {source: fraction}; all powers totals over the five stacks."""
    Pw = {"gpu": PG, ("basephy" if phy else "base"): x * M / STACKS, "uni": (1 - x) * M / STACKS}
    for s, w in (pim or {}).items():
        Pw[s] = Pw.get(s, 0.0) + w * PP / STACKS
    return float(temps(V, Pw, ro)[0])


# ---------------------------------------------------------------- operating points
def gpu(g):
    return _OPS["gpus"][g]


def meff(g, PM):
    """Memory heat of the threshold model: idle + phi x (memory power above idle)."""
    o = gpu(g)
    return o["m0"] + o["phi"] * (np.asarray(PM, float) - o["m0"])


def op_at(g, e):
    """(P_G, memory heat, P_P) of GPU+PIM at in-bank read energy e (linear in e between grid points)."""
    o = gpu(g)
    return (float(np.interp(e, EF, o["P_G"])), float(meff(g, np.interp(e, EF, o["P_M"]))),
            float(np.interp(e, EF, o["P_P"])))


def gpu_only(g):
    o = gpu(g)["gpu_only"]
    return o["P_G"], float(meff(g, o["P_M"]))


def first_cross(d):
    """First e where d goes from < 0 to >= 0 (linear interpolation); None if none or d >= 0 at e = 0."""
    d = np.asarray(d, float)
    pos = d >= 0
    j = int(np.argmax(pos))
    if not pos[j] or j == 0:
        return None
    return float(EF[j - 1] + (EF[j] - EF[j - 1]) * (-d[j - 1]) / (d[j] - d[j - 1]))


def threshold(g, beta):
    """Threshold e_x (pJ/bit) of GPU g, no power constraint, at heat factor beta."""
    o = gpu(g)
    G0, M0 = gpu_only(g)
    base = o["a"] * (np.asarray(o["P_G"]) - G0) + o["b"] * (meff(g, o["P_M"]) - M0)
    return first_cross(base + beta * o["b"] * np.asarray(o["P_P"]))


# ---------------------------------------------------------------- beta
def mem_equiv(V, x, PG, M, rise, phy=False, ro="cell"):
    """Memory heat dM >= 0 with T(PG, M + dM) - T(PG, M) = rise (T is increasing and convex in dM): bisection."""
    T0 = temp(V, x, PG, M, phy=phy, ro=ro)
    f = lambda d: temp(V, x, PG, M + d, phy=phy, ro=ro) - T0 - rise  # noqa: E731
    lo, hi = 0.0, 50.0
    while f(hi) < 0:
        lo, hi = hi, 2 * hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def at_point(V, x, PG, M, PP, pim, phy=False, ro="cell"):
    """beta of PIM power PP (map pim) at the operating point (PG, M)."""
    rise = temp(V, x, PG, M, PP, pim, phy, ro) - temp(V, x, PG, M, phy=phy, ro=ro)
    return mem_equiv(V, x, PG, M, rise, phy, ro) / PP


def fixed_point(V, x, pim, phy=False, g=GPU_MAIN, ro="cell", iters=15, tol=1e-5):
    """beta at the operating point of the threshold of GPU g that this beta gives: e -> beta(e) -> e_x(beta), from
    E_START until e moves less than tol. Also beta at E_START (beta_fixed_e) for comparison."""
    one = lambda e: at_point(V, x, *op_at(g, e), pim, phy, ro)  # noqa: E731
    e, b, n, b_fix = E_START, None, 0, None
    for n in range(1, iters + 1):
        b = one(e)
        if b_fix is None:
            b_fix = b
        e2 = threshold(g, b)
        if e2 is None:                              # no crossing at this beta: move the point up the e axis
            if e >= EF[-1]:
                return dict(beta=b, e_x=None, iters=n, beta_fixed_e=b_fix)
            e = min(2 * e, float(EF[-1]))
            continue
        if abs(e2 - e) < tol:
            e = e2
            break
        e = e2
    return dict(beta=b, e_x=e, iters=n, beta_fixed_e=b_fix)
