"""Analysis helpers on the audit runs (runs2/) with the analysis repository's readout and regression code (hotspot_recal).
Record: imported by the numerics scripts in <audit>/numerics; see ../README.md."""
import json
import os

import numpy as np

from common import HR, B, POINTS, SCR

RUNS2 = os.path.join(SCR, "runs2")
STK = None


def name(point, grid, src, tol, fp="a50_par"):
    pt = POINTS[point]
    return f"{HR.gtag(pt['k'], pt['r'], grid, fp)}_{src}_tol{tol:g}"


def fields_raw(point, grid, src, tol, fp="a50_par"):
    return dict(np.load(os.path.join(RUNS2, name(point, grid, src, tol, fp), "fields.npz")))


def have(point, grid, src, tol, fp="a50_par"):
    return os.path.exists(os.path.join(RUNS2, name(point, grid, src, tol, fp), "fields.npz"))


def unit_vec(F, grid, src):
    amp = HR.GPU_W if src == "gpu" else HR.UNIT_W
    return {a: v / amp for a, v in HR.readout(grid).vec(F).items()}


def vset(point, grid, tol, srcs=("gpu", "uni", "base", "basephy", "array", "edge"), fp_pim="a50_par"):
    V = {}
    for s in srcs:
        fp = fp_pim if s in ("array", "edge") else "a50_par"
        if have(point, grid, s, tol, fp):
            V[s] = unit_vec(fields_raw(point, grid, s, tol, fp), grid, s)
    return V


def vset_repo(k, r, grid=64, srcs=("gpu", "uni", "base", "basephy", "array", "edge")):
    """Unit responses of the repo's own runs (read only)."""
    V = {}
    for s in srcs:
        p = os.path.join(HR.__dict__["HERE"], "runs_recal", "runs", f"{HR.gtag(k, r, grid)}_{s}", "fields.npz")
        if os.path.exists(p):
            V[s] = unit_vec(dict(np.load(p)), grid, s)
    return V


def op():
    return HR.op_beta_decode()


def metrics(V, x, ro="cell", opb=None, phy=False):
    opb = opb or op()
    r = HR.model_fit(V, x, ro, phy, opb)
    keep = ("a_G", "a_M", "ratio", "rm_b", "rm_b_x", "a_M_local", "a_M_amp", "beta_every_bank", "beta_every_bank_reg",
            "beta_every_bank_amp", "beta_base_die", "beta_uniform_dram", "gpu_die_per_nonmem_W", "gpu_die_per_mem_W")
    out = {k: float(r[k]) for k in keep if k in r}
    if "beta_base_die" in r and "beta_uniform_dram" in r:
        out["base_over_dram"] = float(r["beta_base_die"] / r["beta_uniform_dram"])
    return out


def full_field(Fs, weights):
    """Sum of raw unit fields scaled to the given powers: Fs[src] = raw field dict, weights[src] = W (GPU W, others W per stack)."""
    out = None
    for s, w in weights.items():
        amp = HR.GPU_W if s == "gpu" else HR.UNIT_W
        f = {k: v * (w / amp) for k, v in Fs[s].items()}
        out = f if out is None else {k: out[k] + f[k] for k in out}
    return out


def locate_peak(F, grid):
    """Hottest DRAM cell inside the stack footprints: stack, die (1 = bottom), x, y (mm), distance to the GPU-facing edge."""
    ro = HR.readout(grid)
    m = B.load(B.geom(0.50, "par", grid))
    fp = m["stacks"]
    gx = m["gx"]
    n = grid
    xc = (np.arange(n) + 0.5) * HR.TP.S.W / n
    yc = HR.TP.S.H - (np.arange(n) + 0.5) * HR.TP.S.H / n
    best = None
    for d in range(8):
        L = np.where(ro.inside, F[f"L{4 + 2 * d}"], -np.inf)
        i, j = np.unravel_index(np.argmax(L), L.shape)
        if best is None or L[i, j] > best[0]:
            best = (L[i, j], d, i, j)
    T, d, i, j = best
    x, y = xc[j], yc[i]
    stack = [s for s, (x0, y0, x1, y1) in fp.items() if x0 < x < x1 and y0 < y < y1][0]
    x0, y0, x1, y1 = fp[stack]
    gpu_hi = (x0 + x1) / 2 < gx
    dist = (x1 - x) if gpu_hi else (x - x0)
    # per-die and per-stack maxima for context
    per_die = [float(F[f"L{4 + 2 * dd}"][ro.inside].max()) for dd in range(8)]
    per_stack = {int(s): float(max(F[f"L{4 + 2 * dd}"][mk].max() for dd in range(8))) for s, mk in enumerate(ro.stack_masks)}
    return dict(rise=float(T), stack=int(stack), die=d + 1, x_mm=x * 1e3, y_mm=y * 1e3, dist_to_gpu_edge_mm=dist * 1e3,
                stack_xy_mm=[v * 1e3 for v in fp[stack]], per_die_max=per_die, per_stack_max=per_stack,
                cell_mm=HR.TP.S.W / n * 1e3)


def rnd(x, n=5):
    return HR.TP.rnd(x, n)


if __name__ == "__main__":
    print(json.dumps(rnd(op()), indent=0))
