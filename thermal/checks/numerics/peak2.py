"""(d, e) Where the hottest DRAM cell sits, and beta under readouts that avoid the chip boundary.

Fields at the decode operating point of hotspot_recal (GPU 332.7 W, memory 80.1 W split x, PIM 52.1 W at every bank, PE
share 3.5 %), built from the unit runs (superposition). Readouts of the DRAM dies (layers 4..18) inside the stack
footprints: cell (repo main), interior (cells at least 1 mm from the chip boundary), gpu_half (GPU-facing half of each
stack), far_half, and the base die (layer 2) inside the footprints. beta = rise per W of PIM over the local memory slope
(finite difference +-20 W), as hotspot_recal.betas.

python3 peak2.py <grid> [sfx]      sfx: '' (original binary) or '_sym'
"""
import json
import os
import sys

import numpy as np

from analyze import HR, B, POINTS, RUNS2, op, rnd

E = B.E_TABLE


def raw(point, src, grid, sfx=""):
    pt = POINTS[point]
    return dict(np.load(os.path.join(RUNS2, f"{HR.gtag(pt['k'], pt['r'], grid)}_{src}_tol1e-06{sfx}", "fields.npz")))


def geometry(grid):
    m = B.load(B.geom(0.50, "par", grid))
    fp, gx = m["stacks"], m["gx"]
    n = grid
    W, H = HR.TP.S.W, HR.TP.S.H
    xc = (np.arange(n) + 0.5) * W / n
    yc = H - (np.arange(n) + 0.5) * H / n
    X, Y = np.meshgrid(xc, yc)
    st = -np.ones((n, n), int)
    gpu_half = np.zeros((n, n), bool)
    dgpu = np.full((n, n), np.nan)
    for s, (x0, y0, x1, y1) in fp.items():
        mk = (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
        st[mk] = s
        hi = (x0 + x1) / 2 < gx
        gpu_half |= mk & ((X > (x0 + x1) / 2) if hi else (X < (x0 + x1) / 2))
        dgpu[mk] = ((x1 - X) if hi else (X - x0))[mk]
    inside = st >= 0
    bdist = np.minimum.reduce([X, W - X, Y, H - Y])
    masks = dict(cell=inside, interior=inside & (bdist > 1.0e-3), gpu_half=gpu_half, far_half=inside & ~gpu_half)
    return dict(masks=masks, st=st, X=X, Y=Y, dgpu=dgpu, bdist=bdist, fp=fp, cw=W / n)


def field(Fs, w):
    out = None
    for s, p in w.items():
        if not p:
            continue
        amp = HR.GPU_W if s == "gpu" else HR.UNIT_W
        f = {k: v * p / amp for k, v in Fs[s].items()}
        out = f if out is None else {k: out[k] + f[k] for k in out}
    return out


def rmax(F, mk):
    return max(F[f"L{4 + 2 * d}"][mk].max() for d in range(8))


def where(F, g):
    best = None
    for d in range(8):
        L = np.where(g["masks"]["cell"], F[f"L{4 + 2 * d}"], -np.inf)
        i, j = np.unravel_index(np.argmax(L), L.shape)
        if best is None or L[i, j] > best[0]:
            best = (float(L[i, j]), d + 1, i, j)
    T, d, i, j = best
    s = int(g["st"][i, j])
    x0, y0, x1, y1 = g["fp"][s]
    return dict(rise_K=T, stack=s, die_from_bottom=d, x_mm=g["X"][i, j] * 1e3, y_mm=g["Y"][i, j] * 1e3,
                to_gpu_edge_mm=g["dgpu"][i, j] * 1e3, to_chip_boundary_mm=g["bdist"][i, j] * 1e3,
                base_die_max_in_stacks_K=float(F["L2"][g["masks"]["cell"]].max()),
                interior_max_K=float(rmax(F, g["masks"]["interior"])), gpu_half_max_K=float(rmax(F, g["masks"]["gpu_half"])),
                per_die_max_K=[float(F[f"L{4 + 2 * dd}"][g["masks"]["cell"]].max()) for dd in range(8)])


def main():
    grid = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    sfx = sys.argv[2] if len(sys.argv) > 2 else ""
    o = op()
    g = geometry(grid)
    res = dict(op=o, grid=grid, binary="symmetric lateral conductance" if sfx else "original", cell_mm=g["cw"] * 1e3)
    for point in ("x0", "best"):
        try:
            Fs = {s: raw(point, s, grid, sfx) for s in ("gpu", "uni", "base", "array", "edge")}
        except FileNotFoundError as ex:
            res[point] = dict(missing=str(ex))
            continue
        x = POINTS[point]["x"]
        mem = lambda P: {"uni": (1 - x) * P / 5, "base": x * P / 5}  # noqa: E731
        pim = {"array": (1 - E) * o["P_P"] / 5, "edge": E * o["P_P"] / 5}
        cases = {"gpu_only": {"gpu": o["P_G"]}, "gpu_mem": {"gpu": o["P_G"], **mem(o["P_M"])},
                 "gpu_mem_pim": {"gpu": o["P_G"], **mem(o["P_M"]), **pim}, "pim_alone": pim, "mem_alone": mem(o["P_M"])}
        rec = {"location": {c: where(field(Fs, w), g) for c, w in cases.items()}, "beta": {}}
        # beta per readout
        F0 = field(Fs, cases["gpu_mem"])
        Fp = field(Fs, {"gpu": o["P_G"], **mem(o["P_M"]), **pim})
        Fhi = field(Fs, {"gpu": o["P_G"], **mem(o["P_M"] + o["dM"])})
        Flo = field(Fs, {"gpu": o["P_G"], **mem(o["P_M"] - o["dM"])})
        for ro, mk in g["masks"].items():
            t0, tp, th, tl = rmax(F0, mk), rmax(Fp, mk), rmax(Fhi, mk), rmax(Flo, mk)
            rec["beta"][ro] = dict(beta=((tp - t0) / o["P_P"]) / ((th - tl) / (2 * o["dM"])), T0=t0)
        mkb = g["masks"]["cell"]
        t0, tp = F0["L2"][mkb].max(), Fp["L2"][mkb].max()
        th, tl = Fhi["L2"][mkb].max(), Flo["L2"][mkb].max()
        rec["beta"]["base_die_max"] = dict(beta=((tp - t0) / o["P_P"]) / ((th - tl) / (2 * o["dM"])), T0=float(t0))
        res[point] = rec
        L = rec["location"]
        for c in cases:
            l_ = L[c]
            print(point, c, "rise %.2f stack %d die %d  %.2f mm from GPU edge, %.2f mm from chip boundary; interior max %.2f, "
                  "gpu-half max %.2f, base-die max %.2f" % (l_["rise_K"], l_["stack"], l_["die_from_bottom"], l_["to_gpu_edge_mm"],
                                                           l_["to_chip_boundary_mm"], l_["interior_max_K"], l_["gpu_half_max_K"],
                                                           l_["base_die_max_in_stacks_K"]))
        print(point, "beta", json.dumps(rnd({k: v["beta"] for k, v in rec["beta"].items()}, 4)))
    json.dump(rnd(res, 5), open(f"out_peak2_g{grid}{sfx}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
