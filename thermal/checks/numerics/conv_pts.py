"""(a) Grid convergence at the recalibration points (exact k, r of hotspot_recal.json solution.cell): unit runs at grids
32 / 64 / 128 with the original binary, metrics by hotspot_recal.model_fit (cell and coarse readouts), Richardson estimate
from 64 and 128 with the order observed over 32-64-128 (when monotone), and the base-die / DRAM-die heating ratio.

python3 conv_pts.py x0|best
"""
import json
import math
import os
import sys

import numpy as np

from analyze import HR, POINTS, RUNS2, op, rnd, unit_vec

SRCS = ("gpu", "uni", "base", "array", "edge")
KEYS = ("a_G", "a_M", "ratio", "a_M_local", "beta_every_bank", "beta_every_bank_amp", "beta_base_die", "beta_uniform_dram",
        "base_over_dram", "gpu_die_per_nonmem_W")


def V(point, grid):
    pt = POINTS[point]
    out = {}
    for s in SRCS:
        p = os.path.join(RUNS2, f"{HR.gtag(pt['k'], pt['r'], grid)}_{s}_tol1e-06", "fields.npz")
        if not os.path.exists(p):
            return None
        out[s] = unit_vec(dict(np.load(p)), grid, s)
    return out


def met(Vs, x, ro, o):
    r = HR.model_fit(Vs, x, ro, False, o)
    out = {k: float(r[k]) for k in KEYS if k in r}
    out["base_over_dram"] = out["beta_base_die"] / out["beta_uniform_dram"]
    out["beta_over_ratio"] = out["beta_every_bank"] / out["ratio"]
    return out


def main():
    point = sys.argv[1]
    o = op()
    x = POINTS[point]["x"]
    res = dict(point=POINTS[point], op=o)
    peak = {}
    for g in (32, 64, 128):
        Vs = V(point, g)
        if Vs is not None:
            peak[g] = {s: float(Vs[s]["cell"].max()) for s in SRCS}
    res["unit_peak_dram_rise_per_W"] = peak
    if 64 in peak and 128 in peak:
        res["unit_peak_rel_128_vs_64"] = {s: peak[128][s] / peak[64][s] - 1 for s in SRCS}
    print(point, "unit peak K/W (GPU per W, others per W per stack)", json.dumps(rnd(peak, 5)))
    for ro in ("cell", "coarse"):
        rows = {}
        for g in (32, 64, 128):
            Vs = V(point, g)
            if Vs is not None:
                rows[g] = met(Vs, x, ro, o)
        rec = dict(by_grid=rows)
        if 64 in rows and 128 in rows:
            rec["rel_128_vs_64"] = {k: rows[128][k] / rows[64][k] - 1 for k in rows[64]}
            ext = {}
            for k in rows[64]:
                f1, f2 = rows[64][k], rows[128][k]
                p_obs = None
                if 32 in rows:
                    d1, d2 = rows[32][k] - f1, f1 - f2
                    if d1 * d2 > 0 and abs(d2) > 0:
                        p_obs = math.log(abs(d1 / d2), 2)
                p = p_obs if (p_obs is not None and 0.5 < p_obs < 4) else 1.0     # first order when the order is not clear
                ext[k] = dict(order=p_obs, used_order=p, richardson=f2 + (f2 - f1) / (2 ** p - 1),
                              rel_extrap_vs_64=(f2 + (f2 - f1) / (2 ** p - 1)) / f1 - 1)
            rec["richardson"] = ext
        res[ro] = rec
        print(point, ro, json.dumps(rnd({g: {k: v[k] for k in ("a_G", "a_M", "ratio", "beta_every_bank", "base_over_dram", "beta_over_ratio")}
                                          for g, v in rows.items()}, 4)))
        if "rel_128_vs_64" in rec:
            print("  rel 128 vs 64", json.dumps(rnd(rec["rel_128_vs_64"], 4)))
            print("  richardson rel vs 64", json.dumps(rnd({k: [v["order"], v["rel_extrap_vs_64"]] for k, v in rec["richardson"].items()}, 3)))
    json.dump(rnd(res, 6), open(f"out_conv_{point}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
