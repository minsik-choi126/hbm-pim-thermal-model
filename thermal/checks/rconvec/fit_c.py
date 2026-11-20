"""C: recalibration with larger spreader / sink, the repository's method (steps/recal.py) on our own run names.
For each k: r matches a_G for every x (scalar quadratic in r over the r runs), x matches a_M; along the k family the
x = 0 point and the measured row-miss ratio are interpolated linearly between k rows (recal.interp_family).
beta (model/beta.py fixed point) at each family row with the unit responses quadratic in r.

    python3 fit_c.py probe      -> r at k = 1, x = 0 per package; writes phase2_jobs.json
    python3 fit_c.py fit        -> out_c.json (also the repository package as a check against out/recal.json)
"""
import json
import os
import sys

import numpy as np

from common import HERE, RC, B, EVERY, vset, vinterp, measured, sink_mean_per_W, dump, DEFAULT_PKG
from run_batch import PKGS, RC0, SRC5

meas = measured()
TG = dict(a_G=meas["a_G"], a_M=meas["a_M"], ratio=meas["ratio"])


def fit_at(k, x, r, pkg, rs):
    rows = [RC.model_fit(vset(k, rr, SRC5, pkg), x) for rr in rs]
    if r in rs:
        return rows[list(rs).index(r)]
    return {key: RC.interp_r([row[key] for row in rows], rs)(r) for key in rows[0]}


def solve_r(k, x, pkg, rs):
    f = RC.interp_r([RC.model_fit(vset(k, rr, SRC5, pkg), x)["a_G"] for rr in rs], rs)
    lo, hi = rs[0] - 0.008, rs[-1] + 0.008
    flo, fhi = f(lo) - TG["a_G"], f(hi) - TG["a_G"]
    if flo * fhi > 0:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        fm = f(mid) - TG["a_G"]
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_x(k, pkg, rs):
    scan = tuple(round(-1.0 + 0.05 * i, 2) for i in range(81))

    def g(x):
        r = solve_r(k, x, pkg, rs)
        return None if r is None else fit_at(k, x, r, pkg, rs)["a_M"] - TG["a_M"]
    prev = None
    for x in scan:
        gx = g(x)
        if gx is None:
            prev = None
            continue
        if prev is not None and prev[1] * gx <= 0:
            lo, glo, hi = prev[0], prev[1], x
            break
        prev = (x, gx)
    else:
        return None, None
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if (gm > 0) == (glo > 0):
            lo, glo = mid, gm
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    return x, solve_r(k, x, pkg, rs)


def family(pkg, ks, rs):
    rows = []
    for k in ks:
        x, r = solve_x(k, pkg, rs)
        if x is None:
            rows.append(dict(k=k, x=None))
            continue
        f = fit_at(k, x, r, pkg, rs)
        V = vinterp(k, r, [k], rs, SRC5, pkg)
        fp = B.fixed_point(V, x, EVERY)
        rows.append(dict(k=k, x=x, r_convec=r, a_G=f["a_G"], a_M=f["a_M"], ratio=f["ratio"], gpu_die_per_nonmem_W=f["gpu_die_per_nonmem_W"],
                         beta=fp["beta"], e_x=fp["e_x"]))
    return rows


KEYS = ("k", "x", "r_convec", "a_G", "a_M", "ratio", "gpu_die_per_nonmem_W", "beta")


def point(rows, val, along):
    ok = [r_ for r_ in rows if r_.get("x") is not None]
    p = {key: RC.interp_family(ok, key, val, along) for key in KEYS}
    if p["x"] is None:
        return None
    p["e_x"] = B.threshold(B.GPU_MAIN, p["beta"])
    p["gpu_err"] = p["gpu_die_per_nonmem_W"] / meas["gpu_G"] - 1
    return p


def probe():
    out, spec = {}, []
    for nm, pkg in PKGS.items():
        rs = RC0
        r1 = solve_r(1.0, 0.0, pkg, rs)
        f = fit_at(1.0, 0.0, r1, pkg, rs) if r1 else None
        out[nm] = dict(r_at_k1_x0=r1, a_M_over_a_G=None if f is None else f["a_M"] / f["a_G"],
                       sink_mean_per_W=None if r1 is None else sink_mean_per_W(r1, pkg))
        for k in (0.75, 1.25, 1.75, 2.25):
            spec.append(dict(pkg=nm, k=k, r=None, srcs=list(SRC5)))
    jobs = [dict(it, r=r) for it in spec for r in RC0]
    json.dump(jobs, open(os.path.join(HERE, "phase2_jobs.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), "target M/G", TG["a_M"] / TG["a_G"], len(jobs), "jobs")


def fit():
    res = {}
    cases = [("repo_default", None, (0.75, 1.0, 1.25, 1.5, 1.75, 2.0), RC.RS)]
    cases += [(nm, pkg, (0.75, 1.0, 1.25, 1.75, 2.25), RC0) for nm, pkg in PKGS.items()]
    for nm, pkg, ks, rs in cases:
        rows = family(pkg, ks, rs)
        p = pkg or DEFAULT_PKG
        x0, best = point(rows, 0.0, "x"), point(rows, TG["ratio"], "ratio")
        res[nm] = dict(package_mm=dict(spreader=p["sp"] * 1e3, sink=p["sk"] * 1e3, t_spreader=p["tsp"] * 1e3, t_sink=p["tsk"] * 1e3),
                       family=rows, x0=x0, matched_ratio=best, r_grid=list(rs),
                       sink_mean_per_W_x0=None if x0 is None else sink_mean_per_W(x0["r_convec"], p))
        f = lambda d: "-" if d is None else ", ".join(f"{q} {d[q]:.4f}" for q in ("k", "x", "r_convec", "a_G", "a_M", "ratio", "gpu_err", "beta", "e_x"))  # noqa: E731
        print(f"{nm}: x0 [{f(x0)}]\n    matched [{f(best)}]", flush=True)
    dump(os.path.join(HERE, "out_c.json"), res)


if __name__ == "__main__":
    {"probe": probe, "fit": fit}[sys.argv[1]]()
