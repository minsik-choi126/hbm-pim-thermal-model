"""HotSpot run with a chosen binary / stopping tolerance, all layers kept, energy balance recorded.

python3 run2.py <grid> <point> <src[,src..]|combo:NAME> <tol> [fp] [workers] [tagsuffix]
Output: runs2/<name>_tol<tol>/fields.npz (L0..L21 rises over ambient), .steady, .log (HS_ITER), eb.json
"""
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import HR, B, POINTS, SCR
import thermal_pim_sim as TP
import energy as E

HS = os.environ.get("HS_BIN", os.path.join(SCR, "hs_src", "hotspot_tol"))
SFX = os.environ.get("RUN_SFX", "")
OUT2 = os.path.join(SCR, "runs2")
NL = 22

# combined maps for the superposition check: {source: W} (GPU in W, others W per stack)
COMBOS = {
    "opmix": {"gpu": 332.70658, "uni": 80.09513 / 5, "array": 52.10113 / 5 * (1 - B.E_TABLE), "edge": 52.10113 / 5 * B.E_TABLE},
    "opmix_best": {"gpu": 332.70658, "uni": (1 - 0.97511) * 80.09513 / 5, "base": 0.97511 * 80.09513 / 5,
                   "array": 52.10113 / 5 * (1 - B.E_TABLE), "edge": 52.10113 / 5 * B.E_TABLE},
}


def power_row(spec, cols, m):
    v = np.zeros(len(cols))
    for s, w in spec.items():
        unit = np.array(HR.source_row(s, cols, m))
        amp = HR.GPU_W if s == "gpu" else HR.UNIT_W
        v += unit * (w / amp)
    return v


def one(job):
    name, gdir, spec, tol, grid = job
    d = os.path.join(OUT2, name)
    if os.path.exists(os.path.join(d, "eb.json")):
        return name, 0.0
    os.makedirs(d, exist_ok=True)
    m = B.load(gdir)
    m["_dir"] = gdir
    cols = TP.columns(gdir)
    row = power_row(spec, cols, m)
    pt = os.path.join(d, name + ".ptrace")
    with open(pt, "w") as f:
        f.write("\t".join(cols) + "\n")
        f.write("\t".join(f"{x:.6f}".rstrip("0").rstrip(".") or "0" for x in row) + "\n")
    cmd = [HS, "-c", "run.config", "-f", "core_and_mem_ctrl.flp", "-p", pt, "-model_type", "grid",
           "-grid_layer_file", "stack.lcf", "-detailed_3D", "on", "-steady_file", os.path.join(d, name + ".steady"),
           "-grid_steady_file", os.path.join(d, name + ".grid.steady")]
    env = dict(os.environ, HS_TOL=f"{tol:g}")
    t = time.time()
    with open(os.path.join(d, name + ".log"), "w") as lg:
        subprocess.check_call(cmd, cwd=gdir, stdout=subprocess.DEVNULL, stderr=lg, env=env)
    dt = time.time() - t
    arr = {}
    for i in range(NL):
        p = os.path.join(d, f"{name}.grid.steady[{i}]")
        arr[f"L{i}"] = E.read_grid(p, grid) - 273.15 - HR.AMB
    sink_K = arr["L21"] + 273.15 + HR.AMB
    np.savez_compressed(os.path.join(d, "fields.npz"), **arr)
    c = E.cfg(gdir)
    h = E.heat_out(sink_K, E.read_inodes(os.path.join(d, name + ".steady")), c, grid)
    log = open(os.path.join(d, name + ".log")).read()
    it = re.search(r"HS_ITER (\d+) last_delta (\S+)", log)
    rec = dict(name=name, P_in=float(row.sum()), heat_out=h, err_pct=100 * (h["total"] / row.sum() - 1), seconds=dt,
               iters=int(it.group(1)) if it else None, last_delta=float(it.group(2)) if it else None, tol=tol, grid=grid,
               spec=spec, binary=HS)
    json.dump(rec, open(os.path.join(d, "eb.json"), "w"), indent=1)
    for fn in os.listdir(d):
        if ".grid.steady[" in fn:
            os.remove(os.path.join(d, fn))
    return name, dt


def main():
    grid, ptn, srcs, tol = int(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
    fp = sys.argv[5] if len(sys.argv) > 5 else "a50_par"
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    pt = POINTS[ptn]
    g = os.environ.get("GEOM_OVERRIDE") or HR.geom(pt["k"], pt["r"], grid, fp)
    jobs = []
    if srcs.startswith("combo:"):
        for cn in srcs[6:].split(","):
            jobs.append((f"{HR.gtag(pt['k'], pt['r'], grid, fp)}_{cn}_tol{tol:g}{SFX}", g, COMBOS[cn], tol, grid))
    else:
        for s in srcs.split(","):
            amp = HR.GPU_W if s == "gpu" else HR.UNIT_W
            jobs.append((f"{HR.gtag(pt['k'], pt['r'], grid, fp)}_{s}_tol{tol:g}{SFX}", g, {s: amp}, tol, grid))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, dt in ex.map(one, jobs):
            print(f"done {n} {dt:.1f} s", flush=True)


if __name__ == "__main__":
    main()
