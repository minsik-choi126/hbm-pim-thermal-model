"""Launch HotSpot runs (phase name as argument). Cached runs are skipped.
  phase1: A (k 0.95369 / 1, seven r) + B (k 0.3, 0.5, 2, 4 at r 0.1 / 0.143) + C probe (three packages, k 1, r 0.010 / 0.016 / 0.022)
  phase2: C k grid (written by fit_c.py into phase2_jobs.json)
"""
import json
import os
import shutil
import sys

from common import *  # noqa: F401,F403
from common import HS, G, P, jobs, HERE, paths

SRC5 = ("gpu", "uni", "base", "array", "edge")
RA = (0.01102, 0.01361, 0.02883, 0.05, 0.1, 0.143, 0.2)
KA = (0.95369, 1.0)
KB = (0.3, 0.5, 2.0, 4.0)
RB = (0.1, 0.143)
PKGS = {"sp60_sk120": dict(sp=0.060, sk=0.120, tsp=0.001, tsk=0.0069),
        "sp80_sk150": dict(sp=0.080, sk=0.150, tsp=0.001, tsk=0.0069),
        "sp60_sk120_t3": dict(sp=0.060, sk=0.120, tsp=0.003, tsk=0.0069)}
RC0 = (0.010, 0.016, 0.022)


def validation_job():
    """Re-run the cached x = 0 GPU unit run in a copied geometry dir to check the rebuilt binary."""
    src = G.variant(0.95369, 0.01102, 64, "a50_par")
    dst = os.path.join(paths.RUNS, "geom", "val_a50_par_k0.95369_r0.01102_g64")
    if not os.path.isdir(dst):
        shutil.copytree(os.path.realpath(src), dst)
    m, cols = G.load_blocks(dst), G.columns(dst)
    return ("val_a50_par_k0.95369_r0.01102_g64_gpu", dst, [P.source_row("gpu", cols, m)], False)


def phase1():
    J = [validation_job()]
    for k in KA:
        for r in RA:
            J += jobs(k, r, SRC5)
    for k in KB:
        for r in RB:
            J += jobs(k, r, ("gpu", "uni", "base", "array"))
    for nm, pkg in PKGS.items():
        for r in RC0:
            J += jobs(1.0, r, SRC5, pkg)
    return J


def phase2():
    spec = json.load(open(os.path.join(HERE, "phase2_jobs.json")))
    J = []
    for it in spec:
        J += jobs(it["k"], it["r"], tuple(it["srcs"]), PKGS.get(it["pkg"]))
    return J


if __name__ == "__main__":
    ph = sys.argv[1]
    J = {"phase1": phase1, "phase2": phase2}[ph]()
    print(f"{ph}: {len(J)} jobs", flush=True)
    HS.run_all(J, label=ph)
    print("DONE", flush=True)
