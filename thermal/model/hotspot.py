"""HotSpot runs with a cache under paths.RUNS/runs/<name>.

steady run     -> <name>.steady (block centres, K) and fields.npz (grid temperature rise above ambient, K, of the base-die
                  layer L2 and the DRAM-die layers L4..L18)
transient run  -> <name>.steady and ttrace.npz (header, rows x blocks in degC, one row per 0.1 ms)
A run is reused when its 'done' marker exists; the ptrace and the raw grid dumps are deleted after each run.
"""
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import paths
from . import geom as G
from .power import fmt


def run_dir(name):
    return os.path.join(paths.RUNS, "runs", name)


def done(name):
    return os.path.exists(os.path.join(run_dir(name), "done"))


def _grid(gdir):
    return int(re.search(r"-grid_rows\s+(\d+)", open(os.path.join(gdir, "run.config")).read()).group(1))


def run(job):
    """job = (name, gdir, rows, transient). Returns (name, seconds; 0 if cached)."""
    name, gdir, rows, transient = job
    d = run_dir(name)
    if done(name):
        return name, 0.0
    if not os.path.exists(paths.HOTSPOT):
        raise SystemExit(f"HotSpot binary missing: {paths.HOTSPOT} (python -m thermal build)")
    os.makedirs(d, exist_ok=True)
    cols = G.columns(gdir)
    pt = os.path.join(d, name + ".ptrace")
    with open(pt, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(fmt(x) for x in r) + "\n")
    cmd = [paths.HOTSPOT, "-c", "run.config", "-f", "core_and_mem_ctrl.flp", "-p", pt, "-model_type", "grid",
           "-grid_layer_file", "stack.lcf", "-detailed_3D", "on", "-steady_file", os.path.join(d, name + ".steady")]
    gs = os.path.join(d, name + ".grid.steady")
    tt = os.path.join(d, name + ".ttrace")
    cmd += ["-o", tt] if transient else ["-grid_steady_file", gs]
    t0 = time.time()
    with open(os.path.join(d, name + ".log"), "w") as lg:
        subprocess.check_call(cmd, cwd=gdir, stdout=subprocess.DEVNULL, stderr=lg)
    dt = time.time() - t0
    if transient:
        lines = open(tt).read().split("\n")
        head = lines[0].split()
        arr = np.array([[float(x) for x in ln.split()] for ln in lines[1:] if ln.strip()])
        assert arr.shape[0] == len(rows), (name, arr.shape)
        np.savez_compressed(os.path.join(d, "ttrace.npz"), head=np.array(head), T=arr)
        os.remove(tt)
    else:
        n = _grid(gdir)
        arr = {}
        for i in G.LAYERS:
            p = f"{gs}[{i}]"
            arr[f"L{i}"] = np.array([float(ln.split()[1]) for ln in open(p) if ln.strip()]).reshape(n, n) - 273.15 - G.AMB
        np.savez_compressed(os.path.join(d, "fields.npz"), **arr)
        for fn in os.listdir(d):
            if ".grid.steady[" in fn:
                os.remove(os.path.join(d, fn))
    os.remove(pt)
    open(os.path.join(d, "done"), "w").write(f"{dt:.1f} s\n")
    return name, dt


def run_all(jobs, workers=None, label="hotspot"):
    """Run jobs in parallel (cached ones are skipped); prints progress; returns total HotSpot seconds."""
    todo = [j for j in jobs if not done(j[0])]
    seen, uniq = set(), []
    for j in todo:
        if j[0] not in seen:
            seen.add(j[0])
            uniq.append(j)
    if not uniq:
        print(f"[{label}] {len(jobs)} runs cached", flush=True)
        return 0.0
    workers = workers or paths.WORKERS
    print(f"[{label}] {len(uniq)} HotSpot runs ({len(jobs) - len(uniq)} cached), {workers} workers", flush=True)
    t0, tot = time.time(), 0.0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (n, dt) in enumerate(ex.map(run, uniq)):
            tot += dt
            print(f"[{label}] {i + 1}/{len(uniq)} {n} {dt:.0f} s (elapsed {time.time() - t0:.0f} s)", flush=True)
    return tot


def steady(name):
    """Block-centre temperatures (degC) of a run: {layer_<i>_<block>: T}."""
    p = os.path.join(run_dir(name), name + ".steady")
    return {k: float(v) - 273.15 for k, v in (ln.split() for ln in open(p))}


def fields(name):
    """Grid rise above ambient (K): {"L2": array, "L4": ..., "L18": ...}."""
    return dict(np.load(os.path.join(run_dir(name), "fields.npz")))


def ttrace(name):
    z = np.load(os.path.join(run_dir(name), "ttrace.npz"))
    return [str(h) for h in z["head"]], z["T"]
