"""Independent verification helpers: repo code imported read-only from the repository; runs under rconvec_verify/runs_<pkg>."""
import os, re, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "<repo>"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
os.environ["THERMAL_OUT"] = os.path.join(HERE, "out")
os.environ["THERMAL_RUNS"] = os.path.join(HERE, "runs_default")
sys.path.insert(0, REPO)
import numpy as np
from thermal import paths
from thermal.model import geom as G, hotspot as HS, power as P, readout as RD, beta as B
from thermal.steps import recal as RC
REPO_CONFIG = paths.CONFIG
GRID, FP = 64, "a50_par"
PKGS = {"default": None,
        "sp60_sk120": dict(s_spreader="0.06", s_sink="0.12", t_spreader="0.001"),
        "sp80_sk150": dict(s_spreader="0.08", s_sink="0.15", t_spreader="0.001"),
        "sp60_sk120_t3": dict(s_spreader="0.06", s_sink="0.12", t_spreader="0.003")}

def use_pkg(name):
    paths.RUNS = os.path.join(HERE, f"runs_{name}")
    if PKGS[name] is None:
        paths.CONFIG = REPO_CONFIG
        return
    conf = open(REPO_CONFIG).read()
    for k, v in PKGS[name].items():
        conf, n = re.subn(r"(\n\s*-%s\s+)\S+" % k, lambda m: m.group(1) + v, conf)
        assert n == 1, (k, n)
    p = os.path.join(HERE, "configs", f"{name}.config"); open(p, "w").write(conf)
    paths.CONFIG = p

def jobs(k, r, srcs):
    g = G.variant(k, r, GRID, FP)
    m, cols = G.load_blocks(g), G.columns(g)
    return [(f"v_{G.tag(k, r, GRID, FP)}_{s}", g, [P.source_row(s, cols, m)], False) for s in srcs]

def vset(k, r, srcs):
    g = os.path.join(paths.RUNS, "geom", G.tag(k, r, GRID, FP))
    return {s: {q: v / P.amplitude(s) for q, v in RD.readout(GRID).vec(HS.fields(f"v_{G.tag(k, r, GRID, FP)}_{s}")).items()} for s in srcs}
