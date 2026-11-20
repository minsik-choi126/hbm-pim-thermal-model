"""Shared helpers for the r_convec / package-size experiments.

Uses the model code of the scratch clone (thermal/ of hbm-pim-thermal-model f305847) unchanged. Run cache: the clone's
thermal/runs (cached runs of the original repository symlinked in, read only). New package sizes get their own geometry
directories (config with s_spreader, s_sink, t_spreader replaced) and run names with a package suffix.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.join(HERE, "clone")
os.environ.setdefault("THERMAL_RUNS", os.path.join(CLONE, "thermal", "runs"))
os.environ.setdefault("THERMAL_OUT", os.path.join(CLONE, "thermal", "out"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, CLONE)

import numpy as np  # noqa: E402

from thermal import paths  # noqa: E402
from thermal.model import beta as B  # noqa: E402
from thermal.model import geom as G  # noqa: E402
from thermal.model import hotspot as HS  # noqa: E402
from thermal.model import measured as M  # noqa: E402
from thermal.model import power as P  # noqa: E402
from thermal.model import readout as RD  # noqa: E402
from thermal.steps import recal as RC  # noqa: E402

GRID = 64
FP = "a50_par"
K_SINK = 400.0
DEFAULT_PKG = dict(sp=0.0456, sk=0.0821, tsp=0.001, tsk=0.0069)
EVERY = RC.PIM_MAPS["every_bank"]


def pkg_suffix(pkg):
    if pkg is None or pkg == DEFAULT_PKG:
        return ""
    return f"_sp{pkg['sp'] * 1e3:g}_sk{pkg['sk'] * 1e3:g}_tsp{pkg['tsp'] * 1e3:g}"


def variant(k, r, pkg=None):
    """HotSpot run directory of (k, r) on a50_par grid 64 with package sizes pkg (None = repository package)."""
    suf = pkg_suffix(pkg)
    if not suf:
        return G.variant(k, r, GRID, FP)
    cfg = os.path.join(HERE, "configs", f"pkg{suf}.config")
    if not os.path.exists(cfg):
        os.makedirs(os.path.dirname(cfg), exist_ok=True)
        conf = open(os.path.join(CLONE, "thermal", "geometry", "h100_5stack.config")).read()
        for key, val in (("s_spreader", pkg["sp"]), ("s_sink", pkg["sk"]), ("t_spreader", pkg["tsp"]), ("t_sink", pkg["tsk"])):
            conf, n = re.subn(r"(-%s\s+)\S+" % key, lambda m: m.group(1) + f"{val:g}", conf, count=1)
            assert n == 1, key
        open(cfg, "w").write(conf)
    old_cfg, old_tag = paths.CONFIG, G.tag
    paths.CONFIG = cfg
    G.tag = lambda k_, r_, g_, fp_: old_tag(k_, r_, g_, fp_) + suf
    try:
        return G.variant(k, r, GRID, FP)
    finally:
        paths.CONFIG, G.tag = old_cfg, old_tag


def run_name(k, r, src, pkg=None):
    return f"{G.tag(k, r, GRID, FP)}{pkg_suffix(pkg)}_{src}"


def jobs(k, r, srcs, pkg=None):
    g = variant(k, r, pkg)
    m, cols = G.load_blocks(g), G.columns(g)
    return [(run_name(k, r, s, pkg), g, [P.source_row(s, cols, m)], False) for s in srcs]


_V = {}


def vset(k, r, srcs, pkg=None):
    """Unit responses (rise per W; GPU per W, others per W per stack) of each readout element."""
    V = {}
    for s in srcs:
        key = (k, r, s, pkg_suffix(pkg))
        if key not in _V:
            F = HS.fields(run_name(k, r, s, pkg))
            _V[key] = {q: v / P.amplitude(s) for q, v in RD.readout(GRID).vec(F).items()}
        V[s] = _V[key]
    return V


def vinterp(k, r, ks, rs, srcs, pkg=None):
    """Unit responses at (k, r): quadratic in r over rs (exact for 3 points) and linear in log k between the two
    bracketing ks (exact at a run)."""
    def at_k(kk):
        if r in rs:
            return vset(kk, r, srcs, pkg)
        Vs = [vset(kk, rr, srcs, pkg) for rr in rs]
        w = np.vander([r], 3)[0] @ np.linalg.pinv(np.vander(np.asarray(rs, float), 3))
        return {s: {q: sum(wi * V[s][q] for wi, V in zip(w, Vs)) for q in Vs[0][s]} for s in srcs}
    ks = sorted(ks)
    if k in ks:
        return at_k(k)
    j = int(np.searchsorted(ks, k))
    j = min(max(j, 1), len(ks) - 1)
    k0, k1 = ks[j - 1], ks[j]
    t = (np.log(k) - np.log(k0)) / (np.log(k1) - np.log(k0))
    A, Bv = at_k(k0), at_k(k1)
    return {s: {q: (1 - t) * A[s][q] + t * Bv[s][q] for q in A[s]} for s in srcs}


def measured():
    meas = json.load(open(paths.out("measured.json")))
    h = meas["h100_fit"]
    rows, _ = M.load_h100()
    gh, gr, gc, _ = M.fit(rows, y_key="tgpu")
    top = max(rows, key=lambda r_: r_["board"])
    return dict(a_G=h["s_host"], a_M=h["s_rail"], a_G_ci=h["s_host_ci"], a_M_ci=h["s_rail_ci"], c_hbm=h["c"],
                gpu_G=h["gpu_s_host"], gpu_G_ci=h["gpu_s_host_ci"], gpu_M=h["gpu_s_rail"], c_gpu=float(gc),
                ratio=meas["rowmiss"]["ratio"], ratio_ci=meas["rowmiss"]["ratio_ci95"],
                top=dict(file=top["file"], mode=top["mode"], board_W=top["board"], nonmem_W=top["host"], mem_W=top["rail"],
                         thbm=top["thbm"], tgpu=top["tgpu"],
                         hbm_rise_meas=top["thbm"] - h["c"], gpu_rise_meas=top["tgpu"] - float(gc),
                         hbm_rise_fit=h["s_host"] * top["host"] + h["s_rail"] * top["rail"],
                         gpu_rise_fit=float(gh) * top["host"] + float(gr) * top["rail"]))


def point_rise(V, x, nonmem, mem):
    Pw = {"gpu": np.array([nonmem]), "uni": np.array([(1 - x) * mem / G.STACKS])}
    if x:
        Pw["base"] = np.array([x * mem / G.STACKS])
    return float(B.temps(V, Pw, "cell")[0]), float(B.temps(V, Pw, "gpu")[0])


def sink_mean_per_W(r, pkg=None):
    p = pkg or DEFAULT_PKG
    return r + p["tsk"] / (K_SINK * p["sk"] ** 2)


def evaluate(V, x, meas, with_beta=True):
    """Model replicas of the measured regressions (recal.model_fit), beta (model/beta.py fixed point) and the rise at the
    highest-power measured point."""
    f = RC.model_fit(V, x)
    out = dict(a_G=f["a_G"], a_M=f["a_M"], a_M_over_a_G=f["a_M"] / f["a_G"], gpu_per_nonmem_W=f["gpu_die_per_nonmem_W"],
               gpu_per_mem_W=f["gpu_die_per_mem_W"], rowmiss_ratio=f.get("ratio"))
    t = meas["top"]
    h, g = point_rise(V, x, t["nonmem_W"], t["mem_W"])
    out.update(top_hbm_rise=h, top_gpu_rise=g, top_hbm_abs_with_meas_intercept=meas["c_hbm"] + h,
               top_gpu_abs_with_meas_intercept=meas["c_gpu"] + g)
    if with_beta and all(s in V for s in EVERY):
        fp = B.fixed_point(V, x, EVERY)
        out.update(beta=fp["beta"], e_x=fp["e_x"], beta_fixed_e=fp["beta_fixed_e"])
        # decomposition at the fixed-point operating point: rise per W of PIM and per W of memory at the hottest cell
        PG, Mh, PP = B.op_at(B.GPU_MAIN, fp["e_x"]) if fp["e_x"] else B.op_at(B.GPU_MAIN, B.E_START)
        base = B.temp(V, x, PG, Mh)
        out["pim_rise_per_W"] = (B.temp(V, x, PG, Mh, PP, EVERY) - base) / PP
        out["mem_rise_per_W"] = (B.temp(V, x, PG, Mh + PP) - base) / PP
    return out


def dump(path, obj):
    json.dump(obj, open(path, "w"), indent=1, default=float)
