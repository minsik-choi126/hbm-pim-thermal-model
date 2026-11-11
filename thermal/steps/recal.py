"""Step 4: recalibration with memory power split between the base die and the DRAM dies -> out/recal.json.

The calibrated model (step 2) spreads memory power over the DRAM dies. Here a third parameter, the share x of memory
power in the base die, is fitted together with the bond-layer conductivity k and the heat-sink resistance r so that
HotSpot matches three measured values at once (step 1):
    non-memory slope a_G   HBM rise per W of non-memory power (H100 fit)
    memory slope a_M       HBM rise per W of memory power (H100 fit)
    row-miss ratio         extra row-activation power (in the bank arrays) per W over streaming memory power per W
HotSpot is replayed at the measured points (89 H100 fit points, 38 row-miss rows) and regressed like the measurements.
For each k: r matches a_G for every x (quadratic in r over four r runs), x matches a_M; along this family (linear
between k runs) the row-miss ratio picks x.

beta (model/beta.py): the memory power that raises the hottest DRAM grid cell as much as the PIM power does (every bank,
MAC share at the column decoders, or a channel subset), with the memory power split like the fit (x base die, 1 - x DRAM
dies), divided by the PIM power; evaluated at the GPU+PIM operating point of the H100 threshold with no power constraint
that this beta gives (fixed point), at each fitted (k, x, r) with the unit responses quadratic in r. Keys beta_eqv_*;
summary "beta". For comparison only (labelled *_local, keys beta_every_bank, beta_<map>): the local-slope definition,
rise per W of PIM power over the local rise per W of memory power (+-dM = 20 W) at the fixed operating point of
data/inputs/beta_operating_point.json (P_G 332.7 W, P_M 80.1 W, P_P 52.1 W); _reg and _amp are two more normalisations.

Die-average readouts (die_average in the output): the same fit with the HBM temperature read as the mean of the bottom
DRAM die or of the base die of the hottest stack instead of the hottest cell; at each fitted point beta is still read at
the hottest cell (the threshold compares peak DRAM temperatures).

Reference only, not used in the paper: x from our I/O-and-TSV lower bound on the HBM-Power HBM3E model
(data/inputs/hbm3e_read_split.json) and beta there.

Grid 64. Grid 128 moves beta(x = 0) by -0.7 %, the matched beta by +0.3 % and the matched x from 0.97 to 0.90
(thermal/checks/numerics/out_best_xscan_g64_g128.json). The r grid starts at 0.008 K/W; the GPU-side base-die fits
(r 0.0078-0.0080 at k 1.25-1.75, 0.0074 at k = 4) are extrapolated below it, by up to 0.0002 K/W in quick mode
(quadratic in r; direct runs differ from the interpolation by <= 0.3 %,
thermal/checks/numerics/out_step1_direct_vs_interp.json).

Bank floorplan a50_par, unit responses of power.SRC_CORE (and the channel subsets) at k in KS x r in RS.

    python -m thermal.steps.recal [--quick] [--no-subsets]     # quick: k = 0.75-2.0 (x from about -0.2 to 1.05)
"""
import json
import sys
import time

import numpy as np

from .. import paths
from ..model import beta as B
from ..model import geom as G
from ..model import hotspot as HS
from ..model import measured as M
from ..model import power as P
from ..model import readout as RD
from .calibrate import r_convec

KS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0)
KS_QUICK = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
RS = (0.008, 0.012, 0.016, 0.020)
GRID = 64
STACKS = G.STACKS
READ_MAX = B.READ_MAX                               # maximum; bottom / base: hottest stack
temps = B.temps                                     # readout ro at many points: {source: array of W}


# ---------------------------------------------------------------- runs and unit responses
def run_name(k, r, src, fp="a50_par", grid=GRID):
    return f"{G.tag(k, r, grid, fp)}_{src}"


def jobs(ks, srcs):
    out = []
    for k in ks:
        for r in RS:
            g = G.variant(k, r, GRID, "a50_par")
            m, cols = G.load_blocks(g), G.columns(g)
            out += [(run_name(k, r, s), g, [P.source_row(s, cols, m)], False) for s in srcs]
    return out


_FV = {}


def have(k, r, srcs, fp="a50_par", grid=GRID):
    return all(HS.done(run_name(k, r, s, fp, grid)) for s in srcs)


def unit_vec(k, r, src, fp="a50_par", grid=GRID):
    key = (k, r, grid, src, fp)
    if key not in _FV:
        F = HS.fields(run_name(k, r, src, fp, grid))
        a = P.amplitude(src)
        _FV[key] = {q: v / a for q, v in RD.readout(grid).vec(F).items()}
    return _FV[key]


def vset(k, r, srcs=P.SRC_CORE + P.SRC_SUB, fp="a50_par", grid=GRID):
    V = {}
    for s in srcs:
        fpp = fp if s in ("array", "edge") + P.SRC_SUB else "a50_par"
        if have(k, r, [s], fpp, grid):
            V[s] = unit_vec(k, r, s, fpp, grid)
    return V


def vrow(k, r, fp="a50_par", grid=GRID):
    """Unit responses at sink resistance r: per readout element quadratic in r over the runs at RS (exact at a run)."""
    if r in RS:
        return vset(k, r, fp=fp, grid=grid)
    Vs = [vset(k, rr, fp=fp, grid=grid) for rr in RS]
    w = np.vander([r], 3)[0] @ np.linalg.pinv(np.vander(np.asarray(RS, float), 3))
    srcs = [s for s in Vs[0] if all(s in V for V in Vs)]
    return {s: {q: sum(wi * V[s][q] for wi, V in zip(w, Vs)) for q in Vs[0][s]} for s in srcs}


def mem_split(x, phy=False):
    return {("basephy" if phy else "base"): x, "uni": 1.0 - x}


def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X))
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ---------------------------------------------------------------- measured targets and points
_PTS = {}


def pts():
    if not _PTS:
        _PTS["h100"], _PTS["rm"] = M.h100_points(), M.rowmiss_points()
    return _PTS["h100"], _PTS["rm"]


def targets():
    meas = json.load(open(paths.out("measured.json")))
    h, rm = meas["h100_fit"], meas["rowmiss"]
    return (dict(a_G=h["s_host"], a_M=h["s_rail"], ratio=rm["ratio"]),
            dict(a_G=h["s_host_ci"], a_M=h["s_rail_ci"], ratio=rm["ratio_ci95"]),
            dict(gpu_die_per_nonmem_W=h["gpu_s_host"], gpu_die_per_nonmem_W_ci=h["gpu_s_host_ci"],
                 gpu_die_per_mem_W=h["gpu_s_rail"], gpu_die_per_mem_W_ci=h["gpu_s_rail_ci"]))


# ---------------------------------------------------------------- model slopes and beta
def model_fit(V, x, ro="cell", phy=False, op_beta=None):
    """HotSpot replicas of the measured regressions at base-die share x."""
    H, R = pts()
    sp = mem_split(x, phy)
    Pw = {"gpu": H["host"]}
    for s, w in sp.items():
        Pw[s] = w * H["rail"] / STACKS
    _, aG, aM = ols([H["host"], H["rail"]], temps(V, Pw, ro))
    out = dict(a_G=aG, a_M=aM, ratio_a_M_over_a_G=aM / aG)
    _, gG, gM = ols([H["host"], H["rail"]], temps(V, Pw, "gpu"))
    out.update(gpu_die_per_nonmem_W=gG, gpu_die_per_mem_W=gM)
    for lab in ("coarse", "bottom", "base", "cell"):
        _, g_, m_ = ols([H["host"], H["rail"]], temps(V, Pw, lab))
        out[f"ratio_a_M_over_a_G_{lab}"] = m_ / g_
    Pw = {"gpu": R["PG"], "array": R["E"] / STACKS}
    for s, w in sp.items():
        Pw[s] = w * R["E_mem_stream"] / STACKS
    _, a_, b_, bx_ = ols([R["PG"], R["S"], R["E"]], temps(V, Pw, ro))
    out.update(rm_a=a_, rm_b=b_, rm_b_x=bx_, ratio=bx_ / b_)
    if op_beta:
        out.update(betas(V, x, ro, phy, op_beta, aM))
    return out


PIM_MAPS = {"every_bank": {"array": 1 - P.E_TABLE, "edge": P.E_TABLE}, "array_only": {"array": 1.0}}
PIM_MAPS.update({s: {s: 1.0} for s in P.SRC_SUB})
PIM_MAPS.update(base_die={"base": 1.0}, uniform_dram={"uni": 1.0})


EQV_MAPS = ("every_bank",) + P.SRC_SUB


def eqv_betas(k, x, r, ro="cell", phy=False):
    """beta (model/beta.py) at the fitted (k, x, r) for every bank and the channel subsets that were run: H100 fixed
    point (beta_eqv_<map>, e_x_eqv_<map>), beta at the start point e 1.836 and at the H200 / B200 fixed points (every
    bank). Read at the hottest cell (hottest coarse block for the coarse family)."""
    V = vrow(k, r)
    bro = "coarse" if ro == "coarse" else "cell"
    out = {}
    for nm in EQV_MAPS:
        mp = PIM_MAPS[nm]
        if not all(s in V for s in mp):
            continue
        f = B.fixed_point(V, x, mp, phy, ro=bro)
        out[f"beta_eqv_{nm}"], out[f"e_x_eqv_{nm}"] = f["beta"], f["e_x"]
        if nm == "every_bank":
            out["iters_eqv_every_bank"] = f["iters"]
            out["beta_eqv_every_bank_fixed_e"] = f["beta_fixed_e"]
            for g in B.GPUS[1:]:
                out[f"beta_eqv_every_bank_{g}"] = B.fixed_point(V, x, mp, phy, g, bro)["beta"]
    return out


def betas(V, x, ro, phy, op, a_M_reg):
    """Comparison only. beta_<map> = rise per W of PIM power at op over the local rise per W of memory power (local);
    _reg: over a_M of the fit points; _amp: over the rise of the same memory split at the PIM amplitude."""
    sp = mem_split(x, phy)

    def Pw(dm=0.0, extra=None):
        d = {"gpu": np.array([op["P_G"]])}
        for s, w in sp.items():
            d[s] = np.array([w * (op["P_M"] + dm) / STACKS])
        for s, w in (extra or {}).items():
            d[s] = d.get(s, 0.0) + np.array([w])
        return d
    T0 = temps(V, Pw(), ro)[0]
    loc = (temps(V, Pw(dm=op["dM"]), ro)[0] - temps(V, Pw(dm=-op["dM"]), ro)[0]) / (2 * op["dM"])
    amp = (temps(V, Pw(dm=op["P_P"]), ro)[0] - T0) / op["P_P"]
    out = dict(a_M_local=loc, a_M_amp=amp)
    for nm, mp in PIM_MAPS.items():
        if not all(s in V for s in mp):
            continue
        rise = (temps(V, Pw(extra={s: w * op["P_P"] / STACKS for s, w in mp.items()}), ro)[0] - T0) / op["P_P"]
        out[f"beta_{nm}"] = rise / loc
        out[f"beta_{nm}_reg"] = rise / a_M_reg
        if nm == "every_bank":
            out["beta_every_bank_amp"] = rise / amp
    return out


# ---------------------------------------------------------------- fit
def interp_r(vals, rs=RS):
    c = np.polyfit(rs, vals, 2)
    return lambda r: float(np.polyval(c, r))


def at(k, x, r, ro="cell", phy=False, op_beta=None, fp="a50_par", rs=RS):
    """Model slopes at (k, x, r): exact at a run's r, else quadratic in r over the runs at rs."""
    if r in rs or len(rs) == 1:
        return model_fit(vset(k, r, fp=fp), x, ro, phy, op_beta)
    rows = [model_fit(vset(k, rr, fp=fp), x, ro, phy, op_beta) for rr in rs]
    return {key: interp_r([row[key] for row in rows], rs)(r) for key in rows[0]}


def solve_r(k, x, target, ro="cell", phy=False):
    """r_convec at which the model a_G equals the measured non-memory slope."""
    f = interp_r([model_fit(vset(k, rr, srcs=P.SRC_CORE), x, ro, phy)["a_G"] for rr in RS])
    lo, hi = RS[0] - 0.008, RS[-1] + 0.008
    flo, fhi = f(lo) - target["a_G"], f(hi) - target["a_G"]
    if flo * fhi > 0:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        fm = f(mid) - target["a_G"]
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_x(k, target, ro="cell", phy=False):
    """x (and r) at which the model matches both measured slopes: scan of x over -1..3 (0..3 for the GPU-side base-die
    placement) with r solved per x, then bisection in the first sign change."""
    scan = tuple(round((0.0 if phy else -1.0) + 0.05 * i, 2) for i in range(61 if phy else 81))

    def g(x):
        r = solve_r(k, x, target, ro, phy)
        return None if r is None else at(k, x, r, ro, phy)["a_M"] - target["a_M"]
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
        if gm is None:
            return None, None
        if (gm > 0) == (glo > 0):
            lo, glo = mid, gm
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    return x, solve_r(k, x, target, ro, phy)


def family(ks, target, ro="cell", phy=False, op_beta=None):
    rows = []
    for k in ks:
        if not all(have(k, r, P.SRC_CORE) for r in RS):
            continue
        x, r = solve_x(k, target, ro, phy)
        if x is None:
            rows.append(dict(k=k, x=None))
            continue
        row = dict(k=k, x=x, r_convec=r, **at(k, x, r, ro, phy, op_beta))
        if op_beta:
            row.update(eqv_betas(k, x, r, ro, phy))
        rows.append(row)
    return rows


def interp_family(rows, key, val, along="ratio"):
    """Linear interpolation of rows[key] where rows[along] = val, over consecutive k; None outside the rows."""
    p = [(r_[along], r_.get(key)) for r_ in rows if r_.get("x") is not None]
    for (a0, b0), (a1, b1) in zip(p, p[1:]):
        if (a0 - val) * (a1 - val) <= 0 and a0 != a1:
            return None if b0 is None or b1 is None else b0 + (b1 - b0) * (val - a0) / (a1 - a0)
    return None


KEYS = ("k", "r_convec", "a_G", "a_M", "ratio", "rm_a", "rm_b", "rm_b_x", "a_M_local", "ratio_a_M_over_a_G_coarse",
        "ratio_a_M_over_a_G_bottom", "ratio_a_M_over_a_G_base", "gpu_die_per_nonmem_W", "gpu_die_per_mem_W", "beta_every_bank",
        "beta_every_bank_reg", "beta_array_only", "beta_half_far", "beta_half_dies", "beta_quarter_far", "beta_quarter_dies",
        "beta_half_far_reg", "beta_half_dies_reg", "beta_quarter_far_reg", "beta_quarter_dies_reg", "beta_base_die",
        "beta_uniform_dram", "a_M_amp", "beta_every_bank_amp") + tuple(f"beta_eqv_{m}" for m in EQV_MAPS) + tuple(
        f"e_x_eqv_{m}" for m in EQV_MAPS) + ("beta_eqv_every_bank_fixed_e", "beta_eqv_every_bank_H200", "beta_eqv_every_bank_B200")


def point_at(rows, val, along):
    p = {key: interp_family(rows, key, val, along) for key in KEYS + ("x",)}
    if p["x"] is None:
        return None
    p["base_over_dram"] = p["beta_base_die"] / p["beta_uniform_dram"]
    for nm, (a, b) in (("beta_half", ("beta_half_far", "beta_half_dies")), ("beta_quarter", ("beta_quarter_far", "beta_quarter_dies")),
                       ("beta_eqv_half", ("beta_eqv_half_far", "beta_eqv_half_dies")),
                       ("beta_eqv_quarter", ("beta_eqv_quarter_far", "beta_eqv_quarter_dies"))):
        p[nm] = None if p[a] is None or p[b] is None else [min(p[a], p[b]), max(p[a], p[b])]
    return p


def olgun_prior():
    """Base-die share of the in-stack read power: DQ I/O in the base die; TSVs none, half or all in it."""
    sp = json.load(open(paths.inputs("hbm3e_read_split.json")))["patterns"]
    out = {}
    for pat, p in sp.items():
        tot = p["in_stack"]
        out[pat] = dict(in_stack_pJ_bit=tot, io_dq=p["io_dq"], tsv=p["tsv"], dram_core=p["dram_core"], standby=p["standby"],
                        x_lo=p["io_dq"] / tot, x_mid=(p["io_dq"] + 0.5 * p["tsv"]) / tot, x_hi=(p["io_dq"] + p["tsv"]) / tot)
    return out


DA_KEYS = ("k", "x", "r_convec", "ratio", "sensor_beta", "cell_beta", "cell_ratio", "beta_eqv")


def die_average(ks, target, target_ci, op):
    """Fit with the HBM temperature read as a die mean (bottom DRAM die, base die; hottest stack), beta read at the
    hottest cell at each fitted point (beta_eqv; cell_beta and sensor_beta: local definition, comparison). match: x whose ratio (at this readout) equals the measured one; allowed: x in
    [0, 1] whose ratio lies in the measured 95 % interval. Same method as the cell readout above."""
    out = {}
    for ro in ("bottom", "base"):
        fam = []
        for row in family(ks, target, ro, op_beta=op):
            if row.get("x") is None:
                continue
            c = at(row["k"], row["x"], row["r_convec"], "cell", op_beta=op)
            fam.append(dict(k=row["k"], x=row["x"], r_convec=row["r_convec"], ratio=row["ratio"],
                            sensor_beta=row["beta_every_bank"], cell_beta=c["beta_every_bank"], cell_ratio=c["ratio"],
                            beta_eqv=row["beta_eqv_every_bank"]))

        def pt(val, along):
            p = {key: interp_family(fam, key, val, along) for key in DA_KEYS}
            return None if p["x"] is None else p
        lo_r, hi_r = target_ci["ratio"]
        x0, x1 = pt(0.0, "x"), pt(1.0, "x")
        xa = 0.0 if x0 is not None and x0["ratio"] <= hi_r else pt(hi_r, "ratio")["x"]       # ratio falls with x
        xb = 1.0 if x1 is not None and x1["ratio"] >= lo_r else pt(lo_r, "ratio")["x"]
        assert 0.0 <= xa < xb <= 1.0, (ro, xa, xb)
        ends = [pt(xa, "x"), pt(xb, "x")]
        inner = [r_ for r_ in fam if xa < r_["x"] < xb]
        match = pt(target["ratio"], "ratio")
        assert match is not None and 0.0 <= match["x"] <= 1.0, ro
        out[ro] = dict(family=fam, match=match, allowed=dict(
            x=[xa, xb], beta_eqv=[min(p["beta_eqv"] for p in ends + inner), max(p["beta_eqv"] for p in ends + inner)],
            cell_beta=[min(p["cell_beta"] for p in ends + inner), max(p["cell_beta"] for p in ends + inner)],
            sensor_beta=[min(p["sensor_beta"] for p in ends + inner), max(p["sensor_beta"] for p in ends + inner)]))
    out["beta_range"] = [min(v["match"]["beta_eqv"] for v in out.values()), max(v["match"]["beta_eqv"] for v in out.values())]
    out["beta_range_local"] = [min(v["match"]["cell_beta"] for k_, v in out.items() if k_ != "beta_range"),
                               max(v["match"]["cell_beta"] for k_, v in out.items() if k_ != "beta_range")]
    return out


def span(rows, key, xa, xb):
    """[min, max] of rows[key] over x in [xa, xb]: the ends interpolated, the rows between them as they are; None if the
    key is missing (channel subsets in quick mode)."""
    ends = [interp_family(rows, key, v, "x") for v in (xa, xb)]
    vals = ends + [r_[key] for r_ in rows if r_.get("x") is not None and xa < r_["x"] < xb and key in r_]
    if any(v is None for v in vals):
        return None
    return [min(vals), max(vals)]


def _union(a, b):
    return None if a is None or b is None else [min(a[0], b[0]), max(a[1], b[1])]


def _lohi(a, b, i):
    return None if a is None or b is None else (a[i] if i == 0 else b[i])


def main(ks=KS, subsets=True):
    t0 = time.time()
    srcs = P.SRC_CORE + (P.SRC_SUB if subsets else ())
    hs = HS.run_all(jobs(ks, srcs), label="recal")
    target, target_ci, check = targets()
    op = json.load(open(paths.inputs("beta_operating_point.json")))
    op = {k: op[k] for k in ("P_G", "P_M", "dM", "P_P")}
    H, Rm = pts()
    res = dict(targets=target, targets_ci95=target_ci, checks_measured=check, op_beta=op, ks=list(ks), rs=RS, grid=GRID,
               measured_points=dict(h100_fit=H["n"], rowmiss=Rm["n"], rowmiss_m0_W=Rm["m0"]), prior=olgun_prior())
    rc = r_convec(1.0)
    cur = {}
    for ro in ("cell", "coarse"):
        if 1.0 in ks:
            cur[ro] = at(1.0, 0.0, rc, ro, op_beta=op)
        V = vset(1.0, rc, srcs=("gpu", "uni", "array", "edge", "base"))
        if len(V) == 5:
            cur[ro + "_exact_r"] = model_fit(V, 0.0, ro, op_beta=op)
            if ro == "cell":
                cur["cell_eqv"] = B.fixed_point(V, 0.0, PIM_MAPS["every_bank"])
    res["current_model"] = dict(k=1.0, r_convec=rc, x=0.0, **cur)
    fam = dict(cell=family(ks, target, "cell", op_beta=op), coarse=family(ks, target, "coarse", op_beta=op),
               cell_basephy=family(ks, target, "cell", True, op))
    res["family"] = fam
    sol = {}
    for lab, rows in fam.items():
        ok = [r_ for r_ in rows if r_.get("x") is not None]
        rec = dict(x_over_k=[(r_["k"], r_["x"]) for r_ in ok])
        rec["x0"], rec["x1"] = point_at(ok, 0.0, "x"), point_at(ok, 1.0, "x")
        rec["target_ratio"] = point_at(ok, target["ratio"], "ratio")
        rec["ratio_ci_hi"] = point_at(ok, target_ci["ratio"][1], "ratio")
        rec["target_ratio_reachable_x01"] = bool(rec["target_ratio"] is not None and 0.0 <= rec["target_ratio"]["x"] <= 1.0)
        rec["prior"] = {pat: {b: point_at(ok, p[b], "x") for b in ("x_lo", "x_mid", "x_hi")} for pat, p in res["prior"].items()}
        sol[lab] = rec
    res["solution"] = sol
    c = sol["cell"]
    best = c["target_ratio"] if c["target_ratio_reachable_x01"] else c["x1"]
    prior = c["prior"]["whole_row"]["x_mid"]
    res["best_fit"] = dict(best, basis="x in [0, 1] closest to the measured row-miss ratio (x = 1 if out of reach)",
                           residual=dict(a_G=best["a_G"] / target["a_G"] - 1, a_M=best["a_M"] / target["a_M"] - 1,
                                         ratio=best["ratio"] - target["ratio"],
                                         gpu_die=best["gpu_die_per_nonmem_W"] / check["gpu_die_per_nonmem_W"] - 1))
    res["prior_fit"] = dict(prior, basis="reference only: our I/O-and-TSV lower bound on the HBM-Power HBM3E model (DQ I/O and "
                                         "half the TSVs in the base die, whole-row reads); not used in the paper",
                            residual=dict(gpu_die=prior["gpu_die_per_nonmem_W"] / check["gpu_die_per_nonmem_W"] - 1))
    xs_ = [p[b] for p in res["prior"].values() for b in ("x_lo", "x_hi")]
    plo, phi_ = point_at(fam["cell"], min(xs_), "x"), point_at(fam["cell"], max(xs_), "x")
    res["prior_range"] = dict(x=[min(xs_), max(xs_)], beta=span(fam["cell"], "beta_eqv_every_bank", min(xs_), max(xs_)),
                              beta_every_bank=[phi_["beta_every_bank"], plo["beta_every_bank"]],
                              ratio=[phi_["ratio"], plo["ratio"]])
    rci = c["ratio_ci_hi"]
    assert c["x1"]["ratio"] >= target_ci["ratio"][0], "allowed range assumes x = 1 stays inside the ratio CI"
    sub = lambda key: span(fam["cell"], key, rci["x"], 1.0)  # noqa: E731
    res["allowed"] = dict(x=[rci["x"], 1.0], beta=sub("beta_eqv_every_bank"),
                          beta_half=_union(sub("beta_eqv_half_far"), sub("beta_eqv_half_dies")),
                          beta_quarter=_union(sub("beta_eqv_quarter_far"), sub("beta_eqv_quarter_dies")),
                          beta_every_bank=[c["x1"]["beta_every_bank"], rci["beta_every_bank"]],
                          beta_half_local=[_lohi(c["x1"]["beta_half"], rci["beta_half"], 0),
                                           _lohi(c["x1"]["beta_half"], rci["beta_half"], 1)],
                          beta_quarter_local=[_lohi(c["x1"]["beta_quarter"], rci["beta_quarter"], 0),
                                              _lohi(c["x1"]["beta_quarter"], rci["beta_quarter"], 1)],
                          basis="x in [0, 1] whose HotSpot row-miss ratio lies in the measured 95 % interval")
    bp = sol["cell_basephy"]["target_ratio"]
    try:
        f_bank = json.load(open(paths.out("bank.json")))["summary"]["f_bank"]
    except (OSError, KeyError, ValueError):
        f_bank = None
    eb, lb = "beta_eqv_every_bank", "beta_every_bank"
    res["beta_set"] = dict(x0=c["x0"][eb], prior=prior[eb], best=best[eb], x1=c["x1"][eb], allowed_hi=res["allowed"]["beta"][1],
                           basephy_target=None if bp is None else bp[eb],
                           coarse_x0=sol["coarse"]["x0"][eb] if sol["coarse"]["x0"] else None,
                           coarse_x1=sol["coarse"]["x1"][eb] if sol["coarse"]["x1"] else None,
                           comparison=dict(note="comparison normalisations, not used: f_bank (steps/bank.py, k = 1, fixed base point) "
                                                "and the local slope at the fixed operating point", f_bank_k1=f_bank,
                                           x0_local=c["x0"][lb], best_local=best[lb], x1_local=c["x1"][lb]))
    res["readout_ratios"] = {lab: dict(cell=p["a_M"] / p["a_G"], coarse=p["ratio_a_M_over_a_G_coarse"], bottom=p["ratio_a_M_over_a_G_bottom"],
                                       base=p["ratio_a_M_over_a_G_base"]) for lab, p in (("best", best), ("prior", prior), ("x0", c["x0"]))}
    def pt(p):     # beta (fixed point, H100), its H100 threshold, the other GPUs' fixed points; local beta for comparison
        return dict(beta=p[eb], e_x=B.threshold(B.GPU_MAIN, p[eb]), beta_H200=p[eb + "_H200"], beta_B200=p[eb + "_B200"],
                    beta_fixed_e=p[eb + "_fixed_e"], beta_local=p[lb])
    res["summary"] = dict(
        x0=dict(ratio=c["x0"]["ratio"], k=c["x0"]["k"], r=c["x0"]["r_convec"], beta_half=c["x0"]["beta_eqv_half"],
                beta_quarter=c["x0"]["beta_eqv_quarter"], **pt(c["x0"])),
        x1=dict(ratio=c["x1"]["ratio"], k=c["x1"]["k"], r=c["x1"]["r_convec"], **pt(c["x1"])),
        best=dict(x=best["x"], k=best["k"], r=best["r_convec"], ratio=best["ratio"], **pt(best),
                  beta_half=best["beta_eqv_half"], beta_quarter=best["beta_eqv_quarter"],
                  beta_half_local=best["beta_half"], beta_quarter_local=best["beta_quarter"], base_over_dram=best["base_over_dram"],
                  gpu_die_err=res["best_fit"]["residual"]["gpu_die"]),
        prior=dict(x=prior["x"], k=prior["k"], r=prior["r_convec"], ratio=prior["ratio"], **pt(prior),
                   base_over_dram=prior["base_over_dram"], gpu_die_err=res["prior_fit"]["residual"]["gpu_die"]),
        prior_range=res["prior_range"],
        allowed=dict(x=res["allowed"]["x"], beta=res["allowed"]["beta"], beta_half=res["allowed"]["beta_half"],
                     beta_quarter=res["allowed"]["beta_quarter"], beta_local=res["allowed"]["beta_every_bank"]),
        basephy_target=None if bp is None else dict(x=bp["x"], beta=bp[eb], beta_local=bp[lb]),
        current_model_cell=dict(a_G=cur["cell"]["a_G"], ratio=cur["cell"]["ratio"], beta=cur["cell_eqv"]["beta"],
                                beta_local=cur["cell"]["beta_every_bank"]) if "cell_eqv" in cur else None)
    res["die_average"] = die_average(ks, target, target_ci, op)
    res["summary"]["die_average"] = {ro: dict(x=v["match"]["x"], beta=v["match"]["beta_eqv"], allowed_x=v["allowed"]["x"],
                                              allowed_beta=v["allowed"]["beta_eqv"], beta_local=v["match"]["cell_beta"],
                                              sensor_beta_local=v["match"]["sensor_beta"], allowed_beta_local=v["allowed"]["cell_beta"])
                                     for ro, v in res["die_average"].items() if ro in ("bottom", "base")}
    res["summary"]["die_average"]["beta_range"] = res["die_average"]["beta_range"]
    res["summary"]["definition"] = B.DEFINITION
    res["seconds"] = round(time.time() - t0, 1)
    res["hotspot_seconds"] = round(hs, 1)
    json.dump(res, open(paths.out("recal.json"), "w"), indent=1, default=float)
    s = res["summary"]
    print(f"[recal] beta (H100 fixed point) x0 {s['x0']['beta']:.4f} | prior x {s['prior']['x']:.3f} beta {s['prior']['beta']:.4f} | "
          f"best x {s['best']['x']:.3f} beta {s['best']['beta']:.4f} | x1 {s['x1']['beta']:.4f} | allowed "
          f"{s['allowed']['beta'][0]:.4f}-{s['allowed']['beta'][1]:.4f} ({res['seconds']} s)", flush=True)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    main(KS_QUICK if "--quick" in a else KS, subsets="--no-subsets" not in a)
