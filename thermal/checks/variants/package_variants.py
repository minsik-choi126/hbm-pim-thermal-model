"""Record: ran inside the analysis repository (<analysis>), not from this repository. Package variants of the H100 5-stack HotSpot model: does the beta of the split that reaches the measured row-miss ratio
depend on the package build? (HotSpot audit 11/11, the audit report 2.1 and 2.3; scripts geometry/sens.py, geometry/analyze2.py,
geometry/summary.py and skeptic/sens_sk.py, skeptic/analyze_sk.py of the audit, merged.)

Base geometry: hotspot_recal.geom(1, 0.012, 64) (bank floorplan a50_par, grid 64, runs_recal/geom/a50_par_k1_r0.01200_g64).
Each variant changes one physical parameter (VARIANTS below); the bond conductivity k inside the stacks is set per run and
r_convec is 0.011. Sources per run (hotspot_recal units): gpu (GPU die, 1000 W), uni (DRAM dies, area weighted, 200 W per
stack), base (base-die blocks, 200 W per stack), array (bank arrays, 200 W per stack).

Refit per variant and end share x (x = 0: memory power all in the DRAM dies, x = 1: all in the base die), with
hotspot_recal.model_fit on the measured H100 points and the measured row-miss rows (cell readout):
  * at each k with runs, a uniform shift d (K per W of total power, the r_convec change) so that a_G equals the measured
    non-memory slope;
  * then k* at which a_M equals the measured memory slope: over the variant's own k runs if it has two or more, else the
    base curve shifted by the variant's offset at its one k (k = 1 for x = 0, k = 2 for x = 1);
  * beta (PIM in the bank arrays, hotspot_recal beta_array_only, local normalization at the op_beta point of
    hotspot_recal.json), the GPU-die slope per W of non-memory power, and the row-miss ratio at k*.
Summary per variant: beta / ratio at both ends, the share x where the HotSpot row-miss ratio meets the measured 0.70
(linear in x between the ends; x_match_unclamped, and x_match clamped to [0, 1]), beta there (beta_match_unclamped,
beta_match), and the H100 threshold with no power constraint at those betas (hotspot_recal.json threshold_sweep).

Inputs: hotspot_recal.json (op_beta, threshold_sweep), runs_recal/geom/a50_par_k1_r0.01200_g64, the measured points read
by hotspot_recal.pts(). Outputs: <analysis>/pipeline/e2e_joint/runs_recal/hotspot_checks/package/{geom,runs}/ and
<analysis>/pipeline/e2e_joint/hotspot_checks/package_variants.json (copied here).

    python3 <analysis>/pipeline/e2e_joint/hotspot_checks/package_variants.py runs [variant ...]   # HotSpot runs (about 1 min each)
    python3 <analysis>/pipeline/e2e_joint/hotspot_checks/package_variants.py [variant ...]        # refit -> package_variants.json
Worker count: environment variable NW (default 4). Without variants, all of VARIANTS.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

HR = C.HR
OUT = C.runs_dir("package")
STACKS, GPU_W, UNIT_W, AMB = HR.STACKS, HR.GPU_W, HR.UNIT_W, HR.AMB
LC_RE = HR.LC_RE
GX0, GX1, GY0, GY1 = HR.TP.S.GX0, HR.TP.S.GX1, HR.TP.S.GY0, HR.TP.S.GY1
R0 = 0.011
AMP = {"gpu": GPU_W, "uni": UNIT_W, "base": UNIT_W, "array": UNIT_W}

# variant: (description, family). family groups the variants of the paper statements: package (the build of the
# package and stack), gpu_side (closes the GPU-die slope gap), hbm_tim (top TIM over the HBM only).
VARIANTS = {
    "base": ("unchanged model", "package"),
    "tim2": ("top TIM (layer 19) resistance doubled: 20 um at k 2", "package"),
    "topdie": ("top DRAM die 140 um (stack 630 -> 720 um)", "package"),
    "interp100": ("interposer 100 um (CoWoS-S)", "package"),
    "mold1": ("molding / gap filler k 7.7 -> 1 W/m-K in all layers", "package"),
    "vc": ("vapor-chamber-like spreader: k 400 -> 4000 W/m-K, 1 -> 3 mm", "package"),
    "ubump1": ("die-to-interposer bump / underfill layer k 4 -> 1", "package"),
    "base100": ("base die (and GPU-die layer) 100 um", "package"),
    "die35": ("core DRAM dies 35 um (dies 1-7)", "package"),
    "hmslike": ("layer thicknesses of the Hong et al. layer file: interposer 100 um, bottom bond 33 um k 1.5, top die "
                "135 um, TIM 10 mm2-K/W, mold k 0.9", "package"),
    "tim14": ("top TIM 56 um at k 4 (14 mm2-K/W) everywhere", "package"),
    "sec": ("secondary (bottom) path on: interposer -> C4 -> substrate -> board, 0.5 K/W board to air", "package"),
    "timH14": ("top TIM 14 mm2-K/W only over the HBM stacks", "hbm_tim"),
    "timG14": ("top TIM 14 mm2-K/W only over the GPU die", "gpu_side"),
    "core70": ("GPU power only in the central 70 % of the die area", "gpu_side"),
}
OWN_K_X0 = {"base": (0.7, 1.0, 1.4), "tim2": (0.7, 1.0, 1.4), "topdie": (0.7, 1.0, 1.4), "interp100": (0.7, 1.0)}
BASE_K = {0: (0.7, 1.0, 1.4), 1: (1.4, 2.0, 2.8)}
ONE_K = {0: 1.0, 1: 2.0}


def jobs_of(var):
    """[(var, k, r, src)] of one variant: x = 0 sources (gpu, uni, array) at k = 1 (and the own k runs), x = 1 sources
    (gpu, base, array) at k = 2; the base model at all five k with all four sources; core70 only changes the GPU source."""
    if var == "core70":
        return [(var, k, R0, "gpu") for k in (1.0, 2.0)]
    if var == "base":
        return [(var, k, R0, s) for k in (0.7, 1.0, 1.4, 2.0, 2.8) for s in ("gpu", "uni", "base", "array")]
    out = [(var, k, R0, s) for k in OWN_K_X0.get(var, (1.0,)) for s in ("gpu", "uni", "array")]
    return out + [(var, 2.0, R0, s) for s in ("gpu", "base", "array")]


# ---------------------------------------------------------------- geometry
def set_layer(lcf, layer, thickness=None, rho=None):
    out = []
    for blk in lcf.split("\n\n"):
        lines = blk.split("\n")
        idx = [i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith("#")]
        if len(idx) == 7 and int(lines[idx[0]]) == layer:
            if rho is not None:
                lines[idx[4]] = repr(rho)
            if thickness is not None:
                lines[idx[5]] = repr(thickness)
        out.append("\n".join(lines))
    return "\n\n".join(out)


def set_fill_k(tmp, k):
    """Thermal conductivity of the molding / gap-filler blocks (X_g0, X_g1, X_c0, X_c1) in every floorplan file."""
    for fn in os.listdir(tmp):
        if fn.endswith(".flp"):
            p = os.path.join(tmp, fn)
            o = []
            for ln in open(p).read().splitlines():
                q = ln.split()
                if q and q[0] in ("X_g0", "X_g1", "X_c0", "X_c1") and len(q) == 7:
                    q[6] = k
                    ln = "\t".join(q)
                o.append(ln)
            open(p, "w").write("\n".join(o) + "\n")


def geom(var, k, r):
    dst = os.path.join(OUT, "geom", f"{var}_k{k:g}_r{r:.5f}")
    if os.path.isdir(dst):
        return dst
    src = HR.geom(1.0, 0.012, 64)
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    lines = open(os.path.join(tmp, "tim_hbm.flp")).read().splitlines()        # bond k inside the active stacks
    out = []
    for ln in lines:
        p = ln.split()
        if p and p[0] in ("TB_L", "TB_R"):
            p[6] = f"{1.0 / k:.6g}"
            ln = "\t".join(p)
        out.append(ln)
    open(os.path.join(tmp, "tim_hbm.flp"), "w").write("\n".join(out) + "\n")
    conf = open(os.path.join(tmp, "run.config")).read()
    conf, n = re.subn(r"(-r_convec\s+)[0-9.eE+-]+", lambda m: f"{m.group(1)}{r:.8f}", conf, count=1)
    assert n == 1
    lcf = open(os.path.join(tmp, "stack.lcf")).read()
    if var == "tim2":
        lcf = set_layer(lcf, 19, rho=0.5)
    elif var == "interp100":
        lcf = set_layer(lcf, 0, thickness=1.0e-4)
    elif var == "topdie":
        lcf = set_layer(lcf, 18, thickness=1.4e-4)
    elif var == "die35":
        for L in (4, 6, 8, 10, 12, 14, 16):
            lcf = set_layer(lcf, L, thickness=3.5e-5)
    elif var == "tim14":
        lcf = set_layer(lcf, 19, thickness=5.6e-5)
    elif var == "hmslike":
        lcf = set_layer(lcf, 0, thickness=1.0e-4)
        lcf = set_layer(lcf, 1, thickness=3.3e-5, rho=0.67)
        lcf = set_layer(lcf, 18, thickness=1.35e-4)
        lcf = set_layer(lcf, 19, rho=0.5)
        set_fill_k(tmp, "1.11")
    elif var == "base100":
        lcf = set_layer(lcf, 2, thickness=1.0e-4)
    elif var == "ubump1":
        lcf = set_layer(lcf, 1, rho=1.0)
    elif var == "mold1":
        set_fill_k(tmp, "1.0")
    elif var == "vc":
        conf = re.sub(r"(-k_spreader\s+)\S+", lambda m: f"{m.group(1)}4000.0", conf, count=1)
        conf = re.sub(r"(-t_spreader\s+)\S+", lambda m: f"{m.group(1)}0.003", conf, count=1)
    elif var == "sec":
        conf = re.sub(r"(-model_secondary\s+)\S+", lambda m: f"{m.group(1)}1", conf, count=1)
        conf = re.sub(r"(-r_convec_sec\s+)\S+", lambda m: f"{m.group(1)}0.5", conf, count=1)
        # substrate, solder and board larger than the 45.5 x 36.6 mm interposer (HotSpot's 21 mm defaults give negative
        # periphery resistances, HotSpot_HMS package.c:450-451)
        conf = re.sub(r"(-s_sub\s+)\S+", lambda m: f"{m.group(1)}0.06", conf, count=1)
        conf = re.sub(r"(-s_solder\s+)\S+", lambda m: f"{m.group(1)}0.06", conf, count=1)
        conf = re.sub(r"(-s_pcb\s+)\S+", lambda m: f"{m.group(1)}0.15", conf, count=1)
    elif var in ("timH14", "timG14"):        # 20 um at resistivity 0.7 (14 mm2-K/W) only over the stacks / the GPU
        import run_sim3 as S3
        hot = ("TB_L", "TB_R") if var == "timH14" else ("TB_GPU",)
        with open(os.path.join(tmp, "tim_top.flp"), "w") as f:
            for name, x0, x1, y0, y1, _ in S3.TIM_TILES:
                rr = 0.7 if name in hot else 0.25
                f.write(f"{name}\t{x1 - x0:.6f}\t{y1 - y0:.6f}\t{x0:.6f}\t{y0:.6f}\t4000000\t{rr:.6g}\n")
        o2 = []
        for blk in lcf.split("\n\n"):
            d = [ln for ln in blk.strip().split("\n") if ln and not ln.startswith("#")]
            if len(d) == 7 and int(d[0]) == 19:
                blk = blk.replace("tim.flp", "tim_top.flp")
            o2.append(blk)
        lcf = "\n\n".join(o2)
        assert lcf.count("tim_top.flp") == 1
    elif var == "core70":
        p = os.path.join(tmp, "core_and_mem_ctrl.flp")
        side = GX1 - GX0
        s_ = side * 0.70 ** 0.5
        m_ = (side - s_) / 2
        new_ = []
        for ln in open(p).read().splitlines():
            if ln.split() and ln.split()[0] == "C_0":
                assert len(ln.split()) == 5, ln
                new_ += [f"C_0\t{s_:.6f}\t{s_:.6f}\t{GX0 + m_:.6f}\t{GY0 + m_:.6f}",
                         f"C_rb\t{side:.6f}\t{m_:.6f}\t{GX0:.6f}\t{GY0:.6f}",
                         f"C_rt\t{side:.6f}\t{m_:.6f}\t{GX0:.6f}\t{GY1 - m_:.6f}",
                         f"C_rl\t{m_:.6f}\t{s_:.6f}\t{GX0:.6f}\t{GY0 + m_:.6f}",
                         f"C_rr\t{m_:.6f}\t{s_:.6f}\t{GX1 - m_:.6f}\t{GY0 + m_:.6f}"]
            else:
                new_.append(ln)
        open(p, "w").write("\n".join(new_) + "\n")
    elif var != "base":
        raise ValueError(var)
    open(os.path.join(tmp, "stack.lcf"), "w").write(lcf)
    open(os.path.join(tmp, "run.config"), "w").write(conf)
    os.rename(tmp, dst)
    return dst


def powered_flps(gdir):
    out = []
    for blk in open(os.path.join(gdir, "stack.lcf")).read().split("\n\n"):
        d = [ln for ln in blk.strip().split("\n") if ln and not ln.startswith("#")]
        if len(d) == 7 and d[2].upper() == "Y":
            out.append(d[6])
    return out


def columns(gdir):
    cols = []
    for f in powered_flps(gdir):
        for ln in open(os.path.join(gdir, f)):
            p = ln.split()
            if p and not p[0].startswith("#"):
                cols.append(p[0])
    return cols


def source_row(src, cols, m):
    v = dict.fromkeys(cols, 0.0)
    if src == "gpu":
        v["C_0"] = GPU_W
    elif src == "base":
        allb = {}
        for c in cols:
            mt = LC_RE.match(c)
            if mt:
                allb.setdefault(int(mt.group(1)), []).append(c)
        for s in range(STACKS):
            for c in allb[s]:
                v[c] += UNIT_W / len(allb[s])
    else:
        sel = {"uni": lambda b: True, "array": lambda b: b["kind"] == "ar"}[src]
        HR.B.spread(v, m["blocks"], UNIT_W, sel)
    return [v[c] for c in cols]


def run(job):
    var, k, r, src = job
    gdir = geom(var, k, r)
    name = f"{var}_k{k:g}_r{r:.5f}_{src}"
    d = os.path.join(OUT, "runs", name)
    if os.path.exists(os.path.join(d, "fields.npz")):
        return name
    os.makedirs(d, exist_ok=True)
    m = json.load(open(os.path.join(gdir, "blocks.json")))
    cols = columns(gdir)
    pt = os.path.join(d, name + ".ptrace")
    with open(pt, "w") as f:
        f.write("\t".join(cols) + "\n")
        f.write("\t".join(f"{x:.6f}".rstrip("0").rstrip(".") or "0" for x in source_row(src, cols, m)) + "\n")
    cmd = [C.HOTSPOT, "-c", "run.config", "-f", "core_and_mem_ctrl.flp", "-p", pt, "-model_type", "grid",
           "-grid_layer_file", "stack.lcf", "-detailed_3D", "on", "-steady_file", os.path.join(d, name + ".steady"),
           "-grid_steady_file", os.path.join(d, name + ".grid.steady")]
    with open(os.path.join(d, name + ".log"), "w") as lg:
        subprocess.check_call(cmd, cwd=gdir, stdout=subprocess.DEVNULL, stderr=lg)
    grid = int(re.search(r"-grid_rows\s+(\d+)", open(os.path.join(gdir, "run.config")).read()).group(1))
    off = 3 if var == "sec" else 0          # with model_secondary on, substrate, solder and board are layers 0-2
    arr = {}
    for i in [0, 2] + [4 + 2 * dd for dd in range(8)] + [19]:
        p = os.path.join(d, f"{name}.grid.steady[{i + off}]")
        arr[f"L{i}"] = np.array([float(ln.split()[1]) for ln in open(p) if ln.strip()]).reshape(grid, grid) - 273.15 - AMB
    np.savez_compressed(os.path.join(d, "fields.npz"), **arr)
    for fn in os.listdir(d):
        if ".grid.steady[" in fn or fn.endswith(".ptrace"):
            os.remove(os.path.join(d, fn))
    return name


def run_all(variants):
    spec = [j for v in variants for j in jobs_of(v)]
    for j in spec:                          # geometries first, serially (no race on the .tmp copy)
        geom(j[0], j[1], j[2])
    with ThreadPoolExecutor(max_workers=int(os.environ.get("NW", "4"))) as ex:
        for i, n in enumerate(ex.map(run, spec)):
            print(f"done {i + 1}/{len(spec)} {n}", flush=True)


# ---------------------------------------------------------------- refit
KEYS = ("a_G", "a_M", "beta_array_only", "gpu_die_per_nonmem_W", "ratio", "beta_base_die", "beta_uniform_dram")


def V_of(var, k, r=R0):
    V = {}
    for s in AMP:
        v_ = "base" if (var == "core70" and s != "gpu") else var        # core70 changes only the GPU source
        p = os.path.join(OUT, "runs", f"{v_}_k{k:g}_r{r:.5f}_{s}", "fields.npz")
        if os.path.exists(p):
            F = dict(np.load(p))
            V[s] = {a: v / AMP[s] for a, v in HR.readout(64).vec(F).items()}
    return V


def shift(V, d):
    """Uniform rise of d K per W of total power (r_convec change)."""
    return {s: {a: v + d * (1.0 if s == "gpu" else STACKS) for a, v in vv.items()} for s, vv in V.items()}


def need(x):
    return ["gpu", "array"] + (["uni"] if x < 1 else []) + (["base"] if x > 0 else [])


def fit_at_k(var, k, x, op):
    V = V_of(var, k)
    if not all(s in V for s in need(x)):
        return None
    ph = None
    if x == 0 and "base" not in V:
        V["base"], ph = V["uni"], "base"          # placeholder so mem_split(0) has the key (weight 0)
    if x == 1 and "uni" not in V:
        V["uni"], ph = V["base"], "uni"
    f = lambda d: HR.model_fit(shift(V, d), x, "cell", False, op)  # noqa: E731
    q0, q1 = f(0.0), f(0.002)
    d = 0.002 * (HR.TARGET["a_G"] - q0["a_G"]) / (q1["a_G"] - q0["a_G"])
    q = {kk: f(d).get(kk) for kk in KEYS}
    if ph:                                        # base-die / DRAM-die ratio needs both real sources
        q["beta_base_die"] = q["beta_uniform_dram"] = None
    q["d_shift"] = d
    q["r_equiv"] = R0 + d / 1.2                   # a 0.004 change of r_convec moves the field by about 0.0048 K/W
    q["k"] = k
    return q


def interp(ks, ys, kq):
    c = np.polyfit(np.log(ks), ys, min(2, len(ks) - 1))
    return float(np.polyval(c, np.log(kq)))


def solve_k(rows):
    ks = np.array([r["k"] for r in rows])
    aM = np.array([r["a_M"] for r in rows])
    tg = HR.TARGET["a_M"]
    kk = np.exp(np.linspace(np.log(ks.min()) - 0.4, np.log(ks.max()) + 0.4, 8001))
    vals = np.array([interp(ks, aM, x) for x in kk[::40]])
    i = int(np.argmin(np.abs(vals - tg)))
    lo, hi = kk[max(0, i * 40 - 40)], kk[min(len(kk) - 1, i * 40 + 40)]
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if (interp(ks, aM, mid) - tg) * (interp(ks, aM, lo) - tg) <= 0:
            hi = mid
        else:
            lo = mid
    ks_ = (lo * hi) ** 0.5
    out = {key: interp(ks, np.array([r[key] for r in rows]), ks_) for key in KEYS + ("d_shift", "r_equiv")
           if rows[0].get(key) is not None}
    out["k_star"] = ks_
    out["extrapolated"] = bool(not ks.min() <= ks_ <= ks.max())
    return out


def refit(variants, op):
    res = {}
    base_rows = {x: [fit_at_k("base", k, x, op) for k in BASE_K[x]] for x in (0, 1)}
    res["base"] = {f"x{x}": solve_k([r for r in base_rows[x] if r]) for x in (0, 1)}
    for var in variants:
        if var == "base":
            continue
        rec = {}
        for x in (0, 1):
            rows = [r for r in (fit_at_k(var, k, x, op) for k in BASE_K[x]) if r]
            if len(rows) >= 2:
                s = solve_k(rows)
                s["method"] = f"own k runs {[r['k'] for r in rows]}"
            else:
                r1 = next((r for r in rows if r["k"] == ONE_K[x]), None)
                if r1 is None:
                    continue
                b1 = next(r for r in base_rows[x] if r["k"] == ONE_K[x])
                shifted = [{key: (br[key] + (r1[key] - b1[key]) if br.get(key) is not None and r1.get(key) is not None
                                  else None) for key in KEYS + ("d_shift", "r_equiv")} | {"k": br["k"]}
                           for br in base_rows[x]]
                s = solve_k(shifted)
                s["method"] = f"base curve shifted by the variant offset at k = {ONE_K[x]}"
            rec[f"x{x}"] = s
        res[var] = rec
    return res


def thr(beta, g="H100"):
    sw = HR.J("hotspot_recal.json")["threshold_sweep"]
    b = np.array(sw["beta"])
    e = np.array([np.nan if v is None else v for v in sw[g]["timing"]])
    return float(np.interp(beta, b, e))


def summary(R):
    tgt, ci_hi = C.TARGET_RATIO, C.TARGET_RATIO_CI[1]
    out = {}
    for var, rec in R.items():
        if "x0" not in rec or "x1" not in rec:
            continue
        b0, b1 = rec["x0"]["beta_array_only"], rec["x1"]["beta_array_only"]
        q0, q1 = rec["x0"]["ratio"], rec["x1"]["ratio"]
        xs = (q0 - tgt) / (q0 - q1)
        xc = min(max(xs, 0.0), 1.0)
        xhi = min(max((q0 - ci_hi) / (q0 - q1), 0.0), 1.0)
        s = dict(description=VARIANTS[var][0], family=VARIANTS[var][1],
                 beta_x0=b0, ratio_x0=q0, beta_over_ratio_x0=b0 / q0, k_x0=rec["x0"]["k_star"], r_x0=rec["x0"]["r_equiv"],
                 gpu_die_x0=rec["x0"]["gpu_die_per_nonmem_W"], method_x0=rec["x0"].get("method", "own k runs"),
                 beta_x1=b1, ratio_x1=q1, beta_over_ratio_x1=b1 / q1, k_x1=rec["x1"]["k_star"], r_x1=rec["x1"]["r_equiv"],
                 gpu_die_x1=rec["x1"]["gpu_die_per_nonmem_W"], method_x1=rec["x1"].get("method", "own k runs"),
                 x_match_unclamped=xs, x_match=xc, beta_match_unclamped=b0 + xs * (b1 - b0), beta_match=b0 + xc * (b1 - b0),
                 x_ci_hi=xhi, beta_ci_hi=b0 + xhi * (b1 - b0), thr_x0=thr(b0))
        s["thr_match"] = thr(s["beta_match"])
        out[var] = s
    base = out.get("base")
    if base:
        for s in out.values():
            for key in ("beta_x0", "beta_match", "beta_match_unclamped", "thr_x0", "thr_match"):
                s["d_" + key] = s[key] - base[key]
    return out


def main(variants):
    op = C.op_beta()
    R = refit(variants, op)
    S = summary(R)
    res = dict(target_ratio=C.TARGET_RATIO, op_beta=op, r_convec_runs=R0, refit=R, summary=S,
               basis="beta = beta_array_only (PIM in the bank arrays, local normalization at op_beta, cell readout); "
                     "match = HotSpot row-miss ratio equal to the measured ratio, linear in x between x = 0 and x = 1")
    C.dump(res, "package_variants.json")
    for var, s in S.items():
        print(var.ljust(10), " ".join(f"{k}={v:.4f}" for k, v in s.items() if isinstance(v, float)))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["runs"]:
        run_all(a[1:] or list(VARIANTS))
    else:
        main(["base"] + [v for v in (a or list(VARIANTS)) if v != "base"])
