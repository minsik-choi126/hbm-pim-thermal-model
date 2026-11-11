"""Step 5: the heat factor beta on the recalibrated model -> out/beta.json.

beta (model/beta.py) is the memory power that raises the peak DRAM temperature as much as the PIM power does, divided by
the PIM power, at the GPU+PIM operating point of the H100 threshold (no power constraint) that this beta gives.

Direct HotSpot runs at the x = 0 point of the recalibration family (all memory power in the DRAM dies; k and r_convec
from out/recal.json solution.cell.x0, rounded to 5 decimals), grid 64:
    every bank      PIM in the bank arrays with the MAC share at the column decoders: the paper's beta; range over the
                    four bank floorplans (array share 50 / 75 %, stripe parallel / perpendicular) x MAC share 0.9-30 %
    channel subsets the same PIM power in half or a quarter of the channels (power.SUBSETS), floorplans a50 / a75 par
    grid 128        every bank on a50_par (full mode)
The x = 0 value from the family (step 4, unit responses quadratic in r) must agree with the direct run (5e-3).
Also collected here from step 4: beta at the matched split, x = 1, the allowed range, the reference prior, the
GPU-side base-die variant and the die-average readouts; and the same beta at the H200 and B200 operating points.
Comparison only: f_bank (step 3, k = 1, fixed base point, PIM rise over uniform DRAM-die power of the same
amplitude) and the local slope beta of step 4 at the fixed operating point.

    python -m thermal.steps.beta [--grids 64,128]
"""
import json
import sys
import time

from .. import paths
from ..model import beta as B
from ..model import geom as G
from ..model import hotspot as HS
from ..model import power as P
from . import recal as RC
from .bank import SPAN_ES

FPS = ("a50_par", "a75_par", "a50_perp", "a75_perp")
CORE = ("gpu", "uni", "base")
EVERY = RC.PIM_MAPS["every_bank"]
FAMILY_TOL = 5e-3


def x0_point():
    x0 = json.load(open(paths.out("recal.json")))["solution"]["cell"]["x0"]
    return round(x0["k"], 5), round(x0["r_convec"], 5)


def name(k, r, grid, fp, src):
    return f"{G.tag(k, r, grid, fp)}_{src}"


def sources(fp, grid):
    if grid != 64:
        return CORE + ("array", "edge") if fp == "a50_par" else ()
    return (CORE if fp == "a50_par" else ()) + ("array", "edge") + (P.SRC_SUB if fp in ("a50_par", "a75_par") else ())


def jobs(k, r, grids):
    out = []
    for grid in grids:
        for fp in FPS:
            srcs = sources(fp, grid)
            if not srcs:
                continue
            g = G.variant(k, r, grid, fp)
            m, cols = G.load_blocks(g), G.columns(g)
            out += [(name(k, r, grid, fp, s), g, [P.source_row(s, cols, m)], False) for s in srcs]
    return out


def vset(k, r, grid, fp):
    """Unit responses: GPU, memory and base die from a50_par, PIM sources from fp; None if a run is missing."""
    want = {s: "a50_par" for s in CORE}
    want.update({s: fp for s in sources(fp, grid) if s not in CORE})
    if not all(HS.done(name(k, r, grid, f, s)) for s, f in want.items()):
        return None
    return {s: RC.unit_vec(k, r, s, f, grid) for s, f in want.items()}


def main(grids=(64,)):
    t0 = time.time()
    k, r = x0_point()
    hs = HS.run_all(jobs(k, r, grids), label="beta")
    target, _, _ = RC.targets()
    V = vset(k, r, 64, "a50_par")
    mf = RC.model_fit(V, 0.0)
    eb = B.fixed_point(V, 0.0, EVERY)
    out = dict(definition=B.DEFINITION, operating_points=B.SOURCE, model=dict(k=k, r_convec=r, x=0.0, grid=64),
               slopes=dict(a_G=mf["a_G"], a_M=mf["a_M"], ratio=mf["ratio"], target_a_G=target["a_G"], target_a_M=target["a_M"]),
               every_bank=eb, every_bank_gpus={g: B.fixed_point(V, 0.0, EVERY, g=g) for g in B.GPUS})
    fr, sub = {}, {}
    for fp in FPS:
        Vf = vset(k, r, 64, fp)
        for e in SPAN_ES:
            fr[f"{fp}:e{e:.3f}"] = B.fixed_point(Vf, 0.0, {"array": 1 - e, "edge": e})["beta"]
        for s in (P.SRC_SUB if fp in ("a50_par", "a75_par") else ()):
            sub[f"{fp}:{s}"] = B.fixed_point(Vf, 0.0, {s: 1.0})
    out["floorplan_mac"], out["subsets"] = fr, sub
    half = [v["beta"] for key, v in sub.items() if "half" in key]
    quarter = [v["beta"] for key, v in sub.items() if "quarter" in key]
    if 128 in grids:
        out["grid128"] = B.fixed_point(vset(k, r, 128, "a50_par"), 0.0, EVERY)
    rec = json.load(open(paths.out("recal.json")))
    s4 = rec["summary"]
    fam_x0 = s4["x0"]["beta"]
    out["check_family_x0"] = dict(family=fam_x0, direct=eb["beta"], rel=fam_x0 / eb["beta"] - 1, tol=FAMILY_TOL)
    assert abs(fam_x0 / eb["beta"] - 1) < FAMILY_TOL, out["check_family_x0"]
    try:
        bank = json.load(open(paths.out("bank.json")))["summary"]
    except (OSError, KeyError, ValueError):
        bank = None
    e_x = eb["e_x"]
    PG, M, PP = B.op_at("H100", e_x)
    da = s4["die_average"]
    out["summary"] = dict(
        every_bank=eb["beta"], every_bank_range=[min(fr.values()), max(fr.values())], half=[min(half), max(half)],
        quarter=[min(quarter), max(quarter)], every_bank_grid128=out["grid128"]["beta"] if "grid128" in out else None,
        every_bank_by_gpu={g: v["beta"] for g, v in out["every_bank_gpus"].items()},
        matched=dict(x=s4["best"]["x"], beta=s4["best"]["beta"], by_gpu=dict(H100=s4["best"]["beta"], H200=s4["best"]["beta_H200"],
                                                                              B200=s4["best"]["beta_B200"]),
                     half=s4["best"]["beta_half"], quarter=s4["best"]["beta_quarter"]),
        x1=s4["x1"]["beta"], allowed=s4["allowed"]["beta"], basephy_matched=s4["basephy_target"]["beta"],
        die_average=[min(da["bottom"]["beta"], da["base"]["beta"]), max(da["bottom"]["beta"], da["base"]["beta"])],
        prior=dict(beta=s4["prior"]["beta"], range=s4["prior_range"]["beta"]),
        threshold_H100_e_x=e_x, thresholds_e_x={g: B.threshold(g, eb["beta"]) for g in B.GPUS},
        thresholds_e_x_at_1p11={g: B.threshold(g, 1.11) for g in B.GPUS},
        operating_point=dict(gpu="H100", e_pJ_bit=e_x, P_G=PG, memory_heat=M, P_P=PP,
                             P_M=float(B.gpu("H100")["m0"] + (M - B.gpu("H100")["m0"]) / B.gpu("H100")["phi"])),
        comparison=dict(
            note="comparison definitions, not used: f_bank of step 3 (k = 1, r_convec 0.01361, GPU 80 W, DRAM 23.9 W/stack, "
                 "PIM 40 W/stack; PIM rise over the rise of the same W spread uniformly over the DRAM dies) and the local "
                 "slope of step 4 (PIM rise per W over the +-20 W memory slope at P_G 332.7 W, P_M 80.1 W, P_P 52.1 W)",
            f_bank_k1=None if bank is None else bank["f_bank"], f_bank_k1_range=None if bank is None else bank["f_bank_range"],
            half_k1=None if bank is None else bank["half"], quarter_k1=None if bank is None else bank["quarter"],
            local_x0=s4["x0"]["beta_local"], local_matched=s4["best"]["beta_local"], local_x1=s4["x1"]["beta_local"],
            local_allowed=s4["allowed"]["beta_local"],
            local_die_average=[da["bottom"]["beta_local"], da["base"]["beta_local"]],
            beta_at_fixed_e_x0=eb["beta_fixed_e"]))
    out["seconds"] = round(time.time() - t0, 1)
    out["hotspot_seconds"] = round(hs, 1)
    json.dump(out, open(paths.out("beta.json"), "w"), indent=1, default=float)
    s = out["summary"]
    print(f"[beta] every bank at x = 0: {s['every_bank']:.4f} (floorplans x MAC {s['every_bank_range'][0]:.3f}-"
          f"{s['every_bank_range'][1]:.3f}; half {s['half'][0]:.3f}-{s['half'][1]:.3f}; quarter {s['quarter'][0]:.3f}-"
          f"{s['quarter'][1]:.3f}); H100 threshold {e_x:.3f} pJ/bit; matched {s['matched']['beta']:.4f}; x = 1 {s['x1']:.4f}; "
          f"allowed {s['allowed'][0]:.3f}-{s['allowed'][1]:.3f}; die average {s['die_average'][0]:.3f}-{s['die_average'][1]:.3f} "
          f"({out['seconds']} s)", flush=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    main(tuple(int(x) for x in a[a.index("--grids") + 1].split(",")) if "--grids" in a else (64,))
