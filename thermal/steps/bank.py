"""Step 3: PIM heat at bank resolution on the calibrated model (k = 1, r_convec from step 2) -> out/bank.json.

Unit responses on the bank floorplans (geom.py), superposed at the base point of the placement study: GPU 80 W, memory
23.9 W per stack spread over the DRAM dies, PIM 40 W per stack on top.

  heat factor f   comparison only, not the model's beta (model/beta.py, step 5): the "f_bank" normalisation,
                  rise of the hottest DRAM grid cell per W of PIM over that of
                  the same W spread uniformly over the DRAM dies, at the base point above (k = 1, x = 0). PIM at every bank: (1 - e) in the bank arrays, e in the column-decoder strips (MAC units),
                  e = 3.5 % from the AttAcc energy table. f_bank = a50_par, e = 3.5 %; range over bank-array share
                  50 / 75 %, both stripe orientations and e = 0.9 - 30 %.
  channel subsets the same PIM power in half or a quarter of the channels (power.SUBSETS)
  base die        rise per W in the base die over that per W in the DRAM dies
  ripple          decode step with PIM on only during the PIM phase (data/inputs/ripple_phases.json): HotSpot step
                  responses (0.1 ms rows) of the GPU die, uniform memory power, bank arrays and column-decoder strips,
                  fitted as exponential modes; periodic steady state of per-layer PIM bursts and of one lump per step;
                  hottest DRAM block against the steady state of the step-average power. Adiabatic bound for the
                  temperature swing inside one burst.

    python -m thermal.steps.bank [--grids 64,128] [--rows 1000]
"""
import json
import os
import sys
import time

import numpy as np

from .. import paths
from ..model import geom as G
from ..model import hotspot as HS
from ..model import power as P
from ..model import readout as R
from ..model.modal import modal_fit, pss_t
from .calibrate import r_convec

K = 1.0
# DRAM background 23.9022 W per stack: constant of the original runs; data/inputs/dram_map_decode.json sums to 23.8987 W
# (+0.015 %, effect on f below 1e-4). f is a ratio of two rises in a linear model, so the base powers matter only through
# the location of the hottest cell.
BASE_GPU_W, BASE_DRAM_W, PIM_W = 80.0, 23.9022, 40.0
ES = (0.0, 0.009, P.E_TABLE, 0.10, 0.30, 0.50, 1.0)
SPAN_ES = (0.009, P.E_TABLE, 0.10, 0.30)
FP_SOURCES = {"a50_par": ("gpu", "uni", "base", "array", "edge", "qfs") + P.SRC_SUB, "a75_par": ("array", "edge") + P.SRC_SUB,
              "a50_perp": ("array", "edge"), "a75_perp": ("array", "edge")}
RIPPLE_FPS = ("a50_par", "a75_par")
STEP_SRC = ("gpu", "uni", "array", "edge")
CHECK_ROWS = 40


def src_fp(fp, src):
    return "a50_par" if src in P.FP_FREE or src == "qfs" else fp


def unit_name(r, grid, fp, src):
    return f"{G.tag(K, r, grid, src_fp(fp, src))}_{src}"


def steady_jobs(r, grids):
    jobs = []
    for grid in grids:
        for fp, srcs in FP_SOURCES.items():
            g = G.variant(K, r, grid, fp)
            m, cols = G.load_blocks(g), G.columns(g)
            jobs += [(unit_name(r, grid, fp, s), g, [P.source_row(s, cols, m)], False) for s in srcs]
    return jobs


def step_name(r, fp, src, n, grid=64):
    return f"step_{G.tag(K, r, grid, fp)}_{src}_n{n}"


def step_jobs(r, rows, check=False):
    jobs = []
    for fp in RIPPLE_FPS:
        g = G.variant(K, r, 64, fp)
        m, cols = G.load_blocks(g), G.columns(g)
        n = rows[fp]
        jobs += [(step_name(r, fp, s, n), g, [P.source_row(s, cols, m)] * n, True) for s in STEP_SRC]
    if check:                                  # fast-response check of the bank-array step on grid 128
        g = G.variant(K, r, 128, "a50_par")
        m, cols = G.load_blocks(g), G.columns(g)
        jobs.append((step_name(r, "a50_par", "array", CHECK_ROWS, 128), g, [P.source_row("array", cols, m)] * CHECK_ROWS, True))
    return jobs


def unit(r, grid, fp, src):
    """Grid rise per W (GPU) or per W per stack (the rest) of one source."""
    F = HS.fields(unit_name(r, grid, fp, src))
    a = P.amplitude(src)
    return {k: v / a for k, v in F.items()}


def base_field(u):
    return R.combine([(BASE_GPU_W, u["gpu"]), (BASE_DRAM_W, u["uni"])])


# ---------------------------------------------------------------- placement
def placement(r, grid, fp):
    ro = R.readout(grid)
    m = G.load_blocks(G.variant(K, r, grid, fp))
    srcs = set(FP_SOURCES[fp]) | {"gpu", "uni"}
    u = {s: unit(r, grid, fp, s) for s in srcs}
    Fb = base_field(u)
    R0 = ro.hottest(Fb, m)

    def rise(parts):
        h = ro.hottest(R.combine([(1.0, Fb)] + [(PIM_W * w, u[s]) for s, w in parts]), m)
        return {k: (h[k] - R0[k]) / PIM_W for k in h}
    U = rise([("uni", 1.0)])
    cases = {f"bank_e{e:.3f}": rise([("array", 1 - e), ("edge", e)]) for e in ES}
    names = ["array", "edge"] + [s for s in ("qfs", "base") + P.SRC_SUB if s in srcs]
    cases.update({("base_die" if s == "base" else s): rise([(s, 1.0)]) for s in names})
    return dict(uniform_per_W=U, base_point=dict(gpu_W=BASE_GPU_W, dram_W_per_stack=BASE_DRAM_W, pim_W_per_stack=PIM_W),
                cases={nm: dict(rise_per_W=c, f={k: c[k] / U[k] for k in ("cell", "coarse", "block")}) for nm, c in cases.items()})


# ---------------------------------------------------------------- ripple
class Ripple:
    def __init__(self, r, fp, n):
        self.m = G.load_blocks(G.variant(K, r, 64, fp))
        self.resp, self.a, self.amp = {}, {}, {}
        head = None
        for k in STEP_SRC:
            nm = step_name(r, fp, k, n)
            h, arr = HS.ttrace(nm)
            assert head is None or h == head
            head = h
            self.resp[k] = np.vstack([np.zeros((1, arr.shape[1])), arr - G.AMB])
            T = HS.steady(nm)
            self.amp[k] = P.amplitude(k)
            self.a[k] = (np.array([T.get(self.key(x), np.nan) for x in head]) - G.AMB) / self.amp[k]
        self.head = head
        self.dram = [i for i, x in enumerate(head) if x in self.m["info"]]
        self.set_cut(n)

    def key(self, x):
        return f"layer_{self.m['info'][x]['layer']}_{x}" if x in self.m["info"] else f"layer_2_{x}"

    def set_cut(self, nmax):
        self.c, self.fit = {}, {}
        for k in STEP_SRC:
            c, rms, mx = modal_fit(self.resp[k], self.a[k] * self.amp[k], self.dram, nmax)
            self.c[k], self.fit[k] = c / self.amp[k], (rms, mx)

    def case(self, a_ms, b_ms, on, off, e, nt=2000):
        """on / off = (GPU W, memory heat W per stack, PIM W per stack); PIM in the bank map with MAC share e."""
        d = a_ms / (a_ms + b_ms)
        avg = [d * x + (1 - d) * y for x, y in zip(on, off)]
        if e == "uniform":
            aP, cP = self.a["uni"], self.c["uni"]
        else:
            aP = (1 - e) * self.a["array"] + e * self.a["edge"]
            cP = (1 - e) * self.c["array"] + e * self.c["edge"]
        Tss = G.AMB + avg[0] * self.a["gpu"] + avg[1] * self.a["uni"] + avg[2] * aP
        t = np.concatenate([np.linspace(0, a_ms, nt // 4, endpoint=False), np.linspace(a_ms, a_ms + b_ms, nt, endpoint=False)])
        y = ((on[0] - off[0]) * pss_t(self.c["gpu"], a_ms, b_ms, t) + (on[1] - off[1]) * pss_t(self.c["uni"], a_ms, b_ms, t)
             + (on[2] - off[2]) * pss_t(cP, a_ms, b_ms, t))
        T = Tss[None, self.dram] + y[:, self.dram]
        hot = T.max(axis=1)
        ref = Tss[self.dram].max()
        return dict(max_above_step_avg=float(hot.max() - ref), min_below_step_avg=float(hot.min() - ref),
                    peak_to_peak=float(hot.max() - hot.min()), t_peak_ms=float(t[int(np.argmax(hot))]), step_avg_T_hot_model=float(ref))


def adiabatic(P_on, t_on_ms, r, fp, e):
    """Upper bound on the swing within one burst: all burst energy kept in the DRAM-die silicon under the bank arrays
    (or the column-decoder strips), no conduction."""
    m = G.load_blocks(G.variant(K, r, 64, fp))
    one = [b for b in m["blocks"] if b["stack"] == 0 and b["die"] == 0]
    A_ar = sum(b["w"] * b["h"] for b in one if b["kind"] == "ar")
    A_cd = sum(b["w"] * b["h"] for b in one if b["kind"] == "cd")
    C_si = G.SI_CP * 5e-5                                            # J / m2 K, 50 um DRAM die
    q_ar = (1 - e) * P_on / 8 / A_ar
    q_cd = e * P_on / 8 / A_cd if e else 0.0
    return dict(array=q_ar * t_on_ms * 1e-3 / C_si, edge=q_cd * t_on_ms * 1e-3 / C_si, A_array_mm2=A_ar * 1e6, A_edge_mm2=A_cd * 1e6)


def ripple(r, rows):
    ph = json.load(open(paths.inputs("ripple_phases.json")))
    t, ton, L = ph["step_ms"], ph["pim_on_ms"], ph["layers"]
    waves = (("per-layer bursts", ton / L, (t - ton) / L), ("lump: all PIM time in one block", ton, t - ton))
    out = dict(phases={k: ph[k] for k in ("step_ms", "pim_on_ms", "layers", "pim_W_per_stack_on", "on", "off")},
               pim_on_per_layer_us=1e3 * ton / L, floorplans={})
    for fp in RIPPLE_FPS:
        n = rows[fp]
        if not all(HS.done(step_name(r, fp, s, n)) for s in STEP_SRC):
            continue
        Rp = Ripple(r, fp, n)
        rec = dict(window_ms=n * G.DT_MS, cases=[])
        for e in ("uniform", P.E_TABLE, 0.10, 0.30):
            for lab, a_ms, b_ms in waves:
                rec["cases"].append(dict(label=lab, pe_share=e, on_ms=a_ms, off_ms=b_ms, **Rp.case(a_ms, b_ms, ph["on"], ph["off"], e)))
        rec["adiabatic_burst_bound"] = {f"e{e:.3f}": adiabatic(ph["pim_W_per_stack_on"], ton / L, r, fp, e) for e in (P.E_TABLE, 0.30)}
        Rp.set_cut(n // 2)                     # window check: modes fitted on the first half only
        rec["half_window_max_diff"] = max(abs(Rp.case(c["on_ms"], c["off_ms"], ph["on"], ph["off"], c["pe_share"])[k] - c[k])
                                          for c in rec["cases"] for k in ("max_above_step_avg", "peak_to_peak"))
        Rp.set_cut(n)
        rec["modal_fit_residual_rms_max_K"] = Rp.fit
        out["floorplans"][fp] = rec
    bank = [(fp, c) for fp, v in out["floorplans"].items() for c in v["cases"] if c["pe_share"] != "uniform"]
    if bank:
        layer = [c["max_above_step_avg"] for _, c in bank if c["label"] == "per-layer bursts"]
        lump = [c["max_above_step_avg"] for _, c in bank if c["label"].startswith("lump")]
        burst = [c["max_above_step_avg"] + max(out["floorplans"][fp]["adiabatic_burst_bound"][f"e{P.E_TABLE:.3f}"][x] for x in ("array", "edge"))
                 for fp, c in bank if c["label"] == "per-layer bursts" and c["pe_share"] == P.E_TABLE]
        out["summary"] = dict(per_layer_max_above_avg=max(layer), per_layer_plus_burst_max=max(burst), lump_max_above_avg=max(lump))
    return out


def fast_check(r):
    """Hottest DRAM block of the bank-array step per W per stack over the first 4 ms, grid 64 against grid 128."""
    out = {}
    pre = f"step_{G.tag(K, r, 64, 'a50_par')}_array_n"
    g64 = sorted(x for x in os.listdir(os.path.join(paths.RUNS, "runs")) if x.startswith(pre) and HS.done(x))
    for grid, nm in ((64, g64[0] if g64 else None), (128, step_name(r, "a50_par", "array", CHECK_ROWS, 128))):
        if nm is None or not HS.done(nm):
            return None
        m = G.load_blocks(G.variant(K, r, grid, "a50_par"))
        head, T = HS.ttrace(nm)
        idx = [i for i, x in enumerate(head) if x in m["info"]]
        out[f"grid{grid}"] = [float((T[i, idx].max() - G.AMB) / P.UNIT_W) for i in range(CHECK_ROWS)]
    out["ratio_128_over_64"] = [b / a for a, b in zip(out["grid64"], out["grid128"])]
    return out


# ---------------------------------------------------------------- summary
def f_bank(place, grid):
    main = place[f"a50_par_g{grid}"]["cases"][f"bank_e{P.E_TABLE:.3f}"]["f"]["cell"]
    span = {f"{fp}_g{grid}:e{e:.3f}": place[f"{fp}_g{grid}"]["cases"][f"bank_e{e:.3f}"]["f"]["cell"]
            for fp in FP_SOURCES for e in SPAN_ES if f"{fp}_g{grid}" in place}
    sub = lambda names: [place[f"{fp}_g{grid}"]["cases"][s]["f"]["cell"] for fp in ("a50_par", "a75_par") for s in names]  # noqa: E731
    half, quarter = sub(("half_far", "half_dies")), sub(("quarter_far", "quarter_dies"))
    a50 = place[f"a50_par_g{grid}"]["cases"]
    return dict(grid=grid, f_bank=main, f_bank_range=[min(span.values()), max(span.values())], range_over=span,
                half=[min(half), max(half)], quarter=[min(quarter), max(quarter)],
                quarter_block_cell=a50["qfs"]["f"]["cell"], quarter_block_coarse=a50["qfs"]["f"]["coarse"],
                base_die_over_dram=a50["base_die"]["f"]["cell"])


def main(grids=(64,), rows=None, check=False):
    t0 = time.time()
    rows = rows or {"a50_par": 300, "a75_par": 300}
    r = r_convec(K)
    hs = HS.run_all(step_jobs(r, rows, check) + steady_jobs(r, grids), label="bank")
    place = {f"{fp}_g{grid}": placement(r, grid, fp) for grid in grids for fp in FP_SOURCES}
    out = dict(model=dict(k=K, r_convec=r, grids=list(grids), pe_share_table=P.E_TABLE),
               summary=f_bank(place, max(grids)), placement=place)
    if len(grids) > 1:
        out["summary_by_grid"] = {g: f_bank(place, g) for g in grids}
    out["ripple"] = ripple(r, rows)
    if check:
        out["ripple"]["fast_response_grid_check"] = fast_check(r)
    out["seconds"] = round(time.time() - t0, 1)
    out["hotspot_seconds"] = round(hs, 1)
    json.dump(out, open(paths.out("bank.json"), "w"), indent=1)
    s, rp = out["summary"], out["ripple"].get("summary", {})
    print(f"[bank] grid {s['grid']}: f_bank {s['f_bank']:.4f} (range {s['f_bank_range'][0]:.3f}-{s['f_bank_range'][1]:.3f}), "
          f"half {s['half'][0]:.3f}-{s['half'][1]:.3f}, quarter {s['quarter'][0]:.3f}-{s['quarter'][1]:.3f}, "
          f"base die {s['base_die_over_dram']:.3f}; ripple per layer {rp.get('per_layer_max_above_avg', float('nan')):.4f} K "
          f"({out['seconds']} s)", flush=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    grids = tuple(int(x) for x in a[a.index("--grids") + 1].split(",")) if "--grids" in a else (64,)
    n = int(a[a.index("--rows") + 1]) if "--rows" in a else 300
    main(grids, {"a50_par": n, "a75_par": 300}, check=128 in grids)
