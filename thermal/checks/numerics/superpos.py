"""(b) Superposition: one HotSpot run of a combined power map (GPU 332.7 W, memory 80.1 W, PIM 52.1 W, decode operating
point of hotspot_recal.op_beta) against the sum of the unit runs scaled to the same powers, every layer (L0..L21), grid 64,
at the x0 and best points. Also the readouts (cell max, coarse) of both."""
import json
import os

import numpy as np

from analyze import HR, POINTS, RUNS2, rnd
from run2 import COMBOS


def raw(point, src, grid=64):
    pt = POINTS[point]
    return dict(np.load(os.path.join(RUNS2, f"{HR.gtag(pt['k'], pt['r'], grid)}_{src}_tol1e-06", "fields.npz")))


def main():
    out = {}
    for point, combo in (("x0", "opmix"), ("best", "opmix_best")):
        C = raw(point, combo)
        S = None
        for s, w in COMBOS[combo].items():
            amp = HR.GPU_W if s == "gpu" else HR.UNIT_W
            F = raw(point, s)
            f = {k: v * w / amp for k, v in F.items()}
            S = f if S is None else {k: S[k] + f[k] for k in S}
        d = {k: float(np.abs(C[k] - S[k]).max()) for k in C}
        ro = HR.readout(64)
        vc, vs = ro.vec(C), ro.vec(S)
        eb = json.load(open(os.path.join(RUNS2, f"{HR.gtag(POINTS[point]['k'], POINTS[point]['r'], 64)}_{combo}_tol1e-06", "eb.json")))
        out[point] = dict(combo=COMBOS[combo], max_abs_diff_K=max(d.values()), worst_layer=max(d, key=d.get),
                          peak_dram_rise_combo_K=float(vc["cell"].max()), peak_dram_rise_sum_K=float(vs["cell"].max()),
                          coarse_combo_K=float(vc["coarse"].max()), coarse_sum_K=float(vs["coarse"].max()),
                          rel_diff_peak=float(vc["cell"].max() / vs["cell"].max() - 1), combo_energy_balance_pct=eb["err_pct"])
        print(point, json.dumps(rnd(out[point], 5)))
    json.dump(rnd(out, 6), open("out_superpos.json", "w"), indent=1)


if __name__ == "__main__":
    main()
