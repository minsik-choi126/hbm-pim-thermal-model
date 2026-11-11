"""Compare the model step response (backward Euler on the dumped HotSpot network, geom/<tag>/step.npz) with the measured
H100 GEMM steps (meas_transient.json). Model step: dP_G and dP_M of the measured step (memory split x of the tag)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
tag, x = sys.argv[1], float(sys.argv[2])
S = np.load(os.path.join(HERE, "geom", tag, "step.npz"))
t = S["t"]
# measured H100 GEMM step: idle -> GEMM duty 1 (paths_multi.json H100 idle_host 38.4 W, idle rail 36.4 W; GEMM at the
# cap host 590.6 W, rail 107.2 W)
PM = json.load(open("<analysis>/pipeline/e2e_joint/paths_multi.json"))["H100"]
dG = PM["host_gemm_tdp"] - PM["idle_host"]
dM = PM["rail_gemm_tdp"] - PM["idle_rail_per_stack"] * 5
w = np.array([dG, (1 - x) * dM / 5, x * dM / 5, 0.0])        # columns: gpu, uni, base, basephy


def series(key):
    a = S[key]                                     # (times, cells, sources)
    return np.einsum("tcs,s->tc", a, w).max(axis=1), (S["ss_" + key] @ w).max()


out = dict(tag=tag, x=x, dP_G=dG, dP_M=dM)
for key, lab in (("dram", "hbm"), ("gpu", "gpu")):
    y, ss = series(key)
    f = y / ss
    rec = dict(steady_rise=float(ss), share_at={})
    for ts in (0.01, 0.1, 0.5, 1, 2, 4, 6, 10, 20, 30, 60, 120):
        rec["share_at"][str(ts)] = float(np.interp(ts, t, f))
    i = np.searchsorted(f, 1 - np.exp(-1))
    rec["t63_s"] = float(np.interp(1 - np.exp(-1), f[max(i - 1, 0):i + 1], t[max(i - 1, 0):i + 1])) if 0 < i < len(f) else None
    out[lab] = rec
M = json.load(open(os.path.join(HERE, "meas_transient.json")))
meas = {}
for lab, key in (("hbm", "thbm"), ("gpu", "tgpu")):
    vals = {ts: [M[f][key]["share_at"][ts] for f in M if f.startswith("h100_transient_dt2")] for ts in ("2", "4", "6", "10", "20", "30", "60", "120")}
    meas[lab] = {ts: [min(v), max(v)] for ts, v in vals.items()}
out["measured_dt2_runs"] = meas
mj = os.path.join(HERE, "geom", tag, "modes.json")
if os.path.exists(mj):
    out["slow_taus_s"] = json.load(open(mj))["slow_taus_s"][:6]
json.dump(out, open(os.path.join(HERE, f"trans_cmp_{tag}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
