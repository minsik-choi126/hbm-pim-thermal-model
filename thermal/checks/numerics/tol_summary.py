"""Steady-solver tolerance check (S03): the GPU-source unit run at k = 1, r 0.01361, grid 64 solved with HotSpot's
iterative solver stopping at a maximum update of 1e-6 (this repository), 1e-8 and 1e-10 K. Reads the all-layer fields of
the audit runs (run2.py, binary hs_src/hotspot_tol = this repository's HotSpot with the stopping tolerance read from
HS_TOL) and writes tol_summary.json. The DRAM-source runs at 1e-8 / 1e-10 stopped with 'line too long' (log_tol_*.txt).

    AUDIT=<audit folder> python3 tol_summary.py
"""
import json
import os

import numpy as np

R = os.path.join(os.environ.get("AUDIT", "."), "numerics", "runs2")
base = "a50_par_k1_r0.01361_g64_gpu_tol"
F = {t: np.load(os.path.join(R, base + t, "fields.npz")) for t in ("1e-06", "1e-08", "1e-10")}
EB = {t: json.load(open(os.path.join(R, base + t, "eb.json"))) for t in F}
ref = F["1e-10"]
out = dict(run=base + "*", source="GPU die C_0, 1000 W", layers=len(ref.files),
           max_rise_K=float(max(ref[k].max() for k in ref.files)),
           max_abs_diff_vs_1e10_K={t: float(max(np.abs(F[t][k] - ref[k]).max() for k in ref.files)) for t in ("1e-06", "1e-08")},
           iterations={t: EB[t]["iters"] for t in F}, energy_balance_err_pct={t: EB[t]["err_pct"] for t in F},
           note="HotSpot prints 0.01 K; a difference of 0.01 K is one output step")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tol_summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
