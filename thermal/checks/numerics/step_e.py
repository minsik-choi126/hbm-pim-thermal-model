"""Record (HotSpot audit 2026-11-11): ran in <audit>/numerics with the analysis repository's model; see ../README.md."""
import json, sys
from analyze import *
grid = int(sys.argv[1]) if len(sys.argv) > 1 else 64
o = op()
out = {}
for p in ("x0", "best"):
    V = vset(p, grid, 1e-6)
    x = POINTS[p]["x"]
    rec = {"readout": {}, "fd_step": {}}
    for ro in ("cell", "coarse", "bottom", "base"):
        m = metrics(V, x, ro, o)
        rec["readout"][ro] = {k: m[k] for k in ("a_G", "a_M", "ratio", "beta_every_bank", "beta_every_bank_reg", "beta_every_bank_amp", "base_over_dram")}
    for dM in (0.5, 5.0, 10.0, 20.0, 40.0, 60.0):
        oo = dict(o, dM=dM)
        m = metrics(V, x, "cell", oo)
        rec["fd_step"][f"dM{dM:g}"] = {k: m[k] for k in ("a_M_local", "beta_every_bank")}
    # operating-point sensitivity: P_P and P_G
    rec["op"] = {}
    for lab, mod in (("P_P_half", dict(P_P=o["P_P"] / 2)), ("P_P_double", dict(P_P=o["P_P"] * 2)), ("P_G_low", dict(P_G=150.0)), ("P_G_high", dict(P_G=500.0)),
                     ("P_M_low", dict(P_M=40.0)), ("P_M_high", dict(P_M=140.0))):
        m = metrics(V, x, "cell", dict(o, **mod))
        rec["op"][lab] = {k: m[k] for k in ("a_M_local", "beta_every_bank", "beta_every_bank_amp")}
    out[p] = rec
    print(p, json.dumps(rnd(rec, 4)))
json.dump(rnd(dict(op=o, grid=grid, res=out), 5), open(f"out_e_readout_fd_g{grid}.json", "w"), indent=1)
