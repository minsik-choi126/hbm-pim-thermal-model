"""Record (HotSpot audit 2026-11-11): ran in <audit>/numerics with the analysis repository's model; see ../README.md."""
import json
from analyze import *
J = json.load(open("<analysis>/pipeline/e2e_joint/hotspot_recal.json"))
out = {}
for p, key in (("x0", "x0"), ("best", "target_ratio")):
    V = vset(p, 64, 1e-6)
    x = POINTS[p]["x"]
    m = metrics(V, x)
    ref = J["solution"]["cell"][key]
    out[p] = dict(direct=m, interpolated={k: ref[k] for k in ("a_G", "a_M", "ratio", "beta_every_bank", "a_M_local", "beta_base_die", "beta_uniform_dram")})
    print(p, json.dumps(rnd(out[p], 5)))
json.dump(rnd(out, 6), open("out_step1_direct_vs_interp.json", "w"), indent=1)
